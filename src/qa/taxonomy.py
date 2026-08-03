"""Stage 1: induce a candidate error taxonomy from dev-split mispredictions.

Pipeline (keyword-grounded clustering):
  1. Select mispredicted dev cases for the configured models.
  2. Per case, the judge writes an RCA in which EVERY error carries a short
     snake_case KEYWORD naming the error pattern.
  3. Unique keywords (with counts + example root causes) are clustered by the
     judge into candidate classes (target set in STAGE1_CLUSTER_PROMPT), purely
     data-driven (no seeding from prior analyses, so convergence with earlier
     results is evidence, not circularity).
  4. Counting is DETERMINISTIC, not generative: each error's keyword maps to its
     cluster, and we count errors and distinct cases per cluster per model
     (-> cluster_counts.csv). Classes therefore carry known mass before anyone
     commits to them.

The output `taxonomy_candidate.json` is a STARTING POINT. A human (ideally a
clinician) must review, rename, merge, and freeze it into `taxonomy.json` before
stage 2 quantifies anything -- that human step is what makes the classes defensible
rather than "whatever the model clustered". We never let stage 2 run on the
un-reviewed candidate (the driver checks for an `approved: true` flag).
"""
from __future__ import annotations

import asyncio
import json
import os
from typing import List

import pandas as pd

import re
from collections import Counter, defaultdict

from src.qa import data as qa_data, wandb_log
from src.qa.cache import ResponseCache, chunked, text_fingerprint
from src.qa.client import batch_complete, resolve_model_id
from src.qa.prompts import STAGE1_RCA_PROMPT, STAGE1_CLUSTER_PROMPT
from src.qa.schemas import validate_stage1, extract_json_object

PREF = qa_data.PRED_PREFIX


def _rca_prompt(row: pd.Series) -> str:
    return STAGE1_RCA_PROMPT.format(
        prompt=row[f"{PREF}_prompt"],
        predicted_output=row[f"{PREF}_text"],
        labels=list(row["ICD_CODES"]),
        discharge_note=row["discharge_note"],
    )


async def run_stage1(cfg: dict, repo_root: str, base: str) -> None:
    qa_cfg = cfg["QA"]
    out_dir = os.path.join(repo_root, qa_cfg["output_dir"])
    os.makedirs(out_dir, exist_ok=True)
    run = wandb_log.start(qa_cfg, "qa-stage1-taxonomy")

    # 1. Gather cases across the configured stage-1 models.
    round_name = qa_cfg["round"]
    data_root = os.path.join(repo_root, qa_cfg.get("data_root", "data"))
    frames = []
    for name in qa_cfg["stage1"]["models"]:
        df = qa_data.load_eval_results(repo_root, data_root, round_name, name)
        df = qa_data.select_cases(
            df,
            only_mispredicted=True,
            sample=qa_cfg["stage1"].get("sample_per_model"),
            seed=qa_cfg.get("seed", 42),
        )
        frames.append(df)
    cases = pd.concat(frames, ignore_index=True)
    print(f"Stage 1: {len(cases)} mispredicted cases for RCA")

    model_id = await resolve_model_id(
        base, qa_cfg["server"].get("ready_timeout_min", 45))

    # 2. Per-case free-text RCA -- checkpointed: cached (parseable) responses are
    # reused, missing ones are requested in chunks with a cache flush after each,
    # so reruns and crashes only pay for what isn't done yet.
    prompts = [_rca_prompt(r) for _, r in cases.iterrows()]
    keys = [ResponseCache.key(r["model_name"], r["hadm_id"]) for _, r in cases.iterrows()]
    # Fingerprinted by the RCA prompt text itself: editing STAGE1_RCA_PROMPT (e.g.
    # removing/changing the keyword-style examples) automatically starts a fresh
    # cache instead of silently reusing responses generated under the old prompt.
    rca_fp = text_fingerprint(STAGE1_RCA_PROMPT)
    cache = ResponseCache(os.path.join(out_dir, "cache", f"stage1_rca_{rca_fp}.parquet"))
    max_tokens = qa_cfg["stage1"].get("max_tokens", 2500)
    chunk_size = qa_cfg.get("checkpoint_every", 500)

    raw: List = [cache.get(k) for k in keys]
    rcas = [validate_stage1(t) if t else None for t in raw]
    todo = [i for i, r in enumerate(rcas) if r is None]
    print(f"Stage 1: {len(cases) - len(todo)} cases from checkpoint, "
          f"{len(todo)} to request")

    async def request(indices: List[int], tokens: int, seed_shift: int, desc: str):
        res = await batch_complete(
            [prompts[i] for i in indices], base, model_id,
            temperature=qa_cfg["stage1"].get("temperature", 0.4),
            max_tokens=tokens,
            seed=qa_cfg.get("seed", 42) + seed_shift,
            concurrency=qa_cfg.get("concurrency", 32),
            desc=desc,
        )
        for i, t in zip(indices, res):
            if t is not None:
                raw[i] = t
                rcas[i] = validate_stage1(t)
                if rcas[i] is not None:  # only checkpoint what parses
                    cache.put(keys[i], t)
        cache.flush()

    for n, block in enumerate(chunked(todo, chunk_size), 1):
        await request(block, max_tokens, 0,
                      f"stage1-rca {n}/{(len(todo) - 1) // chunk_size + 1}")

    # Re-ask parse failures once with a bigger budget (covers responses that ran
    # out of tokens mid-JSON); request failures were already retried in the client.
    redo = [i for i, (t, r) in enumerate(zip(raw, rcas)) if t is not None and r is None]
    if redo:
        print(f"Stage 1: re-asking {len(redo)} unparseable RCAs with larger budget")
        await request(redo, max_tokens + 1500, 1, "stage1-rca-redo")

    ok = [r for r in rcas if r is not None]
    print(f"Stage 1: {len(ok)}/{len(rcas)} RCAs parsed")
    _dump_parse_failures(raw, rcas, out_dir)
    per_model = cases["model_name"].groupby(cases["model_name"]).count()
    parsed_per_model = pd.Series(
        [row["model_name"] for (_, row), r in zip(cases.iterrows(), rcas) if r is not None]
    ).value_counts()
    for m in per_model.index:
        got, want = int(parsed_per_model.get(m, 0)), int(per_model[m])
        flag = "" if got / max(want, 1) > 0.8 else "  <-- LOW, check server logs"
        print(f"  [{m}] {got}/{want} parsed{flag}")
    if len(ok) < 0.5 * len(rcas):
        raise RuntimeError(
            f"Only {len(ok)}/{len(rcas)} RCAs parsed -- refusing to build a taxonomy "
            "from a decimated sample. Check judge server stability and rerun."
        )

    rca_records = []
    for (_, row), r in zip(cases.iterrows(), rcas):
        if r is None:
            continue
        rca_records.append({
            "model_name": row["model_name"],
            "subject_id": int(row["subject_id"]),
            "hadm_id": int(row["hadm_id"]),
            "case_summary": r.get("case_summary", ""),
            "errors": r.get("errors", []),
        })
    with open(os.path.join(out_dir, "stage1_rca.json"), "w") as f:
        json.dump(rca_records, f, indent=2)

    # 3. Cluster the error keywords into a candidate taxonomy.
    candidate = await _cluster_keywords(rca_records, base, model_id, qa_cfg)
    path = os.path.join(out_dir, "taxonomy_candidate.json")
    with open(path, "w") as f:
        json.dump({"approved": False, "classes": candidate}, f, indent=2)

    # 4. Deterministic counting: keyword -> cluster, per model.
    counts_df = _count_clusters(rca_records, candidate)
    counts_path = os.path.join(out_dir, "cluster_counts.csv")
    counts_df.to_csv(counts_path, index=False)
    print(counts_df.to_string(index=False))

    rca_df = wandb_log.rca_records_to_df(rca_records)
    wandb_log.log_table(run, "stage1_rca_errors", rca_df)
    wandb_log.log_artifact_df(run, rca_df, "qa_stage1_rca")
    wandb_log.log_table(run, "taxonomy_candidate", wandb_log.taxonomy_to_df(candidate))
    wandb_log.log_table(run, "cluster_counts", counts_df)
    wandb_log.log_artifact_df(run, counts_df, "qa_cluster_counts")
    wandb_log.log_summary(run, {
        "cases": len(cases), "rcas_parsed": len(ok),
        "error_rows": len(rca_df), "candidate_classes": len(candidate),
    })
    wandb_log.finish(run)

    print(f"Stage 1 done. Candidate taxonomy ({len(candidate)} classes) -> {path}")
    print(f"Cluster sample counts -> {counts_path}")
    print("REVIEW REQUIRED: edit into taxonomy.json and set \"approved\": true "
          "before running stage 2.")


def _dump_parse_failures(raw: List, rcas: List, out_dir: str, limit: int = 20) -> None:
    """Categorize failures and save raw samples so 'why' is inspectable, not guessed."""
    reasons = Counter()
    samples = []
    for t, r in zip(raw, rcas):
        if r is not None:
            continue
        if t is None:
            reasons["request_failed"] += 1
        elif "<think>" in t and "</think>" not in t:
            reasons["thinking_ate_budget"] += 1
        elif "{" not in t:
            reasons["no_json_at_all"] += 1
        else:
            reasons["json_invalid_or_truncated"] += 1
        if t is not None and len(samples) < limit:
            samples.append(t)
    if not reasons:
        return
    print(f"Stage 1: parse-failure breakdown: {dict(reasons)}")
    path = os.path.join(out_dir, "stage1_parse_failures.txt")
    with open(path, "w") as f:
        f.write(f"breakdown: {dict(reasons)}\n\n")
        for i, s in enumerate(samples):
            f.write(f"--- sample {i} ---\n{s}\n\n")
    print(f"Stage 1: raw failure samples -> {path}")


def normalize_keyword(kw: str) -> str:
    """missed_Z-Code / 'Missed Z code' / missed z_code -> missed_z_code."""
    kw = re.sub(r"[^a-z0-9]+", "_", str(kw).lower()).strip("_")
    return kw or "unspecified"


def _keyword_stats(rca_records: List[dict]):
    """counts + example root causes per normalized keyword."""
    counts: Counter = Counter()
    examples: defaultdict = defaultdict(list)
    for rec in rca_records:
        for e in rec["errors"]:
            kw = normalize_keyword(e.get("keyword", ""))
            counts[kw] += 1
            if len(examples[kw]) < 2 and e.get("root_cause"):
                examples[kw].append(str(e["root_cause"])[:160])
    return counts, examples


async def _cluster_keywords(rca_records: List[dict], base: str, model_id: str,
                            qa_cfg: dict) -> List[dict]:
    counts, _examples = _keyword_stats(rca_records)
    total_errors = sum(counts.values())

    # Singleton tail -> 'other' by default: with thousands of unique tags, the
    # long tail carries little mass but blows the clustering context.
    min_count = qa_cfg["stage1"].get("min_keyword_count", 2)
    kept = dict((k, n) for k, n in counts.most_common() if n >= min_count)
    kept_mass = sum(kept.values())
    print(f"Stage 1: {total_errors} errors, {len(counts)} unique keywords; "
          f"clustering {len(kept)} with count >= {min_count} "
          f"({kept_mass}/{total_errors} errors = {kept_mass / max(total_errors, 1):.1%}; "
          f"tail counts as 'other')")

    # Token-budget chunking on 'keyword | count' lines (~chars/3 per token est.).
    budget = qa_cfg["stage1"].get("cluster_input_tokens", 8000)
    chunks: List[List[str]] = []
    cur: List[str] = []
    cur_tok = 0
    for kw, n in kept.items():
        line = f"{kw} | {n}"
        tok = len(line) // 3 + 2
        if cur and cur_tok + tok > budget:
            chunks.append(cur)
            cur, cur_tok = [], 0
        cur.append(line)
        cur_tok += tok
    if cur:
        chunks.append(cur)
    print(f"Stage 1: clustering in {len(chunks)} chunk(s)")

    def chunk_mass(chunk_lines: List[str]) -> int:
        return sum(kept.get(l.split(" | ", 1)[0], 0) for l in chunk_lines)

    async def cluster(chunk_lines: List[str], desc: str) -> List[dict]:
        base_tokens = qa_cfg["stage1"].get("cluster_max_tokens", 6000)

        async def attempt(tokens: int) -> List[dict]:
            prompt = STAGE1_CLUSTER_PROMPT.format(keyword_data="\n".join(chunk_lines))
            raw = await batch_complete(
                [prompt], base, model_id, temperature=0.3,
                max_tokens=tokens, seed=qa_cfg.get("seed", 42),
                concurrency=1, desc=desc,
            )
            obj = extract_json_object(raw[0] or "")
            return obj.get("classes", []) if obj else []

        classes = await attempt(base_tokens)
        # Enumerating every keyword under every class can overrun the output
        # budget on a large chunk (this is what silently drops keywords -- the
        # JSON gets cut off mid-array); a bigger-budget retry catches that.
        if not classes or (chunk_lines and not any(c.get("keywords") for c in classes)):
            print(f"[{desc}] empty/truncated response at {base_tokens} tokens, "
                  f"retrying with {base_tokens * 2}")
            classes = await attempt(base_tokens * 2)

        # Per-chunk visibility: which chunk under-delivers, not just the final
        # aggregate -- this is what a global-only check can't tell you.
        if chunk_lines:
            mapped = {normalize_keyword(k) for c in classes for k in c.get("keywords", [])}
            mass = chunk_mass(chunk_lines)
            cov = sum(kept.get(k, 0) for k in mapped if k in kept) / max(mass, 1)
            flag = "" if cov > 0.7 else "  <-- LOW"
            print(f"[{desc}] {len(classes)} classes, {cov:.1%} of this chunk's "
                  f"mass assigned{flag}")
        return classes

    def check(classes: List[dict], what: str) -> List[dict]:
        if not classes:
            raise RuntimeError(
                f"Keyword clustering ({what}) returned no classes -- the judge call "
                "failed or produced unparseable output. NOT writing an all-'other' "
                "taxonomy; see the error above and rerun stage1."
            )
        mapped = {normalize_keyword(k) for c in classes for k in c.get("keywords", [])}
        coverage = sum(n for k, n in kept.items() if k in mapped) / max(kept_mass, 1)
        print(f"Stage 1: clustering ({what}) -> {len(classes)} classes, "
              f"{coverage:.1%} of clustered-keyword mass assigned")
        if coverage < 0.5:
            raise RuntimeError(
                f"Clustering only covered {coverage:.1%} of keyword mass -- "
                "assignments are missing; rerun stage1 (judge output was degenerate)."
            )
        return classes

    if len(chunks) == 1:
        return check(await cluster(chunks[0], "stage1-cluster"), "single pass")

    # Two-level: cluster each chunk into partial classes, then cluster the
    # PARTIAL CLASSES (name | mass | definition -- a few dozen lines, not
    # thousands of keywords), and expand memberships back to real keywords.
    partial: dict = {}
    for ci, c in enumerate(chunks):
        for cls in await cluster(c, f"stage1-cluster {ci + 1}/{len(chunks)}"):
            partial[f"c{ci}_{cls.get('name', 'unnamed')}"] = cls
    if not partial:
        check([], "chunk passes")

    def partial_mass(cls: dict) -> int:
        return sum(kept.get(normalize_keyword(k), 0) for k in cls.get("keywords", []))

    merge_lines = [
        f"{uname} | {partial_mass(cls)} | {cls.get('definition', '')[:100]}"
        for uname, cls in partial.items()
    ]
    merged = await cluster(merge_lines, "stage1-cluster-merge")

    # Expand partial-class memberships back to real keywords. The judge often
    # rewrites names (drops the cN_ prefix, tweaks casing), so match exact ->
    # normalized -> bare name (prefix stripped, may hit several chunks).
    def bare(name: str) -> str:
        return re.sub(r"^c\d+_", "", str(name))

    bare_map: defaultdict = defaultdict(list)
    for uname in partial:
        bare_map[normalize_keyword(bare(uname))].append(uname)

    consumed = set()
    for cls in merged:
        expanded = set()
        for member in cls.get("keywords", []):
            m_norm = normalize_keyword(member)
            unames = ([member] if member in partial
                      else [m_norm] if m_norm in partial
                      else bare_map.get(normalize_keyword(bare(member)), []))
            if unames:
                for un in unames:
                    consumed.add(un)
                    expanded.update(normalize_keyword(k) for k in partial[un].get("keywords", []))
            else:  # judge echoed a real keyword instead of a partial name
                expanded.add(m_norm)
        cls["keywords"] = expanded

    # Deterministic fallback: partial classes the judge never assigned are
    # attached to the final class with the highest name/definition token overlap
    # (to 'other' when nothing overlaps) -- never silently dropped.
    leftovers = [un for un in partial if un not in consumed]
    if leftovers:
        print(f"Stage 1: merge left {len(leftovers)}/{len(partial)} partial classes "
              "unassigned -- attaching by token overlap")

        def toks(s: str) -> set:
            return set(re.split(r"[^a-z0-9]+", str(s).lower())) - {"", "of", "the", "and"}

        other = next((c for c in merged if c.get("name") == "other"), merged[-1])
        for un in leftovers:
            pt = toks(bare(un)) | toks(partial[un].get("definition", "")[:80])
            best, best_overlap = other, 0
            for cls in merged:
                overlap = len(pt & (toks(cls.get("name", ""))
                                    | toks(cls.get("definition", "")[:80])))
                if overlap > best_overlap:
                    best, best_overlap = cls, overlap
            best["keywords"].update(
                normalize_keyword(k) for k in partial[un].get("keywords", []))

    for cls in merged:
        cls["keywords"] = sorted(cls["keywords"])
    return check(merged, "merge pass")


def _count_clusters(rca_records: List[dict], classes: List[dict]):
    """Deterministic mapping keyword -> class; count errors and distinct cases
    per class, overall and per model. Unassigned keywords count as 'other'."""
    kw2cls = {}
    for cls in classes:
        for kw in cls.get("keywords", []):
            kw2cls.setdefault(normalize_keyword(kw), cls["name"])

    err_counts: Counter = Counter()
    case_sets: defaultdict = defaultdict(set)
    model_case_sets: defaultdict = defaultdict(set)
    models = sorted({rec["model_name"] for rec in rca_records})
    for rec in rca_records:
        case_id = (rec["model_name"], rec["hadm_id"])
        for e in rec["errors"]:
            cls = kw2cls.get(normalize_keyword(e.get("keyword", "")), "other")
            err_counts[cls] += 1
            case_sets[cls].add(case_id)
            model_case_sets[(cls, rec["model_name"])].add(rec["hadm_id"])

    order = [c["name"] for c in classes]
    if "other" not in order:
        order.append("other")
    rows = []
    for cls in order:
        row = {
            "error_class": cls,
            "errors": err_counts.get(cls, 0),
            "cases": len(case_sets.get(cls, set())),
        }
        for m in models:
            row[f"cases_{m}"] = len(model_case_sets.get((cls, m), set()))
        rows.append(row)
    return pd.DataFrame(rows)
