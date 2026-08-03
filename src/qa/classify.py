"""Stage 2: quantify errors against the frozen taxonomy.

For every mispredicted case in each target model's result set, the judge makes one
independent yes/no decision per taxonomy class (guided decoding -> flat boolean
object, temperature 0 for reproducibility). We write:

  * qa_classifications_{model}.parquet -- per-case boolean matrix (audit trail).
  * qa_error_shift.csv -- per-class positive counts and rates per model, plus the
    base-vs-fine-tuned delta that replaces the old keyword-mapped Table 3.

Requires an APPROVED taxonomy.json (`approved: true`); refuses to run otherwise so
counts are never produced from an un-reviewed candidate.
"""
from __future__ import annotations

import asyncio
import json
import os
from typing import List

import pandas as pd

from src.qa import data as qa_data, wandb_log
from src.qa.cache import ResponseCache, chunked, taxonomy_fingerprint
from src.qa.client import batch_complete, resolve_model_id
from src.qa.prompts import STAGE2_CLASSIFY_PROMPT
from src.qa.schemas import (
    Taxonomy, build_classification_model, taxonomy_block, example_block,
    validate_stage2,
)

PREF = qa_data.PRED_PREFIX


def load_approved_taxonomy(repo_root: str, qa_cfg: dict) -> Taxonomy:
    path = os.path.join(repo_root, qa_cfg["output_dir"], qa_cfg["stage2"]["taxonomy_file"])
    with open(path) as f:
        raw = json.load(f)
    if not raw.get("approved"):
        raise RuntimeError(
            f"{path} is not approved. Review the taxonomy and set \"approved\": true "
            "before running stage 2."
        )
    return Taxonomy.model_validate({"classes": raw["classes"]})


def _classify_prompt(row: pd.Series, tax: Taxonomy) -> str:
    names = tax.names()
    return STAGE2_CLASSIFY_PROMPT.format(
        prompt=row[f"{PREF}_prompt"],
        predicted_output=row[f"{PREF}_text"],
        labels=list(row["ICD_CODES"]),
        discharge_note=row["discharge_note"],
        taxonomy_block=taxonomy_block(tax.classes),
        example_block=example_block(names),
    )


async def run_stage2(cfg: dict, repo_root: str, base: str) -> None:
    qa_cfg = cfg["QA"]
    out_dir = os.path.join(repo_root, qa_cfg["output_dir"])
    os.makedirs(out_dir, exist_ok=True)

    tax = load_approved_taxonomy(repo_root, qa_cfg)
    run = wandb_log.start(qa_cfg, "qa-stage2-quantify")
    names = tax.names()
    schema = build_classification_model(names) if qa_cfg["stage2"].get("guided_decoding", True) else None
    print(f"Stage 2: {len(names)} classes -> {names}")

    model_id = await resolve_model_id(
        base, qa_cfg["server"].get("ready_timeout_min", 45))
    round_name = qa_cfg["round"]
    data_root = os.path.join(repo_root, qa_cfg.get("data_root", "data"))

    # Checkpoint cache is fingerprinted with the taxonomy: editing class
    # definitions invalidates old verdicts instead of silently reusing them.
    fp = taxonomy_fingerprint(tax.classes)
    cache = ResponseCache(os.path.join(out_dir, "cache", f"stage2_cls_{fp}.parquet"))
    chunk_size = qa_cfg.get("checkpoint_every", 500)
    val_model = build_classification_model(names)

    per_model_counts = {}
    per_model_n = {}
    for name in qa_cfg["stage2"]["models"]:
        df = qa_data.load_eval_results(repo_root, data_root, round_name, name)
        cases = qa_data.select_cases(df, only_mispredicted=True)
        if qa_cfg["stage2"].get("limit"):
            cases = cases.head(qa_cfg["stage2"]["limit"]).reset_index(drop=True)

        prompts = [_classify_prompt(r, tax) for _, r in cases.iterrows()]
        keys = [ResponseCache.key(name, r["hadm_id"]) for _, r in cases.iterrows()]
        raw = [cache.get(k) for k in keys]
        todo = [i for i, t in enumerate(raw)
                if t is None or validate_stage2(t, val_model) is None]
        print(f"Stage 2 [{name}]: {len(cases)} mispredicted cases, "
              f"{len(cases) - len(todo)} from checkpoint, {len(todo)} to request")

        for n_blk, block in enumerate(chunked(todo, chunk_size), 1):
            res = await batch_complete(
                [prompts[i] for i in block], base, model_id,
                temperature=qa_cfg["stage2"].get("temperature", 0.0),
                max_tokens=qa_cfg["stage2"].get("max_tokens", 512),
                seed=qa_cfg.get("seed", 42),
                concurrency=qa_cfg.get("concurrency", 32),
                schema=schema,
                desc=f"stage2-{name} {n_blk}/{(len(todo) - 1) // chunk_size + 1}",
            )
            for i, t in zip(block, res):
                if t is not None:
                    raw[i] = t
                    if validate_stage2(t, val_model) is not None:
                        cache.put(keys[i], t)
            cache.flush()

        rows = []
        for (_, case), t in zip(cases.iterrows(), raw):
            verdict = validate_stage2(t, val_model) if t else None
            rec = {"subject_id": int(case["subject_id"]), "hadm_id": int(case["hadm_id"]),
                   "model_name": name, "parsed": verdict is not None}
            for n in names:
                rec[n] = bool(verdict[n]) if verdict else None
            rows.append(rec)
        result = pd.DataFrame(rows)
        result.to_parquet(os.path.join(out_dir, f"qa_classifications_{name}.parquet"))
        wandb_log.log_artifact_df(run, result, f"qa_classifications_{name}")

        parsed = result[result["parsed"]]
        per_model_counts[name] = {n: int(parsed[n].sum()) for n in names}
        per_model_n[name] = len(parsed)
        print(f"Stage 2 [{name}]: {len(parsed)}/{len(result)} parsed")

    shift = _write_error_shift(out_dir, names, per_model_counts, per_model_n, qa_cfg)
    wandb_log.log_table(run, "qa_error_shift", shift)
    wandb_log.log_artifact_df(run, shift, "qa_error_shift")
    wandb_log.log_summary(run, {
        "classes": len(names),
        **{f"n_parsed_{m}": n for m, n in per_model_n.items()},
    })
    wandb_log.finish(run)
    print(f"Stage 2 done -> {os.path.join(out_dir, 'qa_error_shift.csv')}")


def _write_error_shift(out_dir: str, names: List[str], counts: dict, n: dict,
                       qa_cfg: dict) -> pd.DataFrame:
    rows = []
    for cls in names:
        row = {"error_class": cls}
        for model, c in counts.items():
            row[f"{model}__count"] = c[cls]
            row[f"{model}__rate"] = round(c[cls] / n[model], 4) if n[model] else None
        rows.append(row)
    table = pd.DataFrame(rows)

    # shift_pairs: [[from, to], ...] -- one delta_count/delta_rate column pair
    # per comparison (e.g. base->full at each size, full->LoRA at a fixed size).
    # shift_pair (singular, legacy): a single [from, to] pair, still supported.
    pairs = qa_cfg["stage2"].get("shift_pairs")
    if not pairs:
        single = qa_cfg["stage2"].get("shift_pair")
        pairs = [single] if single else []

    for pair in pairs:
        if not pair or len(pair) != 2 or not all(p in counts for p in pair):
            continue
        from_m, to_m = pair
        suffix = f"{to_m}_vs_{from_m}"
        table[f"delta_count__{suffix}"] = table[f"{to_m}__count"] - table[f"{from_m}__count"]
        table[f"delta_rate__{suffix}"] = (
            table[f"{to_m}__rate"] - table[f"{from_m}__rate"]
        ).round(4)

    table.to_csv(os.path.join(out_dir, "qa_error_shift.csv"), index=False)
    return table
