#!/usr/bin/env python3
"""Head-to-head comparison of MERLIN against CliniBench's encoder baselines,
recomputed end-to-end from CliniBench's *released per-admission predictions*
rather than from the numbers printed in their paper (arXiv 2509.26136v2).

Motivation: a reader will ask why MERLIN's ICD macro-F1
(20.9 at 32B) sits below the 26.0 CliniBench reports for a fine-tuned encoder.
Answering that from published numbers is not possible -- the two are measured
on different test sets, over different label vocabularies, at different output
budgets, and (see below) their F1 column does not reproduce from their own
released predictions. So we recompute every encoder number ourselves, on OUR
test admissions, under one stated protocol.


WHAT MAKES THE COMPARISON POSSIBLE
----------------------------------
CliniBench's prediction dumps (Encoder_<model>_<cfg>_<dataset>-icd-<v>.json)
carry no hadm_id -- just three positionally-indexed lists. But they are row-
aligned with the split parquet they were produced from:

    Encoder_GatorTronS_tuned_hosp-icd-10.json[i]
      <-> data/mimic-iv/admission_notes/icd-10/hosp/test_10_hosp.pq[i]

verified by an exact ordered match of the gold code lists after 3-char
truncation (19,779/19,779 on hosp-icd-10). `assert_row_alignment()` re-checks
this every run; if CliniBench ever re-exports in a different order this module
fails loudly instead of silently comparing mismatched admissions.

That gives us a hadm_id per CliniBench row, hence an exact join to our own
test split. All 2,184 of our test admissions fall in CliniBench's *test* split
(0 in their train, 0 in their dev) -- `check_split_membership()` asserts this,
since a train-side overlap would invalidate the encoder numbers.


THEIR PROTOCOL, RECOVERED FROM THE PAPER
----------------------------------------
Table 2 cells are (i) macro-averaged over labels, (ii) computed on the top-20
prediction list, and (iii) the *unweighted mean of the ICD-9 and ICD-10 runs*.
`reproduce_table2()` implements exactly that and checks it against the printed
values. Result of the audit (ours - published):

    row Rec Prec MD Acc F1
    BiomedBERT untuned 0.00 +0.00 +0.00 +7.37
    BiomedBERT tuned 0.00 +0.00 +0.00 -1.03
    GatorTronS untuned +4.91 +0.42 +0.70 +7.31
    GatorTronS tuned 0.00 +0.00 -0.00 -0.62

Two things follow, both encoded as expectations in REPRODUCTION_EXPECTATIONS:

1. Three of the four rows reproduce to 0.00 on recall/precision/MD-accuracy.
   Those are asserted (TOL_EXACT) and act as a regression test on our metric
   implementations: if we ever break macro-recall or the top-20 truncation,
   these fire.

2. `Encoder_GatorTronS_untuned_*` does NOT match its published row (recall off
   by ~5 points) while its three siblings do -- most likely a stale or
   different-checkpoint export on their side. It is therefore reported but
   NOT asserted, and `PROVISIONAL_CONFIGS` marks it so downstream tables can
   flag it. Use BiomedBERT-untuned when an un-threshold-tuned encoder is
   needed for a claim; that one reproduces exactly and shows the same
   qualitative behaviour.

3. The F1 column reproduces for NO row, and not randomly: the two untuned rows
   are off by ~+7.3 and the two tuned rows by ~-0.6..-1.0, in the same files
   where every other column lands on 0.00. Their macro-F1 recipe is therefore
   not recoverable from the released artifacts. This is why the paper must not
   quote their 25.96 -- we recompute instead.

Their MAP is macro-averaged over labels too, which is why it reads 22.06 where
a per-instance AP gives 41.16 on the same predictions. We cannot compute their
macro-MAP for MERLIN (it needs per-label scores; a generative model emits a
ranked list, not a scored vocabulary), so `map_at_n` here is per-instance AP
and is only ever compared against our own recomputation of theirs -- never
against their printed MAP column.


MACRO AVERAGING NEEDS A FIXED VOCABULARY
----------------------------------------
Deliberately NOT reusing classification_metrics.calculate_icd_metrics_cpu for
the macro columns: it macro-averages over the labels *observed for that model*
(union of its own tp/fp/fn keys). Across systems that means different
denominators -- 923 labels for MERLIN-32B vs 1,032 for GatorTronS on the same
admissions -- and the resulting "macro-F1" values are not comparable. An
earlier pass at this comparison reported a 0.65-point gap that was entirely an
artifact of that mismatch; the real gap over a common vocabulary is ~4.5.
`macro_metrics()` therefore takes an explicit `vocab`, defaulting to the gold
codes observed on the evaluation subset. Micro-F1 is vocabulary-independent
and does reuse the shared implementation, so it stays consistent with
results_test.csv.


ORDERING: USE v4_json, NOT v4_preds
-----------------------------------
`v4_preds` in our eval parquets is ~`set()` of the 3-char codes -- Python hash
order, emission order destroyed (set-equality with v4_json is only 0.92-0.95,
so it is not even a clean set(); there is retry/best-of-N filtering in
between). Every rank-sensitive metric computed on it is measuring a random
shuffle: MAP@20 reads 32.3 from v4_preds vs 40.3 from v4_json, and gold-primary
-in-top-3 reads 24.4 vs 59.8. This module reads the ranked list out of
`v4_json`, which preserves the order the V4 prompt asks for ("in order of
likelihood"). Set-based metrics are unaffected either way.

Run: python scripts/eval_analysis.py clinibench
      python -m src.eval.clinibench_headtohead (uses the same config)

Config: the `CliniBench:` block of scripts/eval_config.yaml.
Outputs (under CliniBench.output_dir):
    clinibench_reproduction.csv per-row published vs. recomputed + delta
    clinibench_headtohead.csv MERLIN vs. encoder configs on our test split
"""
from __future__ import annotations

import ast
import json
import logging
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from src.eval.classification_metrics import calculate_icd_metrics_cpu

log = logging.getLogger(__name__)

REPO = Path(__file__).resolve().parents[2]

# Table 2 (HOSP, zero-shot column) of arXiv 2509.26136v2, as printed:
# (macro recall, macro precision, MAP, MD accuracy, macro F1)
PUBLISHED_TABLE2_HOSP = {
    ("BiomedBERT", "untuned"): (30.74, 19.91, 33.61, 78.21, 14.63),
    ("BiomedBERT", "tuned"): (22.91, 29.55, 19.34, 59.16, 21.74),
    ("GatorTronS", "untuned"): (30.45, 22.70, 35.26, 80.94, 18.46),
    ("GatorTronS", "tuned"): (28.61, 33.44, 22.06, 65.22, 25.96),
}

# Columns we expect to reproduce exactly, per row. See the audit in
# the module docstring for why GatorTronS/untuned is excluded and why "f1" is
# excluded everywhere.
REPRODUCTION_EXPECTATIONS = {
    ("BiomedBERT", "untuned"): ("recall", "precision", "md_acc"),
    ("BiomedBERT", "tuned"): ("recall", "precision", "md_acc"),
    ("GatorTronS", "untuned"): (),
    ("GatorTronS", "tuned"): ("recall", "precision", "md_acc"),
}

# Configs whose released predictions disagree with their own published row.
# Reported, never asserted, and flagged in the output CSV.
PROVISIONAL_CONFIGS = {("GatorTronS", "untuned")}

TOL_EXACT = 0.01  # published values are printed to 2dp
ROUND_DIGITS = 2


# ── parsing helpers ──────────────────────────────────────────────────────────
def safe_parse(value) -> list:
    """Our parquets store list columns as either real lists or their repr."""
    if isinstance(value, str):
        try:
            return list(ast.literal_eval(value))
        except (ValueError, SyntaxError):
            return []
    if isinstance(value, (list, tuple, np.ndarray)):
        return list(value)
    return []


def dedup_codes(codes) -> list[str]:
    """Order-preserving dedup, no truncation.

    Used for CliniBench's own dumps, whose codes are already at the benchmark's
    target granularity. Do NOT push them through short_codes(): ICD-9 keeps
    108 four-character E-codes (E800-E999) that a blanket [:3] would collapse
    into each other, which silently shifts macro recall/precision by ~1.3
    points and breaks the Table 2 reproduction.
    """
    out, seen = [], set()
    for code in codes:
        code = str(code)
        if code and code not in seen:
            seen.add(code)
            out.append(code)
    return out


def short_codes(codes) -> list[str]:
    """3-char truncation + order-preserving dedup, for OUR ICD-10 data.

    Our gold (`ICD_CODES`) and our model's output are full-length ICD-10, so
    they do need truncating to match the benchmark's 3-character label space.
    """
    out, seen = [], set()
    for code in codes:
        cat = str(code).replace(".", "")[:3]
        if cat and cat not in seen:
            seen.add(cat)
            out.append(cat)
    return out


def ranked_codes_from_v4_json(v4_json) -> list[str]:
    """Ranked 3-char list out of v4_json. NOT v4_preds -- see module docstring."""
    parsed = safe_parse(v4_json)
    if not (isinstance(parsed, list) and parsed and isinstance(parsed[0], dict)):
        return []
    return short_codes(entry.get("icd_code", "") for entry in parsed)


# ── metrics ──────────────────────────────────────────────────────────────────
def macro_metrics(preds: list[list[str]], golds: list[list[str]],
                  vocab: list[str]) -> dict[str, float]:
    """Macro recall/precision/F1 over an EXPLICIT label vocabulary.

    The explicit vocab is the whole point: averaging over each system's own
    observed labels gives incomparable denominators across systems.
    """
    tp, fp, fn = Counter(), Counter(), Counter()
    for pred, gold in zip(preds, golds):
        pred, gold = set(pred), set(gold)
        tp.update(pred & gold)
        fp.update(pred - gold)
        fn.update(gold - pred)

    recalls, precisions, f1s = [], [], []
    for label in vocab:
        p = tp[label] / (tp[label] + fp[label]) if tp[label] + fp[label] else 0.0
        r = tp[label] / (tp[label] + fn[label]) if tp[label] + fn[label] else 0.0
        recalls.append(r)
        precisions.append(p)
        f1s.append(2 * p * r / (p + r) if p + r else 0.0)
    return {
        "recall": 100 * float(np.mean(recalls)),
        "precision": 100 * float(np.mean(precisions)),
        "f1": 100 * float(np.mean(f1s)),
    }


def md_accuracy(preds: list[list[str]], golds: list[list[str]]) -> float:
    """CliniBench's MD Acc: is the FIRST annotated code among the predictions?

    Order-independent on the prediction side (membership, not rank), so this
    one is safe to compute from either v4_json or v4_preds.
    """
    hits = [1.0 if (g and g[0] in set(p)) else 0.0 for p, g in zip(preds, golds)]
    return 100 * float(np.mean(hits)) if hits else 0.0


def map_at_n(preds: list[list[str]], golds: list[list[str]],
             top_n: int | None = 20) -> float:
    """Per-instance mean average precision, AP normalised by |gold|.

    Unretrieved relevant codes contribute 0, so short prediction lists are
    penalised on recall rather than flattered -- AP is monotone non-decreasing
    in list length (appending never changes precision at earlier ranks), which
    is why a ~12-code generative output cannot game this against a 20- or
    34-code encoder list.

    NOT comparable to CliniBench's printed MAP column, which is macro-averaged
    over labels. Only ever compare this against our own recomputation.
    """
    aps = []
    for pred, gold in zip(preds, golds):
        gold_set = set(gold)
        if not gold_set:
            continue
        hits, acc = 0, 0.0
        for rank, code in enumerate(pred[:top_n] if top_n else pred, start=1):
            if code in gold_set:
                hits += 1
                acc += hits / rank
        aps.append(acc / len(gold_set))
    return 100 * float(np.mean(aps)) if aps else 0.0


def score_system(preds, golds, vocab, top_n=20) -> dict[str, float]:
    """Every column of the head-to-head table for one system."""
    micro = calculate_icd_metrics_cpu(preds, golds)  # shared impl -> micro only
    row = macro_metrics(preds, golds, vocab)
    row.update({
        "micro_f1": micro["ICD F1 Micro"] * 100 if micro["ICD F1 Micro"] <= 1
        else micro["ICD F1 Micro"],
        "map_at_n": map_at_n(preds, golds, top_n),
        "md_acc": md_accuracy(preds, golds),
        "codes_per_adm": float(np.mean([len(p) for p in preds])),
    })
    return {k: round(v, ROUND_DIGITS) for k, v in row.items()}


# ── CliniBench artifact loading ──────────────────────────────────────────────
def clinibench_json_path(encoder_dir: Path, model: str, cfg: str,
                         dataset: str, icd_version: int) -> Path:
    return encoder_dir / f"Encoder_{model}_{cfg}_{dataset}-icd-{icd_version}.json"


def split_parquet_path(splits_dir: Path, dataset: str, icd_version: int,
                       split: str = "test") -> Path:
    return (splits_dir / f"icd-{icd_version}" / dataset
            / f"{split}_{icd_version}_{dataset}.pq")


def assert_row_alignment(payload: dict, split_df: pd.DataFrame) -> None:
    """CliniBench rows are positional -- verify against the split parquet.

    Guards the whole module: without hadm_ids in their JSON, positional
    alignment is the only thing letting us join to our admissions.
    """
    gold_json = payload["annotated_codes"]
    if len(gold_json) != len(split_df):
        raise AssertionError(
            f"row-count mismatch: CliniBench has {len(gold_json)}, "
            f"split parquet has {len(split_df)}")
    gold_pq = [[str(c)[:3] for c in codes] for codes in split_df["SHORT_CODES"]]
    mismatches = sum(1 for a, b in zip(gold_json, gold_pq) if list(a) != b)
    if mismatches:
        raise AssertionError(
            f"positional alignment broken: {mismatches}/{len(gold_pq)} rows "
            "have different gold code lists. CliniBench may have re-exported "
            "in a different order -- do NOT trust any join built on this.")
    log.info("row alignment verified: %d/%d exact gold matches",
             len(gold_pq), len(gold_pq))


def load_clinibench(encoder_dir: Path, splits_dir: Path, model: str, cfg: str,
                    dataset: str, icd_version: int,
                    verify_alignment: bool = True) -> dict:
    """Load one prediction dump; with `verify_alignment`, attach hadm_ids.

    Alignment holds for hosp-icd-10 (19,779/19,779 exact) but NOT for
    hosp-icd-9, where their dump has 36,003 rows against the split parquet's
    36,007 -- four admissions dropped somewhere in their export. So ICD-9 dumps
    are usable for reproducing their published (self-contained) numbers, where
    gold comes out of the same JSON, but must never be positionally joined to
    anything. Callers that only need their own gold pass verify_alignment=False.
    """
    path = clinibench_json_path(encoder_dir, model, cfg, dataset, icd_version)
    payload = json.loads(path.read_text())
    hadm_ids = None
    if verify_alignment:
        split_df = pd.read_parquet(
            split_parquet_path(splits_dir, dataset, icd_version),
            columns=["hadm_id", "SHORT_CODES"])
        assert_row_alignment(payload, split_df)
        hadm_ids = list(split_df["hadm_id"])
    return {
        "hadm_id": hadm_ids,
        "gold": [dedup_codes(g) for g in payload["annotated_codes"]],
        "full": [dedup_codes(p) for p in payload["predicted_codes"]],
        "top_n": [dedup_codes(p) for p in payload["predicted_codes@20"]],
        "vocab": list(payload["unique_labels"]),
    }


# ── stage 1: reproduce their Table 2 ─────────────────────────────────────────
def reproduce_table2(encoder_dir: Path, splits_dir: Path, dataset: str,
                     icd_versions: list[int], strict: bool = True) -> pd.DataFrame:
    """Recompute Table 2 under their protocol and diff against the printed row.

    Protocol: macro over labels, top-20 list, unweighted mean of ICD versions.
    Asserts only the columns listed in REPRODUCTION_EXPECTATIONS.
    """
    rows, failures = [], []
    for (model, cfg), published in PUBLISHED_TABLE2_HOSP.items():
        per_version = []
        for icd_version in icd_versions:
            # gold comes out of their own dump here, so no join is needed and
            # hosp-icd-9's row-count gap (see load_clinibench) is harmless
            data = load_clinibench(encoder_dir, splits_dir, model, cfg,
                                   dataset, icd_version, verify_alignment=False)
            scored = macro_metrics(data["top_n"], data["gold"], data["vocab"])
            scored["md_acc"] = md_accuracy(data["top_n"], data["gold"])
            per_version.append(scored)

        # unweighted mean across ICD-9 / ICD-10, as their caption specifies
        got = {k: float(np.mean([v[k] for v in per_version]))
               for k in per_version[0]}
        pub = dict(zip(("recall", "precision", "map", "md_acc", "f1"), published))
        provisional = (model, cfg) in PROVISIONAL_CONFIGS

        for column in ("recall", "precision", "f1", "md_acc"):
            delta = got[column] - pub[column]
            rows.append({
                "model": model, "config": cfg, "metric": column,
                "published": pub[column],
                "recomputed": round(got[column], ROUND_DIGITS),
                "delta": round(delta, ROUND_DIGITS),
                "asserted": column in REPRODUCTION_EXPECTATIONS[(model, cfg)],
                "provisional_artifact": provisional,
            })
            if (column in REPRODUCTION_EXPECTATIONS[(model, cfg)]
                    and abs(delta) > TOL_EXACT):
                failures.append(
                    f"{model}/{cfg} {column}: published {pub[column]}, "
                    f"recomputed {got[column]:.2f} (delta {delta:+.2f})")

    if failures and strict:
        raise AssertionError(
            "CliniBench reproduction regressed -- these reproduced exactly on "
            "at the time of writing, so a break means our metric code changed (or their "
            "artifacts were re-exported):\n  " + "\n  ".join(failures))
    for f in failures:
        log.warning("reproduction mismatch: %s", f)
    return pd.DataFrame(rows)


# ── stage 2: head-to-head on our test admissions ─────────────────────────────
def check_split_membership(our_hadm_ids: set, splits_dir: Path, dataset: str,
                           icd_version: int) -> dict[str, int]:
    """Our test admissions must sit in THEIR test split, never their train."""
    counts = {}
    for split in ("train", "dev", "test"):
        path = split_parquet_path(splits_dir, dataset, icd_version, split)
        counts[split] = len(our_hadm_ids & set(pd.read_parquet(
            path, columns=["hadm_id"])["hadm_id"]))
    if counts["train"] or counts["dev"]:
        raise AssertionError(
            f"leakage: {counts['train']} of our test admissions are in "
            f"CliniBench's train split and {counts['dev']} in their dev split. "
            "Encoder numbers on those admissions are inflated.")
    log.info("split membership: %d in their test, 0 train, 0 dev", counts["test"])
    return counts


def load_our_run(eval_results_dir: Path, collection: str) -> pd.DataFrame:
    path = eval_results_dir / f"eval_results_{collection}" / f"eval_results_{collection}.pq"
    df = pd.read_parquet(path, columns=["hadm_id", "ICD_CODES", "v4_json"])
    df["pred"] = df["v4_json"].apply(ranked_codes_from_v4_json)
    df["gold"] = df["ICD_CODES"].apply(lambda v: short_codes(safe_parse(v)))
    return df[["hadm_id", "pred", "gold"]]


def head_to_head(cfg: dict) -> pd.DataFrame:
    """MERLIN vs. every CliniBench encoder config, on OUR test admissions."""
    encoder_dir = Path(cfg["encoder_dir"])
    splits_dir = REPO / cfg["splits_dir"]
    eval_dir = REPO / cfg["eval_results_dir"]
    dataset, icd_version, top_n = cfg["dataset"], cfg["icd_version"], cfg["top_n"]

    ours = {label: load_our_run(eval_dir, coll)
            for coll, label in cfg["models"].items()}
    reference = next(iter(ours.values()))
    hadm_ids = list(reference["hadm_id"])
    golds = list(reference["gold"])
    check_split_membership(set(hadm_ids), splits_dir, dataset, icd_version)

    # macro vocabulary: gold codes observed on THIS subset, shared by all rows
    vocab = sorted({code for gold in golds for code in gold})
    log.info("macro vocabulary: %d gold codes over %d admissions",
             len(vocab), len(golds))

    rows = []
    for label, df in ours.items():
        if list(df["hadm_id"]) != hadm_ids:
            df = df.set_index("hadm_id").loc[hadm_ids].reset_index()
        rows.append({"system": label, "family": "MERLIN", "budget": "emitted",
                     "provisional_artifact": False,
                     **score_system(list(df["pred"]), golds, vocab, top_n)})

    for model, encoder_cfg in cfg["encoders"]:
        data = load_clinibench(encoder_dir, splits_dir, model, encoder_cfg,
                               dataset, icd_version)
        position = {h: i for i, h in enumerate(data["hadm_id"])}
        take = [position[h] for h in hadm_ids]
        for budget, key in (("full", "full"), (f"top-{top_n}", "top_n")):
            preds = [data[key][i] for i in take]
            rows.append({
                "system": f"{model} ({encoder_cfg})", "family": "encoder",
                "budget": budget,
                "provisional_artifact": (model, encoder_cfg) in PROVISIONAL_CONFIGS,
                **score_system(preds, golds, vocab, top_n),
            })

    columns = ["system", "family", "budget", "codes_per_adm", "recall",
               "precision", "f1", "micro_f1", "map_at_n", "md_acc",
               "provisional_artifact"]
    return pd.DataFrame(rows)[columns]


# ── entry point ──────────────────────────────────────────────────────────────
def run(cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    out_dir = REPO / cfg["output_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)

    repro = reproduce_table2(
        Path(cfg["encoder_dir"]), REPO / cfg["splits_dir"], cfg["dataset"],
        cfg["icd_versions"], strict=cfg.get("strict_reproduction", True))
    repro.to_csv(out_dir / "clinibench_reproduction.csv", index=False)

    table = head_to_head(cfg)
    table.to_csv(out_dir / "clinibench_headtohead.csv", index=False)

    log.info("wrote %s and %s", out_dir / "clinibench_reproduction.csv",
             out_dir / "clinibench_headtohead.csv")
    return repro, table


def main() -> None:
    import yaml
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    cfg = yaml.safe_load((REPO / "scripts" / "eval_config.yaml").read_text())["CliniBench"]
    repro, table = run(cfg)
    print("\n=== CliniBench Table 2 reproduction (ours - published) ===")
    print(repro.to_string(index=False))
    print("\n=== Head-to-head on our test admissions ===")
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()
