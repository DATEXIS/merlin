#!/usr/bin/env python3
"""Backfill ICD Precision/Recall (Micro + Macro) into
data/eval_metrics_merlin-eval-1.4.csv from data/results/evaluation/results_test.csv.

Why: calculate_icd_metrics_cpu only started returning precision/recall on
2026-07-22 (see src/eval/classification_metrics.py). The seed-43/44 reruns were
evaluated after that fix and carry the four columns; the original seed-42 rows
in eval_metrics_*.csv do not. src/eval/backfill_icd_metrics.py already
recomputed them into results_test.csv from the cached .pq artifacts, so this
script just carries those same values across into the flat per-collection CSV
that src/eval/paper_tables.py reads, rather than recomputing them a second time.

Consequence for the paper tables: the main table's R_mac / P_mac columns were
being averaged over 2 eval seeds while every other column used 3. After this
backfill the Qwen3 rows use all 3. The two external baselines
(test-llama-3-70b-instruct, test-medgemma-27b-it) have no seed-42 row in
results_test.csv, so those stay 2-seed; the script prints what it could not fix.

Only the four ICD precision/recall columns are written. Existing F1, V1/V2
metrics, JSON validity and n_samples are never touched, and a row is skipped
if its F1 in the two CSVs disagrees by more than 0.02 (that would mean the two
files describe different runs).

Run:
    python -m src.eval.backfill_eval_metrics_pr
    python -m src.eval.backfill_eval_metrics_pr --dry-run
"""
import argparse
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[2]
TARGET_CSV = REPO / "data" / "eval_metrics_merlin-eval-1.4.csv"
SOURCE_CSV = REPO / "data" / "results" / "evaluation" / "results_test.csv"
PR_COLUMNS = ["ICD Precision Micro", "ICD Recall Micro",
              "ICD Precision Macro", "ICD Recall Macro"]
F1_COLUMNS = ["ICD F1 Micro", "ICD F1 Macro"]
DRIFT_TOL = 0.02


def main(target: Path = TARGET_CSV, source: Path = SOURCE_CSV, dry_run: bool = False):
    tgt = pd.read_csv(target, index_col=0)
    src = pd.read_csv(source).set_index("collection")

    filled, conflicted, unfixable = [], [], []
    for coll in tgt.index[tgt["ICD Recall Macro"].isna()]:
        if coll not in src.index or pd.isna(src.loc[coll, "ICD Recall Macro"]):
            unfixable.append(coll)
            continue
        drift = [(c, tgt.loc[coll, c], src.loc[coll, c]) for c in F1_COLUMNS
                 if pd.notna(tgt.loc[coll, c])
                 and abs(float(tgt.loc[coll, c]) - float(src.loc[coll, c])) > DRIFT_TOL]
        if drift:
            conflicted.append((coll, drift))
            continue
        for c in PR_COLUMNS:
            tgt.loc[coll, c] = src.loc[coll, c]
        filled.append(coll)

    if not dry_run:
        tgt.to_csv(target)
    print(f"{'[dry-run] would fill' if dry_run else 'Filled'} {len(filled)} row(s) in {target.name}")
    for c in filled:
        print(f"  + {c}")
    if conflicted:
        print(f"  CONFLICT ({len(conflicted)}): F1 disagrees between the two CSVs, left untouched")
        for coll, drift in conflicted:
            for c, a, b in drift:
                print(f"    {coll} {c}: target={a} source={b}")
    # Only report unfixable rows that a paper table actually reads (test-*);
    # the dev epoch-sweep rows are expected to lack these columns.
    test_unfixable = [c for c in unfixable if str(c).startswith("test-")]
    if test_unfixable:
        print(f"  no source row ({len(test_unfixable)} test collections, stay as-is): {test_unfixable}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()
    main(dry_run=args.dry_run)
