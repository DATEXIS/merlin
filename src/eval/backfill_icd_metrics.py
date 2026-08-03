#!/usr/bin/env python3
"""One-off local backfill: add ICD Precision/Recall (Micro + Macro) to every
row already sitting in an evaluation-results CSV (data/results/evaluation/),
from CACHED local artifacts only -- no WandB API calls, no re-running
eval_analysis.py's new_evals stage.

Why this exists: calculate_icd_metrics_cpu (src/eval/classification_metrics.py)
computed precision/recall internally all along but only returned F1 -- fixed
2026-07-22 (see that module). Every row written before the fix is missing the
4 new columns. Re-running `eval_analysis.py new_evals --force` would
recompute them too, but load_artifact_df's `force` path re-downloads every
artifact from WandB even when already cached locally under
data/results/evaluation/{version}/ -- unnecessary network/auth cost just to
pick up 4 columns that only need locally-cached data. This script reuses
those cached .pq files directly and fails loudly (lists them, doesn't crash)
for any row whose artifact isn't cached, rather than silently leaving it half
right.

Scope: only touches ICD Precision/Recall/F1 -- never writes V1/V2 metrics,
JSON validity, n_samples, etc., so it can't regress anything else in the row.
Recomputed ICD F1 Micro/Macro are used only as a sanity check (warn on >0.02
drift from the existing CSV value); the CSV's existing F1 is never
overwritten by this script.

Encoder rows (mode == "encoder") are skipped -- those bypass the WandB
pipeline entirely and are backfilled by src/eval/encoder_metrics.py instead,
which recomputes directly from data/results/encoder_results/.

Run:
    python -m src.eval.backfill_icd_metrics
    python -m src.eval.backfill_icd_metrics --csv data/results/evaluation/results_dev.csv

If a collection shows up under CONFLICT and you've confirmed the CSV row (not
the cached artifact) is the stale one -- e.g. We confirmed 2026-07-22 that
8b-r128-thrfull-icd2-e2/e3/e4 in results_dev.csv were old numbers from before
a re-eval fixed a request-failure bug, while results_test.csv had already been
refreshed -- re-run with --force-collections to overwrite ICD F1 Micro/Macro
(not just precision/recall) from the cached artifact for just those rows:
    python -m src.eval.backfill_icd_metrics --csv data/results/evaluation/results_dev.csv \\
        --force-collections 8b-r128-thrfull-icd2-e2,8b-r128-thrfull-icd2-e3,8b-r128-thrfull-icd2-e4
Even with --force-collections, this ONLY overwrites the ICD F1/Precision/Recall
columns for the named rows -- V1/V2 metrics, JSON validity, CosSim/DotProduct,
etc. in that same row are left as-is (this script can't recompute them locally,
see Scope below) and may also be stale if the artifact was genuinely re-run;
re-run the full `eval_analysis.py new_evals --force` pipeline to refresh those.
"""
import argparse
from pathlib import Path

import pandas as pd

from src.eval.classification_metrics import calculate_icd_metrics_cpu
from src.eval.eval_experiments import safe_parse_list
from src.eval.eval_results import default_download_dir
from src.utils import convert_codes_to_short_codes

REPO = Path(__file__).resolve().parents[2]
DEFAULT_CSV = REPO / "data" / "results" / "evaluation" / "results_test.csv"
NEW_COLUMNS = ["ICD Precision Micro", "ICD Recall Micro", "ICD Precision Macro", "ICD Recall Macro"]
SKIP_MODES = ("encoder",)


def _artifact_path(download_dir: Path, collection: str) -> Path | None:
    art_dir = download_dir / f"eval_results_{collection}"
    if not art_dir.exists():
        return None
    direct = art_dir / f"eval_results_{collection}.pq"
    if direct.exists():
        return direct
    candidates = list(art_dir.glob("*.pq"))
    return candidates[0] if candidates else None


def recompute_icd_metrics(pq_path: Path) -> dict:
    df = pd.read_parquet(pq_path)
    v4_pred = df["v4_preds"].apply(safe_parse_list).tolist()
    v4_label = df["ICD_CODES"].apply(convert_codes_to_short_codes).tolist()
    metrics = calculate_icd_metrics_cpu(v4_pred, v4_label)
    return {k: round(v * 100, 2) for k, v in metrics.items()}


def main(csv_path: Path = DEFAULT_CSV, project: str = "merlin-eval-1.4", force_collections=()):
    download_dir = default_download_dir(project)
    force_collections = set(force_collections)
    df = pd.read_csv(csv_path)
    for col in NEW_COLUMNS:
        if col not in df.columns:
            df[col] = None

    updated, forced, skipped, missing, conflicted = [], [], [], [], []
    for i, row in df.iterrows():
        if row.get("mode") in SKIP_MODES:
            skipped.append(row["collection"])
            continue
        pq_path = _artifact_path(download_dir, row["collection"])
        if pq_path is None:
            missing.append(row["collection"])
            continue

        recomputed = recompute_icd_metrics(pq_path)

        # Drift check FIRST, before writing anything: if the cached artifact's
        # F1 disagrees with what's already in the CSV, the artifact on disk
        # isn't the same run the CSV's row was built from (e.g. a re-eval
        # landed locally after the CSV was last generated -- has happened,
        # see .pq.stale files under data/results/evaluation/). Writing
        # precision/recall from a DIFFERENT run than the row's own F1 would
        # make the row internally inconsistent, so skip it entirely and flag
        # it instead of guessing which number is right -- UNLESS the caller
        # already confirmed via --force-collections that the CSV (not the
        # artifact) is the stale one, in which case F1 gets overwritten too.
        row_conflicted = False
        for f1_col in ("ICD F1 Micro", "ICD F1 Macro"):
            if f1_col in df.columns and pd.notna(row.get(f1_col)):
                if abs(recomputed[f1_col] - row[f1_col]) > 0.02:
                    conflicted.append((row["collection"], f1_col, row[f1_col], recomputed[f1_col]))
                    row_conflicted = True
        if row_conflicted and row["collection"] not in force_collections:
            continue

        for col in NEW_COLUMNS:
            df.at[i, col] = recomputed[col]
        if row_conflicted:
            df.at[i, "ICD F1 Micro"] = recomputed["ICD F1 Micro"]
            df.at[i, "ICD F1 Macro"] = recomputed["ICD F1 Macro"]
            forced.append(row["collection"])
        else:
            updated.append(row["collection"])

    df.to_csv(csv_path, index=False)
    print(f"Backfilled {len(updated)} row(s) -> {csv_path}")
    if forced:
        print(f"  {len(forced)} row(s) force-overwritten (ICD F1 Micro/Macro + precision/recall, "
              f"per --force-collections): {forced}")
    if skipped:
        print(f"  {len(skipped)} encoder row(s) skipped (handled by src/eval/encoder_metrics.py): {skipped}")
    if missing:
        print(f"  {len(missing)} row(s) had no cached artifact under {download_dir}, left untouched: {missing}")
    still_conflicted = sorted(set(c for c, *_ in conflicted) - force_collections)
    if still_conflicted:
        print(f"  CONFLICT: {len(still_conflicted)} row(s) left untouched -- cached artifact's F1 disagrees "
              f"with the CSV's existing F1 (>0.02 abs), so the on-disk .pq is probably a different run "
              f"than what's in the CSV. Investigate, then re-run with --force-collections if the artifact "
              f"is the correct one: {still_conflicted}")
        for coll, col, old, new in conflicted:
            if coll in still_conflicted:
                print(f"    {coll} {col}: csv={old} recomputed_from_cached_artifact={new}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--csv", default=str(DEFAULT_CSV))
    p.add_argument("--project", default="merlin-eval-1.4",
                   help="Used only to derive the local artifact cache dir (data/results/evaluation/{version}).")
    p.add_argument("--force-collections", default="",
                   help="Comma-separated collection names known to have a stale CSV row (confirmed "
                        "the cached artifact, not the CSV, is correct): overwrite ICD F1 Micro/Macro "
                        "too instead of skipping on drift.")
    args = p.parse_args()
    force = [c.strip() for c in args.force_collections.split(",") if c.strip()]
    main(Path(args.csv), args.project, force_collections=force)
