#!/usr/bin/env python3
"""Fold the encoder-classifier baselines (BioClinical ModernBERT, PubMedBERT,
S-Proto -- added under data/results/encoder_results/ 2026-07-22, moved there
from data/checkpoint_analysis/encoder_results/ on 2026-07-24 along with the
rest of that directory -- it no longer exists)
into data/results/evaluation/results_test.csv (also moved there from
data/checkpoint_analysis/ on 2026-07-24 -- it's an evaluation-results table,
not a checkpoint-analysis intermediate), alongside the LLM
checkpoint/base/external rows produced by src/eval/checkpoint_metrics.py.

These encoders never touch the vLLM/wandb pipeline the rest of src/eval/
drives (no JSON generation, no QA stages) -- they're plain multi-label ICD
classifiers, evaluated straight from precomputed per-seed prediction parquet
files. `<model_export>_seed{1,2,3}_test_v4_preds.parquet` each hold one row
per test admission with the gold `ICD_CODES` list and the model's predicted
`v4_preds` list. `encoder_table_metrics.parquet` in the same directory is the
index of which models/seeds exist (encoder_model display name, model_export
prefix, prediction_files) -- this module reads that index rather than
hardcoding model names here.

ICD F1 Micro/Macro are recomputed from the raw per-seed parquets with this
repo's own src/eval/classification_metrics.calculate_icd_metrics_cpu (one
score per seed, averaged across the 3 seeds), for consistency with how every
other row in results_test.csv was scored.

NOTE (2026-07-22): this deliberately does NOT reuse
encoder_table_metrics.parquet's own Macro-F1 column. That column's Macro-F1
does not reproduce from calculate_icd_metrics_cpu on the per-seed files --
recomputing here gives a consistently ~37% higher Macro-F1 for all three
models, which looks like the upstream number macro-averaged over a larger
(training-time) label vocabulary than the one observed in this test split.
Since that vocabulary isn't recoverable from the files here, we chose to
recompute with this repo's own metric fn rather than import a number that
isn't apples-to-apples with the rest of the table -- so the Macro-F1 written
here WILL differ from encoder_table_metrics.parquet / table_encoders.tex in
the paper. Micro-F1 reproduces exactly either way (confirmed to 4 decimals),
so that one's unaffected by this choice.

Micro/Macro AUROC (also in encoder_table_metrics.parquet) are intentionally
NOT added to results_test.csv -- that table has no AUROC column for any LLM
row (the generation pipeline never computes it), and the raw per-seed parquet
files only carry hard label predictions (no scores), so AUROC can't be
recomputed here anyway. AUROC stays in encoder_table_metrics.parquet / the
paper's table_encoders.tex only.

Run: python -m src.eval.encoder_metrics
     python scripts/eval_analysis.py encoder_results   (via scripts/eval_analysis.py)
"""
import re
from pathlib import Path

import pandas as pd

from src.eval.classification_metrics import calculate_icd_metrics_cpu

REPO = Path(__file__).resolve().parents[2]
DEFAULT_RESULTS_DIR = REPO / "data" / "results" / "encoder_results"
DEFAULT_METRICS_FILE = "encoder_table_metrics.parquet"
DEFAULT_RESULTS_TEST_CSV = REPO / "data" / "results" / "evaluation" / "results_test.csv"
DEFAULT_ENCODER_SEEDS_CSV = REPO / "data" / "results" / "evaluation" / "encoder_seeds.csv"

ROUND_DIGITS = 2  # matches src/eval/eval_results.py's `round(v * 100, 2)` convention

# Columns that don't apply to encoder classifiers (no JSON generation, no QA
# stages) -- left NaN, same as the existing "is_base" rows leave model_name NaN.
NOT_APPLICABLE_COLUMNS = [
    "model_size", "dataset", "epoch",
    "V2 F1 Micro@1", "V2 F1 Macro@1", "V2 MRR", "V2 Accuracy@1",
    "V2 Recall@1", "V2 Recall@3", "V2 Recall@5", "V2 Recall@10",
    "CosSim", "DotProduct",
    "V1 JSON Valid Rate", "V2 JSON Valid Rate", "V4 JSON Valid Rate",
    "ICD len", "ICD label len",
]


def _collection_name(model_export: str) -> str:
    return f"test-encoder-{model_export.replace('_', '-')}"


def score_one_model(results_dir: Path, encoder_model: str, model_export: str,
                     prediction_files: list, per_seed_sink: list = None) -> dict:
    """Average per-seed ICD F1/Precision/Recall (Micro+Macro -- whatever keys
    calculate_icd_metrics_cpu returns) over every prediction file listed for
    this model in encoder_table_metrics.parquet.

    `per_seed_sink`, if given, collects the *unaveraged* per-seed scores.
    results_test.csv only ever stores one row per model, so the seed spread is
    lost there -- but the paper's main table reports mean +- std for every row
    (src/eval/paper_tables.py), and an encoder row without a std would be the
    only bare number in the column. The sink is written out separately by
    main() as ENCODER_SEEDS_CSV."""
    per_seed, n_samples = [], []
    for fname in prediction_files:
        df = pd.read_parquet(results_dir / fname)
        scores = calculate_icd_metrics_cpu(df["v4_preds"], df["ICD_CODES"])
        per_seed.append(scores)
        n_samples.append(len(df))
        if per_seed_sink is not None:
            per_seed_sink.append({
                "collection": _collection_name(model_export),
                "model_name": encoder_model,
                "prediction_file": fname,
                "n_samples": len(df),
                **{k: round(v * 100, ROUND_DIGITS) for k, v in scores.items()},
            })

    if len(set(n_samples)) != 1:
        raise ValueError(
            f"{model_export}: prediction files disagree on row count {n_samples} "
            f"({prediction_files}) -- refusing to average across mismatched splits."
        )

    metric_keys = per_seed[0].keys()
    averaged = {
        k: round(sum(s[k] for s in per_seed) / len(per_seed) * 100, ROUND_DIGITS)
        for k in metric_keys
    }

    return {
        "collection": _collection_name(model_export),
        "model_size": None,
        "dataset": None,
        "mode": "encoder",
        "epoch": None,
        "is_base": False,
        "is_test": True,
        "is_external": True,
        "model_name": encoder_model,
        **averaged,
        "n_samples": float(n_samples[0]),
    }


def compute_encoder_rows(results_dir: Path, metrics_file: str,
                          per_seed_sink: list = None) -> pd.DataFrame:
    index = pd.read_parquet(results_dir / metrics_file)
    rows = [
        score_one_model(results_dir, r.encoder_model, r.model_export,
                        list(r.prediction_files), per_seed_sink)
        for r in index.itertuples()
    ]
    return pd.DataFrame(rows)


def upsert_rows(results_test_csv: Path, new_rows: pd.DataFrame) -> pd.DataFrame:
    """Insert `new_rows` into results_test_csv, replacing any existing rows
    with the same `collection` (idempotent re-runs) and filling columns that
    don't apply to encoders (NOT_APPLICABLE_COLUMNS, see module docstring)."""
    existing = pd.read_csv(results_test_csv) if results_test_csv.exists() else pd.DataFrame()

    for col in NOT_APPLICABLE_COLUMNS:
        if col not in new_rows.columns:
            new_rows[col] = None

    if not existing.empty:
        # align column order/set to the existing file; add any missing cols as NaN
        for col in existing.columns:
            if col not in new_rows.columns:
                new_rows[col] = None
        new_rows = new_rows[existing.columns]
        existing = existing[~existing["collection"].isin(new_rows["collection"])]

    frames = [df for df in (existing, new_rows) if not df.empty]
    out = pd.concat(frames, ignore_index=True) if frames else new_rows
    out.to_csv(results_test_csv, index=False)
    return out


def main(config: dict = None):
    if config is None:
        config = {}
    results_dir = Path(config.get("results_dir", DEFAULT_RESULTS_DIR))
    if not results_dir.is_absolute():
        results_dir = REPO / results_dir
    metrics_file = config.get("metrics_file", DEFAULT_METRICS_FILE)
    results_test_csv = Path(config.get("results_test_csv", DEFAULT_RESULTS_TEST_CSV))
    if not results_test_csv.is_absolute():
        results_test_csv = REPO / results_test_csv

    if not (results_dir / metrics_file).exists():
        print(f"{results_dir / metrics_file} not found -- nothing to integrate.")
        return

    per_seed = []
    new_rows = compute_encoder_rows(results_dir, metrics_file, per_seed)
    out = upsert_rows(results_test_csv, new_rows)

    seeds_csv = Path(config.get("encoder_seeds_csv", DEFAULT_ENCODER_SEEDS_CSV))
    if not seeds_csv.is_absolute():
        seeds_csv = REPO / seeds_csv
    pd.DataFrame(per_seed).to_csv(seeds_csv, index=False)
    print(f"Wrote {len(per_seed)} per-seed row(s) -> {seeds_csv}")
    print(f"Wrote {len(new_rows)} encoder row(s) -> {results_test_csv} "
          f"({len(out)} row(s) total):")
    print(new_rows[["collection", "model_name", "ICD F1 Micro", "ICD F1 Macro",
                     "n_samples"]].to_string(index=False))


if __name__ == "__main__":
    import sys

    cfg = {}
    if len(sys.argv) > 1 and sys.argv[1] not in ("-h", "--help"):
        import yaml
        with open(sys.argv[1]) as f:
            full = yaml.safe_load(f) or {}
        cfg = full.get("Encoder", {})
    main(cfg)
