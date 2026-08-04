#!/usr/bin/env python3
"""ICD-3-digit long-tail figure: code rank (x, sorted by frequency) vs.
count (y, log scale), shaded by which codes account for the top N% vs. the
remaining long tail of samples. Reproduces the reference figure a colleague
printed from the paper (merlin_icd_3digit_longtail.pdf).

Source: a single-model eval.pq under data/results/evaluation/ (one row per
hadm_id -- one admission = one sample -- so no de-duplication is needed
beyond picking one eval file: ICD_CODES is a property of the dataset, not of
the model/checkpoint that happened to produce that particular eval run).
Non-"mimic" eval filenames are the MERLIN dataset.

Note: the printed reference shows ~1200 unique 3-digit codes; the default
input here (a single 917-admission dev-split eval) only has ~650. If the
printout used the full train+dev+test set, point `input` at a source that
covers all splits and the code count should scale up accordingly.

Config: the ICD_LONGTAIL dict at the top of scripts/dataset_analyses.py
(input pq, digit truncation, split fraction, colors) -- edit it there and
run:
    python scripts/dataset_analyses.py
This module also runs standalone with its own built-in DEFAULTS below via
`python -m src.eval.dataset_analyses.icd_longtail` (run from the repo root,
or with PYTHONPATH=. set).
"""
from pathlib import Path

import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

REPO = Path(__file__).resolve().parents[3]  # src/eval/dataset_analyses/this_file.py -> repo root
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
from src.eval.colors import BLUE, YELLOW, tint

OUT_ROOT = REPO / "figures" / "dataset_analyses"
DEFAULT_INPUT = REPO / (
    "data/results/evaluation/1_4/eval_results_8b-full-icd2-e1/"
    "eval_results_8b-full-icd2-e1.pq"
)

# Merged over by `cfg` in run() -- edit here, or override per-key via the
# ICD_LONGTAIL dict in scripts/dataset_analyses.py. Colors come from the
# house palette, tinted toward white since the
# brand colors are quite strong at full intensity for a dense bar plot.
DEFAULTS = {
    "input": str(DEFAULT_INPUT),
    "digits": 3,
    "split_fraction": 0.8,        # "80% of total samples" cutoff in the reference
    "majority_color": tint(BLUE, 0.5),    # codes making up the top split_fraction of samples
    "tail_color": tint(YELLOW, 0.1),      # remaining long-tail codes
    "out_stem": "icd_3digit_longtail",
}

DPI = 300


def load_icd_counts(pq_path, n_digits: int = 3) -> pd.Series:
    """One row per admission -> flatten ICD_CODES -> truncate to n_digits ->
    value_counts, sorted descending (rank 1 = most frequent)."""
    df = pd.read_parquet(pq_path)
    codes = np.concatenate(df["ICD_CODES"].to_numpy())
    return pd.Series(codes).str[:n_digits].value_counts()


def plot_longtail(counts: pd.Series, split_fraction: float = 0.8,
                   majority_color: str = "#6b6ecf", tail_color: str = "#e6a532",
                   out_stem: str = "icd_3digit_longtail", out_root: Path = None):
    out_root = Path(out_root) if out_root is not None else OUT_ROOT
    out_root.mkdir(parents=True, exist_ok=True)

    total = counts.sum()
    cum_frac = counts.cumsum() / total
    cutoff_rank = int(np.searchsorted(cum_frac.values, split_fraction) + 1)

    ranks = np.arange(1, len(counts) + 1)
    colors = [majority_color if r <= cutoff_rank else tail_color for r in ranks]

    # Sized to match the ~3.5-3.8in-per-panel / ~9-10pt-label convention used
    # by the other paper figures (ablation_plots, etc.) --
    # this figure now renders at roughly half a table* width in the appendix
    # (paired with table_longtail), not a full 8in-wide standalone figure, so
    # a canvas sized for the latter left its labels reading as too small.
    fig, ax = plt.subplots(figsize=(3.8, 2.9))
    ax.bar(ranks, counts.values, width=1.0, color=colors, edgecolor="none")
    ax.set_yscale("log")
    ax.set_xlabel("ICD code rank (sorted by frequency)", fontsize=9.5, fontweight="bold")
    ax.set_ylabel("Count (log scale)", fontsize=9.5, fontweight="bold")
    ax.set_xlim(0, len(counts))
    ax.tick_params(axis="both", labelsize=8)

    handles = [
        Patch(facecolor=majority_color, label=f"{int(round(split_fraction * 100))}% of total samples"),
        Patch(facecolor=tail_color, label=f"{int(round((1 - split_fraction) * 100))}% of total samples"),
    ]
    ax.legend(handles=handles, fontsize=7.5, loc="upper right", frameon=True)
    fig.tight_layout()

    for ext in ("png", "pdf"):
        fig.savefig(out_root / f"{out_stem}.{ext}", dpi=DPI, bbox_inches="tight")
    plt.close(fig)

    print(f"dataset-analyses: saved {out_stem}.{{png,pdf}} to {out_root} "
          f"({len(counts)} unique codes, cutoff rank for "
          f"{int(round(split_fraction * 100))}% of samples: {cutoff_rank})")


def run(cfg: dict = None, out_root=None):
    """Programmatic entry point -- merges `cfg` over DEFAULTS, loads the ICD
    counts, and writes the figure. Called by scripts/dataset_analyses.py."""
    merged = dict(DEFAULTS)
    if cfg:
        merged.update({k: v for k, v in cfg.items() if v is not None})
    counts = load_icd_counts(merged["input"], n_digits=merged["digits"])
    plot_longtail(
        counts,
        split_fraction=merged["split_fraction"],
        majority_color=merged["majority_color"],
        tail_color=merged["tail_color"],
        out_stem=merged["out_stem"],
        out_root=out_root,
    )


if __name__ == "__main__":
    run()
