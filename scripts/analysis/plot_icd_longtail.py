#!/usr/bin/env python3
"""ICD-3-digit long-tail plot: rank (x, codes sorted by frequency) vs.
count (y, log scale), shaded by which codes account for the top 80% vs.
the remaining 20% of samples. Reproduces merlin_icd_3digit_longtail.pdf.

Source: any single-model eval .pq under data/results/evaluation/1_4/
(non-mimic filename = Merlin dataset). Each file has one row per hadm_id
(one admission = one sample), so no de-duplication is needed beyond
picking one eval file -- ICD_CODES is a property of the dataset, not the
model/checkpoint.

Usage:
    python plot_icd_longtail.py
    python plot_icd_longtail.py --input <path-to-eval.pq> --out <path.png>
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[0]  # override via --input if run elsewhere
DEFAULT_INPUT = (
    "data/results/evaluation/1_4/eval_results_8b-full-icd2-e1/"
    "eval_results_8b-full-icd2-e1.pq"
)

# Colors picked to match the printed reference (matplotlib "tab10"-ish
# slate blue / amber pair) -- tweak these two lines to align with your
# colleague's printout.
MAJORITY_COLOR = "#6b6ecf"  # codes making up the top --split% of samples
TAIL_COLOR = "#e6a532"      # remaining long-tail codes
SPLIT_FRACTION = 0.8        # "80% of total samples" cutoff in the reference


def load_icd_counts(pq_path: str, n_digits: int = 3) -> pd.Series:
    """One row per admission -> flatten ICD_CODES -> truncate to n_digits
    -> value_counts, sorted descending (rank 1 = most frequent)."""
    df = pd.read_parquet(pq_path)
    codes = np.concatenate(df["ICD_CODES"].to_numpy())
    codes_short = pd.Series(codes).str[:n_digits]
    counts = codes_short.value_counts()  # already sorted descending
    return counts


def plot_longtail(counts: pd.Series, split_fraction: float = SPLIT_FRACTION,
                   out_path: str = "paper/figures/dataset_analyses"):
    total = counts.sum()
    cum_frac = counts.cumsum() / total
    # first rank where cumulative share crosses split_fraction
    cutoff_rank = int(np.searchsorted(cum_frac.values, split_fraction) + 1)

    ranks = np.arange(1, len(counts) + 1)
    colors = [MAJORITY_COLOR if r <= cutoff_rank else TAIL_COLOR for r in ranks]

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.bar(ranks, counts.values, width=1.0, color=colors, edgecolor="none")
    ax.set_yscale("log")
    ax.set_xlabel("ICD code rank (sorted by frequency)", fontsize=13, fontweight="bold")
    ax.set_ylabel("Count (log scale)", fontsize=13, fontweight="bold")
    ax.set_xlim(0, len(counts))

    from matplotlib.patches import Patch
    handles = [
        Patch(facecolor=MAJORITY_COLOR, label=f"{int(split_fraction*100)}% of total samples"),
        Patch(facecolor=TAIL_COLOR, label=f"{int((1-split_fraction)*100)}% of total samples"),
    ]
    ax.legend(handles=handles, fontsize=11, loc="upper right", frameon=True)

    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"saved {out_path}")
    print(f"n unique 3-digit codes: {len(counts)}, cutoff rank for "
          f"{int(split_fraction*100)}% of samples: {cutoff_rank}")
    return fig, ax


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default=DEFAULT_INPUT,
                         help="path to a single-model eval .pq with an ICD_CODES column")
    parser.add_argument("--out", default="icd_3digit_longtail.png")
    parser.add_argument("--digits", type=int, default=3)
    parser.add_argument("--split-fraction", type=float, default=SPLIT_FRACTION)
    args = parser.parse_args()

    counts = load_icd_counts(args.input, n_digits=args.digits)
    plot_longtail(counts, split_fraction=args.split_fraction, out_path=args.out)


if __name__ == "__main__":
    main()
