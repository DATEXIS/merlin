#!/usr/bin/env python3
"""Epoch trade-off figure for the appendix (checkpoint selection).

Three panels over the DEV epoch sweep, one line per model size: the metric we
select on (ICD macro-F1) against the two quantities that pay for it
(diagnosis Recall@3, symptom-stage JSON validity). At 14B the trade is stark;
at 0.6B/8B it is mild. Backs the claim in body.tex Sec. "Diagnosis Ranking and
Symptom Extraction" that selecting by ICD macro-F1 is not free.

Rows, metrics and colors all come from src/eval/epoch_tradeoff_config.py --
edit that file, not this one.

Data source: data/eval_metrics_merlin-eval-1.4.csv (dev rows).

Run as `python -m src.eval.epoch_tradeoff_plots` from repo root.
"""
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

from src.eval.epoch_tradeoff_config import (
    EPOCHS, ROW_TEMPLATES, SELECTED_EPOCH, METRICS,
    SIZE_COLORS, SIZE_MARKERS, FIGSIZE,
)

REPO = Path(__file__).resolve().parents[2]
IN_CSV = REPO / "data" / "eval_metrics_merlin-eval-1.4.csv"
OUT_DIRS = [REPO / "paper" / "figures" / "main",
            REPO / "paper" / "EACL_2026_v2" / "figures"]


def load_sweeps(in_csv: Path = IN_CSV) -> dict:
    """{size: DataFrame indexed by epoch} for every configured size."""
    df = pd.read_csv(in_csv, index_col=0)
    out = {}
    for size, tmpl in ROW_TEMPLATES.items():
        rows = {e: tmpl.format(e=e) for e in EPOCHS}
        present = {e: r for e, r in rows.items() if r in df.index}
        missing = sorted(set(rows) - set(present))
        if missing:
            print(f"  warn: {size} missing epochs {missing}")
        sub = df.loc[list(present.values())].copy()
        sub.index = list(present.keys())
        sub.index.name = "epoch"
        out[size] = sub
    return out


def plot_tradeoff(sweeps: dict, out_dirs=OUT_DIRS):
    fig, axes = plt.subplots(1, len(METRICS), figsize=FIGSIZE)

    for ax, (col, title, ylabel) in zip(axes, METRICS):
        for size, sub in sweeps.items():
            if col not in sub.columns:
                continue
            ax.plot(sub.index, sub[col], color=SIZE_COLORS[size],
                    marker=SIZE_MARKERS[size], markersize=5, linewidth=1.8,
                    label=size, zorder=2)
            sel = SELECTED_EPOCH.get(size)
            if sel in sub.index:
                ax.scatter([sel], [sub.loc[sel, col]], s=150, facecolors="none",
                           edgecolors=SIZE_COLORS[size], linewidths=1.8, zorder=3)
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("training epoch", fontsize=8)
        ax.set_ylabel(ylabel, fontsize=8)
        ax.set_xticks(EPOCHS)
        ax.tick_params(labelsize=8)
        ax.grid(axis="y", alpha=0.3, zorder=0)
        ax.set_ylim(bottom=0)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(labels),
               bbox_to_anchor=(0.5, 1.06), frameon=False, fontsize=9)
    fig.tight_layout()

    for out_dir in out_dirs:
        out_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_dir / "epoch_tradeoff.png", dpi=200, bbox_inches="tight")
        fig.savefig(out_dir / "epoch_tradeoff.pdf", bbox_inches="tight")
        print(f"Saved {out_dir / 'epoch_tradeoff.pdf'} (+ .png)")
    plt.close(fig)


def main():
    plot_tradeoff(load_sweeps())


if __name__ == "__main__":
    main()
