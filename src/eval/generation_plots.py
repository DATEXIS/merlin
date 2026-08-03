#!/usr/bin/env python3
"""Generation-dynamics figure for the appendix (fig:generation_plots).

2x2 grid, one panel per pipeline stage (V1-V4), one line per generator,
x-axis = generation-budget step (1-4 regeneration rounds), y-axis = that
stage's verifier score. Shows the loop's regeneration mechanism working:
sub-threshold traces get resampled and scores rise with budget, most sharply
for the weakest generator (medgemma-27b-it) on the weakest stage (V1).

Design matches Figure 7 (src/eval/epoch_tradeoff_plots.py): percent-scale
y-axis grounded at 0, one marker shape per line on top of color, solid
low-alpha gridlines, same font sizes and top-center single-row legend. Same
family of figure (line-per-model over an ordinal step axis), so it reads as
the same visual language rather than a one-off style.

Stages, titles, colors/markers and the (mostly logged, partly estimated)
scores all come from src/eval/generation_plots_config.py -- edit that file,
not this one; see its docstring for what's a real logged number vs. a PDF
read-off.

Run as `python -m src.eval.generation_plots` from repo root.
"""
from pathlib import Path

import matplotlib.pyplot as plt

from src.eval.generation_plots_config import (
    STEPS, STAGE_TITLES, STAGE_ORDER, XLABEL, YLABEL,
    MODEL_COLORS, MODEL_MARKERS, MODEL_ORDER,
    SCORES, FIGSIZE, DPI, OUT_STEM,
)

REPO = Path(__file__).resolve().parents[2]
OUT_DIRS = [REPO / "figures" / "main"]


def plot_generation_dynamics():
    fig, axes = plt.subplots(2, 2, figsize=FIGSIZE)

    for ax, stage in zip(axes.flat, STAGE_ORDER):
        for model in MODEL_ORDER:
            scores_pct = [s * 100 for s in SCORES[model][stage]]
            ax.plot(STEPS, scores_pct, color=MODEL_COLORS[model],
                    marker=MODEL_MARKERS[model], markersize=6, linewidth=1.8,
                    label=model, zorder=2)
        ax.set_title(STAGE_TITLES[stage], fontsize=9)
        ax.set_xlabel(XLABEL, fontsize=8)
        ax.set_ylabel(YLABEL, fontsize=8)
        ax.set_xticks(STEPS)
        ax.tick_params(labelsize=8)
        ax.grid(axis="y", alpha=0.3, zorder=0)
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_ylim(bottom=0)

    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(labels),
               bbox_to_anchor=(0.5, 1.04), frameon=False, fontsize=9)
    fig.tight_layout(rect=[0, 0, 1, 0.93])

    for out_dir in OUT_DIRS:
        out_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_dir / f"{OUT_STEM}.pdf", dpi=DPI, bbox_inches="tight")
        fig.savefig(out_dir / f"{OUT_STEM}.png", dpi=DPI, bbox_inches="tight")
        print(f"Saved {out_dir / (OUT_STEM + '.pdf')} (+ .png)")
    plt.close(fig)


def main():
    plot_generation_dynamics()


if __name__ == "__main__":
    main()
