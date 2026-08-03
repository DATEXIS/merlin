#!/usr/bin/env python3
"""One-column methodology figure: 8B full fine-tuning, base vs. no-trace vs.
with-trace, on ICD F1 Macro (the paper's long-tail metric). TEST split,
3 eval seeds (42/43/44) per condition, error bars = std across seeds.

Same data source and row-naming convention as src/eval/robustness_plots.py
(data/eval_metrics_merlin-eval-1.4.csv, keyed by run short_name; the doubled
"-seedN" suffix on the seed43/44 rows is a naming quirk from
src/wandb/run.py, harmless since we key off the row name directly):
    base       -> test-8b-base(-seed43-seed43 / -seed44-seed44)         untuned
    no trace   -> test-8b-mimic-icd2-e4(-seed43-e4-seed43 / ...-seed44)  label-only + ICD x2
    with trace -> test-8b-full-e3(-seed43-e3-seed43 / ...-seed44)       thrfull-icd2 + ICD x2, i.e. MERLIN

Both fine-tuned conditions use the ICD x2 oversampling setting and their
dev-selected epoch (no-trace: e4, with-trace: e3 -- see
src/eval/ablation_plots.py / table_ablation), so the only factor that
differs between them is whether reasoning traces were used.
"""
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from src.eval.checkpoint_plots import BASE_COLOR, LORA_COLOR, FULL_COLOR

REPO_ROOT = Path(__file__).resolve().parents[2]
IN_CSV = REPO_ROOT / "data" / "eval_metrics_merlin-eval-1.4.csv"
OUT_DIR = REPO_ROOT / "paper" / "figures" / "main"

METRIC = "ICD F1 Macro"

ROWS = {
    "base": ["test-8b-base", "test-8b-base-seed43-seed43", "test-8b-base-seed44-seed44"],
    "no trace": ["test-8b-mimic-icd2-e4", "test-8b-mimic-icd2-seed43-e4-seed43",
                 "test-8b-mimic-icd2-seed44-e4-seed44"],
    "with trace": ["test-8b-full-e3", "test-8b-full-seed43-e3-seed43", "test-8b-full-seed44-e3-seed44"],
}

# Reuse the paper's one true base/lora/full palette (src/eval/checkpoint_plots.py,
# also used by main_figure_plots.py / robustness_plot_config.py) instead of the
# separate blue/orange pairing ablation_plot_config.py uses -- Jan wants a single
# consistent scheme across figures. "no trace" takes the LORA slot color since
# it's the secondary/baseline condition here, "with trace" takes FULL_COLOR
# since it's the headline MERLIN recipe, same as everywhere else full-FT wins.
NO_TRACE_COLOR = LORA_COLOR
WITH_TRACE_COLOR = FULL_COLOR
HIGHLIGHT_EDGE = "#1F1F1F"
COLORS = {"base": BASE_COLOR, "no trace": NO_TRACE_COLOR, "with trace": WITH_TRACE_COLOR}
HIGHLIGHT = {"base": False, "no trace": False, "with trace": True}
# Error-bar halo: a thin dark line alone disappears against BASE_COLOR
# (#4D4D4D is already dark grey), so draw a thicker white line underneath it
# and a thin black line on top -- shows up against any bar color and against
# the white page background above the bars.
ERR_HALO_COLOR = "white"
ERR_LINE_COLOR = "#1A1A1A"


def load_values(in_csv: Path = IN_CSV):
    df = pd.read_csv(in_csv, index_col=0)
    out = {}
    for label, rows in ROWS.items():
        vals = df.loc[rows, METRIC].to_numpy(dtype=float)
        mean = vals.mean()
        std = vals.std(ddof=1)
        out[label] = (mean, std, COLORS[label], HIGHLIGHT[label])
    return out


def plot():
    data = load_values()
    labels = list(data.keys())
    means = [v[0] for v in data.values()]
    stds = [v[1] for v in data.values()]
    colors = [v[2] for v in data.values()]
    highlight = [v[3] for v in data.values()]

    # single-column width (~3.3in is standard for ACL-style two-column papers);
    # a bit wider than the top-legend version since the legend now sits
    # beside the axes rather than above them.
    fig, ax = plt.subplots(figsize=(3.9, 2.6))

    x = list(range(len(labels)))
    for xi, mean, std, color, hl in zip(x, means, stds, colors, highlight):
        kw = dict(width=0.6, zorder=3)
        if hl:
            kw.update(edgecolor=HIGHLIGHT_EDGE, linewidth=1.4)
        ax.bar(xi, mean, color=color, **kw)
        # halo trick: thick white line first, thin dark line on top -- keeps
        # the error bar legible over the dark grey base bar as well as over
        # the plain white background above each bar
        ax.errorbar(xi, mean, yerr=std, fmt="none", ecolor=ERR_HALO_COLOR,
                     elinewidth=3.2, capsize=5, capthick=3.2, zorder=4)
        ax.errorbar(xi, mean, yerr=std, fmt="none", ecolor=ERR_LINE_COLOR,
                     elinewidth=1.1, capsize=4, capthick=1.1, zorder=5)
        ax.text(xi, mean + std + 0.4, f"{mean:.1f}", ha="center", va="bottom",
                fontsize=8.5, zorder=6)

    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("ICD F1 macro (%)", fontsize=9)
    ax.set_title("8B, full fine-tuning — TEST, 3 seeds", fontsize=9, loc="left")
    ax.set_ylim(0, max(m + s for m, s in zip(means, stds)) * 1.32)
    ax.tick_params(axis="y", labelsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", linestyle=":", linewidth=0.6, alpha=0.5, zorder=0)

    # xtick labels already name each bar; the legend just reinforces the
    # color coding (base=grey, no trace=blue, with trace=orange/highlighted)
    # for standalone use.
    handles = [
        plt.Rectangle((0, 0), 1, 1, color=BASE_COLOR, label="base"),
        plt.Rectangle((0, 0), 1, 1, color=NO_TRACE_COLOR, label="no trace"),
        plt.Rectangle((0, 0), 1, 1, facecolor=WITH_TRACE_COLOR, edgecolor=HIGHLIGHT_EDGE,
                      linewidth=1.0, label="with trace"),
    ]
    ax.legend(handles=handles, loc="center left", bbox_to_anchor=(1.0, 0.5),
              ncol=1, frameon=False, fontsize=7.8, handlelength=1.2)

    fig.tight_layout()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(OUT_DIR / f"methodology_bars.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("saved methodology_bars.png / .pdf")
    print(data)


if __name__ == "__main__":
    plot()
