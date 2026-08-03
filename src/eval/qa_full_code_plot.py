#!/usr/bin/env python3
"""ICD full-code specificity figure: does the model degrade gracefully from
category (3-char) down to the exact full code, and does that hold up on rare
codes (macro) as well as frequent ones (micro)? Companion to
scripts/qa_full_code.py, which must be run first -- it writes the input CSV
(data/qa/full_code_by_granularity.csv) and calls this module automatically.

    python -m src.eval.qa_full_code_plot

2x2 grid: rows are micro-F1 / macro-F1 (cfg.METRIC_TYPES), columns are
category / full code (cfg.PANEL_ORDER). Each cell is the same
base-vs-full-FT-per-size bar layout as src/eval/main_figure_plots.py.
Everything configurable lives in qa_full_code_plot_config.py.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.eval import qa_full_code_plot_config as cfg

REPO_ROOT = Path(__file__).resolve().parents[2]
IN_CSV = REPO_ROOT / "data" / "qa" / "full_code_by_granularity.csv"
OUT_DIR = REPO_ROOT / "figures" / "qa"


def plot(out_dir: Path = OUT_DIR, in_csv: Path = IN_CSV):
    df = pd.read_csv(in_csv)
    sizes = [s for s in cfg.SIZE_ORDER if s in set(df["size"])]
    x = np.arange(len(sizes), dtype=float)

    n_rows, n_cols = len(cfg.METRIC_TYPES), len(cfg.PANEL_ORDER)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.8 * n_cols, 2.8 * n_rows),
                              squeeze=False)

    for row_i, metric_type in enumerate(cfg.METRIC_TYPES):
        for col_i, level in enumerate(cfg.PANEL_ORDER):
            ax = axes[row_i, col_i]
            sub = df[(df["granularity"] == level) & (df["metric_type"] == metric_type)]
            vals = []
            for i, size in enumerate(sizes):
                for variant, off, color in (("base", -cfg.BAR_W / 2, cfg.BASE_COLOR),
                                            ("full", cfg.BAR_W / 2, cfg.FULL_COLOR)):
                    r = sub[(sub["size"] == size) & (sub["variant"] == variant)]
                    if r.empty:
                        continue
                    v = float(r.iloc[0][cfg.VALUE]) * 100
                    ax.bar(x[i] + off, v, cfg.BAR_W * 0.92, color=color, zorder=3)
                    vals.append(v)

            ax.set_xticks(x)
            ax.set_xticklabels([cfg.SIZE_LABELS.get(s, s) for s in sizes], fontsize=8.5)
            ax.set_xlim(-0.6, len(sizes) - 0.4)
            if row_i == 0:
                ax.set_title(cfg.PANEL_TITLES.get(level, level), fontsize=10)
            ax.set_ylabel(f"{cfg.METRIC_TYPE_LABELS.get(metric_type, metric_type)}-"
                           f"{cfg.VALUE.capitalize()} (%)", fontsize=9)
            ax.set_ylim(0, max(vals) * 1.18 if vals else 1)
            ax.tick_params(axis="y", labelsize=8)
            ax.spines[["top", "right"]].set_visible(False)
            ax.grid(axis="y", linestyle=":", linewidth=0.6, alpha=0.5, zorder=0)

    handles = [plt.Rectangle((0, 0), 1, 1, color=cfg.BASE_COLOR, label="untuned base"),
               plt.Rectangle((0, 0), 1, 1, color=cfg.FULL_COLOR, label="+ MERLIN (full FT)")]
    fig.legend(handles=handles, fontsize=8.5, ncol=2, loc="upper center",
               bbox_to_anchor=(0.5, 1.03), frameon=False)
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    out_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(out_dir / f"{cfg.OUT_STEM}.{ext}", dpi=cfg.DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"qa full-code figure: saved {cfg.OUT_STEM} to {out_dir}")


if __name__ == "__main__":
    plot()
