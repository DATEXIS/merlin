#!/usr/bin/env python3
"""Main results figure: ICD macro-F1 by model size, base vs. MERLIN.

    python -m src.eval.main_figure_plots_macro

This is the figure the paper's \\S1 actually includes
(figures/main/main_icd_by_size_macro.pdf) -- the same base-vs-MERLIN
bars per model size as the 3x1 headline figure, but restricted to the
primary metric so it fits a single column.

Panel title deliberately does NOT carry a "(long-tail)" qualifier: macro-F1 is defined in the text, and the parenthetical
pre-announces an interpretation the figure itself does not show. Frequency
strata live in the head/body/tail figure instead.

Everything else (colors, bar width, DPI, data source) comes from
main_figure_config.py.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.eval import main_figure_config as cfg
from src.eval.plot_common import _agg_seeds_by_size, _bar_mean_err, _size_sort_key, ERROR_BAR_KW

REPO_ROOT = Path(__file__).resolve().parents[2]
TEST_CSV = REPO_ROOT / "data" / "results" / "evaluation" / "checkpoint_metrics.csv"
OUT_DIR = REPO_ROOT / "figures" / "main"

METRIC = "ICD F1 Macro"
PANEL_TITLE = "ICD codes: macro-F1"
PANEL_YLABEL = "macro-F1 (%)"
FIGSIZE = (4.2, 2.15)  # ~ the 1.7 aspect the paper's \columnwidth slot already had
OUT_STEM = cfg.OUT_STEM + "_macro"


def _load():
    df = pd.read_csv(TEST_CSV)
    if "seed" not in df.columns:
        df["seed"] = 42
    test = df[df["is_test"] == True]  # noqa: E712
    base = test[(test["is_base"] == True) & (test["is_external"] != True)]  # noqa: E712
    full = test[(test["mode"] == "full") & (test["is_external"] != True)]  # noqa: E712
    return base, full


def plot(out_dir: Path = OUT_DIR):
    base, full = _load()
    sizes = sorted(full["model_size"].dropna().unique(), key=_size_sort_key)
    base_agg = _agg_seeds_by_size(base)
    full_agg = _agg_seeds_by_size(full)

    x = np.arange(len(sizes), dtype=float)
    fig, ax = plt.subplots(figsize=FIGSIZE)

    vals = []
    for i, size in enumerate(sizes):
        for agg, off, color in ((base_agg, -cfg.BAR_W / 2, cfg.BASE_COLOR),
                                (full_agg, cfg.BAR_W / 2, cfg.FULL_COLOR)):
            hit = _bar_mean_err(agg, size, METRIC)
            if hit is None:
                continue
            v, err = hit
            kw = dict(color=color, zorder=3)
            if err:
                kw.update(yerr=err, error_kw=ERROR_BAR_KW)
            ax.bar(x[i] + off, v, cfg.BAR_W * 0.92, **kw)
            vals.append(v + (err or 0))

    ax.set_xticks(x)
    ax.set_xticklabels([s.upper() for s in sizes], fontsize=8.5)
    ax.set_xlim(-0.6, len(sizes) - 0.4)
    ax.set_title(PANEL_TITLE, fontsize=10)
    ax.set_ylabel(PANEL_YLABEL, fontsize=9)
    ax.set_ylim(0, max(vals) * 1.18 if vals else 1)
    ax.tick_params(axis="y", labelsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", linestyle=":", linewidth=0.6, alpha=0.5, zorder=0)

    handles = [plt.Rectangle((0, 0), 1, 1, color=cfg.BASE_COLOR, label="untuned base"),
               plt.Rectangle((0, 0), 1, 1, color=cfg.FULL_COLOR, label="+ MERLIN (full FT)")]
    fig.legend(handles=handles, fontsize=8.5, ncol=2, loc="upper center",
               bbox_to_anchor=(0.5, 1.05), frameon=False)
    fig.tight_layout(rect=[0, 0, 1, 0.93])

    out_dir.mkdir(parents=True, exist_ok=True)
    for ext_ in ("png", "pdf"):
        fig.savefig(out_dir / f"{OUT_STEM}.{ext_}", dpi=cfg.DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"main figure (macro only): saved {OUT_STEM} to {out_dir}")


main = plot  # standard entry point, used by scripts/eval_analysis.py


if __name__ == "__main__":
    main()
