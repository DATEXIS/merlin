#!/usr/bin/env python3
"""Headline results figure for the paper (held-out TEST split).

    python -m src.eval.main_figure_plots

Base vs. \\merlin{} full fine-tuning per model size, with seed error bars,
plus a dashed reference line for the strongest untuned external baseline.

Differences from paper_plots.plot_main_figure_2x1 (which this deliberately
does not replace -- that one stays the general-purpose internal figure):

  * panel titles/axis labels use the paper's clinical vocabulary
    (symptoms / diagnoses / ICD codes), never the internal V1..V4 stage
    names;
  * the best untuned external competitor is drawn as a horizontal reference
    line, so "a fine-tuned 8B beats every untuned 27-70B baseline" is
    readable off the figure instead of only off the table;
  * error bars are the std across the eval-repeat seeds (42/43/44) and are
    only drawn where more than one seed exists.

Everything configurable lives in main_figure_config.py.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.eval import main_figure_config as cfg
from src.eval.paper_plots import _agg_seeds_by_size, _bar_mean_err, _size_sort_key, ERROR_BAR_KW

REPO_ROOT = Path(__file__).resolve().parents[2]
TEST_CSV = REPO_ROOT / "data" / "results" / "evaluation" / "checkpoint_metrics.csv"
OUT_DIR = REPO_ROOT / "figures" / "main"


def _load():
    df = pd.read_csv(TEST_CSV)
    if "seed" not in df.columns:
        df["seed"] = 42
    test = df[df["is_test"] == True]  # noqa: E712
    base = test[(test["is_base"] == True) & (test["is_external"] != True)]  # noqa: E712
    full = test[(test["mode"] == "full") & (test["is_external"] != True)]  # noqa: E712
    ext = test[test["is_external"] == True]  # noqa: E712
    return base, full, ext


def plot(out_dir: Path = OUT_DIR):
    base, full, ext = _load()
    sizes = sorted(full["model_size"].dropna().unique(), key=_size_sort_key)
    base_agg = _agg_seeds_by_size(base)
    full_agg = _agg_seeds_by_size(full)

    x = np.arange(len(sizes), dtype=float)
    fig, axes = plt.subplots(1, len(cfg.METRICS), figsize=(3.6 * len(cfg.METRICS), 2.9))
    axes = np.atleast_1d(axes)

    for ax, (metric, _) in zip(axes, cfg.METRICS):
        vals = []
        for i, size in enumerate(sizes):
            for agg, off, color in ((base_agg, -cfg.BAR_W / 2, cfg.BASE_COLOR),
                                    (full_agg, cfg.BAR_W / 2, cfg.FULL_COLOR)):
                hit = _bar_mean_err(agg, size, metric)
                if hit is None:
                    continue
                v, err = hit
                kw = dict(color=color, zorder=3)
                if err:
                    kw.update(yerr=err, error_kw=ERROR_BAR_KW)
                ax.bar(x[i] + off, v, cfg.BAR_W * 0.92, **kw)
                vals.append(v + (err or 0))

        if cfg.EXTERNAL_REFERENCE:
            ref = ext[ext["collection"].str.contains(cfg.EXTERNAL_REFERENCE, na=False)]
            if not ref.empty and metric in ref.columns:
                v = float(ref.iloc[0][metric])
                ax.axhline(v, color=cfg.REF_COLOR, linestyle="--", linewidth=1.1, zorder=2)
                ax.text(-0.55, v, cfg.EXTERNAL_LABEL, va="bottom", ha="left", fontsize=6.8,
                        color=cfg.REF_COLOR, zorder=5,
                        bbox=dict(facecolor="white", edgecolor="none", pad=0.6))
                vals.append(v)

        ax.set_xticks(x)
        ax.set_xticklabels([s.upper() for s in sizes], fontsize=8.5)
        ax.set_xlim(-0.6, len(sizes) - 0.4)
        ax.set_title(cfg.PANEL_TITLES.get(metric, metric), fontsize=10)
        ax.set_ylabel(cfg.PANEL_YLABELS.get(metric, metric), fontsize=9)
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
        fig.savefig(out_dir / f"{cfg.OUT_STEM}.{ext_}", dpi=cfg.DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"main figure: saved {cfg.OUT_STEM} to {out_dir}")


if __name__ == "__main__":
    plot()
