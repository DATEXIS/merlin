#!/usr/bin/env python3
"""8B data-construction ablation figure for the paper.

    python -m src.eval.ablation_plots

Reads data/results/evaluation/results_dev.csv (dev split, one row per
dataset/epoch) and picks each variant's dev-selected best epoch from
checkpoint_best_epochs.csv -- exactly the same selection rule as
table_ablation, so figure and table can never disagree.

Why this figure exists as its own module: a quick-look variant
draws one bar per dataset variant in arbitrary order, which hides the fact
that the eight variants are really a 2-factor design (trace quality filter x
ICD-step oversampling). Laid out as a factorial, the ablation's actual claim
-- filtering for traces that pass their verifier threshold raises
rare-diagnosis macro-F1 at BOTH oversampling settings -- is readable
directly off the chart. Layout, labels and colors are all in
ablation_plot_config.py; edit that file, not this one.

Note the `replace` variant's dev-selected best epoch is e1: epochs 2-4 of
that run collapse to near-zero (a known training instability, see
the ablation design). It is plotted as-is, consistent with
checkpoint_best_epochs.csv, and flagged in the paper caption.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.eval import ablation_plot_config as cfg

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = REPO_ROOT / "data" / "results" / "evaluation"
DEV_CSV = RESULTS_DIR / "results_dev.csv"
BEST_EPOCHS_CSV = RESULTS_DIR / "checkpoint_best_epochs.csv"
OUT_DIR = REPO_ROOT / "figures" / "dataset_analyses"


def _load() -> tuple[dict, float | dict]:
    """Return {dataset: {metric: value}} at each variant's dev-selected best
    epoch, plus the untuned 8B base row as a dict of the same shape."""
    dev = pd.read_csv(DEV_CSV)
    best = pd.read_csv(BEST_EPOCHS_CSV)
    best = best[best["model_size"] == cfg.SIZE]

    runs = dev[(dev["model_size"] == cfg.SIZE) & (dev["mode"] == cfg.MODE)]
    out = {}
    for _, row in best.iterrows():
        hit = runs[(runs["dataset"] == row["dataset"]) & (runs["epoch"] == row["chosen_epoch"])]
        if hit.empty:
            continue
        out[row["dataset"]] = hit.iloc[0].to_dict()

    base_rows = dev[(dev["model_size"] == cfg.SIZE) & (dev["is_base"] == True)]  # noqa: E712
    base = base_rows.iloc[0].to_dict() if not base_rows.empty else {}
    return out, base


def _bar(ax, x, value, color, width, label=None, hatch=None, highlight=False):
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    kw = dict(color=color, width=width, zorder=3, label=label)
    if hatch:
        kw.update(hatch=hatch, edgecolor="#6E6E6E", linewidth=0.8)
    if highlight:
        kw.update(edgecolor=cfg.HIGHLIGHT_EDGE, linewidth=1.4)
    ax.bar(x, value, **kw)
    return value


def plot(out_dir: Path = OUT_DIR):
    runs, base = _load()
    n_fam = len(cfg.FAMILIES)
    x = np.arange(n_fam, dtype=float)
    width = 0.34

    fig, axes = plt.subplots(1, len(cfg.METRICS), figsize=(3.5 * len(cfg.METRICS) + 1.4, 3.0))
    axes = np.atleast_1d(axes)

    for ax, (metric, ylabel) in zip(axes, cfg.METRICS):
        vals = []
        for i, (_, plain_ds, icd2_ds) in enumerate(cfg.FAMILIES):
            slots = []
            if plain_ds:
                slots.append((plain_ds, cfg.PLAIN_COLOR, None))
            ctrl = [d for d, (fam, col, _) in cfg.CONTROLS.items() if fam == i and col == "plain"]
            slots.extend((d, cfg.CONTROL_COLOR, "//") for d in ctrl)
            if icd2_ds:
                slots.append((icd2_ds, cfg.ICD2_COLOR, None))

            # centre the slots of this family on x[i]
            k = len(slots)
            offs = (np.arange(k) - (k - 1) / 2) * width
            for off, (ds, color, hatch) in zip(offs, slots):
                v = runs.get(ds, {}).get(metric)
                got = _bar(ax, x[i] + off, v, color, width * 0.92, hatch=hatch,
                           highlight=(ds == cfg.HEADLINE_DATASET))
                if got is None:
                    continue
                vals.append(got)
                if metric in cfg.VALUE_LABEL_METRICS:
                    ax.text(x[i] + off, got, f"{got:.1f}", ha="center", va="bottom",
                            fontsize=7.6, zorder=4)

        if base.get(metric) is not None and not np.isnan(base.get(metric, np.nan)):
            # labelled once in the figure legend, not per-axes, so the
            # annotation can never collide with a bar
            ax.axhline(base[metric], color=cfg.BASE_COLOR, linestyle="--", linewidth=1.1, zorder=2)
            vals.append(base[metric])

        ax.set_xticks(x)
        ax.set_xticklabels([f for f, _, _ in cfg.FAMILIES], fontsize=8.0)
        ax.set_xlim(-0.65, n_fam - 0.35)
        ax.set_title(cfg.PANEL_TITLES.get(metric, metric), fontsize=10)
        ax.set_ylabel(cfg.PANEL_YLABELS.get(metric, ylabel), fontsize=8.5)
        ax.set_ylim(0, max(vals) * 1.22 if vals else 1)
        ax.tick_params(axis="y", labelsize=8)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", linestyle=":", linewidth=0.6, alpha=0.5, zorder=0)

    handles = [
        plt.Rectangle((0, 0), 1, 1, color=cfg.PLAIN_COLOR, label="no ICD oversampling"),
        plt.Rectangle((0, 0), 1, 1, color=cfg.ICD2_COLOR, label="ICD rows $\\times$2"),
        plt.Rectangle((0, 0), 1, 1, facecolor=cfg.CONTROL_COLOR, hatch="//",
                      edgecolor="#6E6E6E", label="drop instead of fallback"),
        plt.Line2D([0], [0], color=cfg.BASE_COLOR, linestyle="--", linewidth=1.1,
                   label="untuned 8B"),
    ]
    fig.legend(handles=handles, fontsize=8.5, ncol=1, loc="center left",
               bbox_to_anchor=(0.85, 0.5), frameon=False)
    fig.tight_layout(rect=[0, 0, 0.86, 1])

    out_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(out_dir / f"{cfg.OUT_STEM}.{ext}", dpi=cfg.DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"ablation: saved {cfg.OUT_STEM} to {out_dir}")


main = plot  # standard entry point, used by scripts/eval_analysis.py


if __name__ == "__main__":
    main()
