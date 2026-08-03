#!/usr/bin/env python3
"""EXPLORATORY variant of main_figure_plots.py (2026-07-30).

Same 3x1 headline figure, but with the 8B "no trace" leg of the trace
ablation squeezed in as a third bar -- 8B only, so that column has three
bars where every other size has two. Written as a separate script on
purpose: this is a look-and-see cut, not a replacement for
main_figure_plots.py, and nothing in the paper points here yet.

    python -m src.eval.main_figure_plots_notrace

The no-trace rows are NOT in checkpoint_metrics.csv -- they live in
data/eval_metrics_merlin-eval-1.4.csv keyed by short_name (see
memory: 8b-trace-ablation-test-seed-rows):
  test-8b-mimic-icd2-e4 / -seed43-e4-seed43 / -seed44-e4-seed44
i.e. label-only + ICD x2, epoch 4, TEST split, 3 seeds.

Bars are laid out per-size by count, so a 3-bar group stays centred on its
tick instead of overhanging into the neighbouring size.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.eval import main_figure_config as cfg
from src.eval.paper_plots import _agg_seeds_by_size, _bar_mean_err, _size_sort_key, ERROR_BAR_KW

REPO_ROOT = Path(__file__).resolve().parents[2]
TEST_CSV = REPO_ROOT / "data" / "results" / "evaluation" / "checkpoint_metrics.csv"
RAW_CSV = REPO_ROOT / "data" / "eval_metrics_merlin-eval-1.4.csv"
OUT_DIR = REPO_ROOT / "figures" / "main"

# --- the extra leg -------------------------------------------------------
NOTRACE_SIZE = "8b"
NOTRACE_ROWS = [
    "test-8b-mimic-icd2-e4",
    "test-8b-mimic-icd2-seed43-e4-seed43",
    "test-8b-mimic-icd2-seed44-e4-seed44",
]
NOTRACE_COLOR = "#C4A484"  # muted sibling of FULL_COLOR: same family, clearly the ablation
NOTRACE_LABEL = "+ MERLIN, no reasoning trace (8B)"
OUT_STEM = cfg.OUT_STEM + "_with_notrace"


def _load():
    df = pd.read_csv(TEST_CSV)
    if "seed" not in df.columns:
        df["seed"] = 42
    test = df[df["is_test"] == True]  # noqa: E712
    base = test[(test["is_base"] == True) & (test["is_external"] != True)]  # noqa: E712
    full = test[(test["mode"] == "full") & (test["is_external"] != True)]  # noqa: E712
    return base, full


def _notrace_stats():
    """{metric: (mean, std-or-None)} for the no-trace leg, over its 3 seeds."""
    raw = pd.read_csv(RAW_CSV, index_col=0)
    have = [r for r in NOTRACE_ROWS if r in raw.index]
    if not have:
        raise SystemExit(f"no no-trace rows found in {RAW_CSV.name}: {NOTRACE_ROWS}")
    sub = raw.loc[have]
    out = {}
    for metric, _ in cfg.METRICS:
        if metric not in sub.columns:
            continue
        vals = pd.to_numeric(sub[metric], errors="coerce").dropna()
        if vals.empty:
            continue
        out[metric] = (float(vals.mean()), float(vals.std(ddof=1)) if len(vals) > 1 else None)
    return out, len(have)


def plot(out_dir: Path = OUT_DIR):
    base, full = _load()
    notrace, n_seeds = _notrace_stats()
    sizes = sorted(full["model_size"].dropna().unique(), key=_size_sort_key)
    base_agg = _agg_seeds_by_size(base)
    full_agg = _agg_seeds_by_size(full)

    x = np.arange(len(sizes), dtype=float)
    fig, axes = plt.subplots(1, len(cfg.METRICS), figsize=(3.6 * len(cfg.METRICS), 2.9))
    axes = np.atleast_1d(axes)

    for ax, (metric, _) in zip(axes, cfg.METRICS):
        vals = []
        for i, size in enumerate(sizes):
            # (value, err, color) triples for this size, in draw order
            group = []
            for agg, color in ((base_agg, cfg.BASE_COLOR),):
                hit = _bar_mean_err(agg, size, metric)
                if hit is not None:
                    group.append((hit[0], hit[1], color))
            if size == NOTRACE_SIZE and metric in notrace:
                v, err = notrace[metric]
                group.append((v, err, NOTRACE_COLOR))
            hit = _bar_mean_err(full_agg, size, metric)
            if hit is not None:
                group.append((hit[0], hit[1], cfg.FULL_COLOR))

            # keep total group width constant across sizes -> a 3-bar group has
            # narrower bars than a 2-bar one, but occupies the same slot.
            n = len(group)
            total_w = cfg.BAR_W * 2
            w = total_w / n
            for j, (v, err, color) in enumerate(group):
                off = -total_w / 2 + w * (j + 0.5)
                kw = dict(color=color, zorder=3)
                if err:
                    kw.update(yerr=err, error_kw=ERROR_BAR_KW)
                ax.bar(x[i] + off, v, w * 0.92, **kw)
                vals.append(v + (err or 0))

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
               plt.Rectangle((0, 0), 1, 1, color=NOTRACE_COLOR, label=NOTRACE_LABEL),
               plt.Rectangle((0, 0), 1, 1, color=cfg.FULL_COLOR, label="+ MERLIN (full FT)")]
    fig.legend(handles=handles, fontsize=8.5, ncol=3, loc="upper center",
               bbox_to_anchor=(0.5, 1.06), frameon=False)
    fig.tight_layout(rect=[0, 0, 1, 0.92])

    out_dir.mkdir(parents=True, exist_ok=True)
    for ext_ in ("png", "pdf"):
        fig.savefig(out_dir / f"{OUT_STEM}.{ext_}", dpi=cfg.DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"exploratory main figure: saved {OUT_STEM} to {out_dir} "
          f"(no-trace leg from {n_seeds} seed(s))")


if __name__ == "__main__":
    plot()
