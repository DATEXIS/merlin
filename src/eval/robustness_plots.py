#!/usr/bin/env python3
"""Seed-robustness bar charts: N training seeds of the same recipe, eval'd
once each on the TEST split, per ablation ARM (see ARMS below).

Arm 1, "thrfull2icd" (the paper's headline run, "with reasoning traces"):
ft-8b-full-thrfull2icd-v1-4, paper seed 42 + two robustness-check seeds,
43/44 -- see run_plan.yaml runs 14-15. Each seed trained epochs=3 only
(the dev-selected best epoch for this model/dataset).

Arm 2, "mimic-icd2" (the traces-vs-no-traces ablation's other arm, "no
reasoning traces" / label-only): ft-8b-full-mimic-2icd-v1-4, seed 42 +
43/44 -- see run_plan.yaml runs 16-17. Each seed trained the full 4 epochs
(unlike thrfull2icd, epoch 4 is the dev-selected best here, not an early-stop
point -- we confirmed 2026-07-30 that epoch 4 is best for both new seeds
too, matching seed 42).

Metrics plotted and colors both come from src/eval/robustness_plot_config.py
-- edit that file, not this one, to change either. Metric selection matches
checkpoint_plots.DEFAULT_METRIC_SELECTION (the same 4 metrics every other
bar plot in the paper uses), so these figures stay comparable to the rest.

Data source: data/eval_metrics_merlin-eval-1.4.csv, rows named in each arm's
`rows` dict below. Both arms' seed-43/44 rows have the doubled "-seed{N}"
suffix (a naming quirk from src/wandb/run.py's automatic -seed{N} suffix
stacking with the short_name already containing it); harmless here since we
key off the row name directly instead of parsing it with
checkpoint_metrics.py's FAMILY_RE.

Outputs per arm (data/results/evaluation/, figures/main/):
  {csv_stem}_per_seed_metrics.csv / {csv_stem}_summary.csv, {fig_stem}.png/.pdf
thrfull2icd keeps its original filenames (csv_stem="robustness", i.e.
robustness_per_seed_metrics.csv / robustness_summary.csv, fig_stem=
"robustness_seed_variance") for backward compat with paper/
the revision notes and any .tex that already points at them.

Plus one combined figure once every arm in COMBINED_ARM_ORDER has run: a 1
row x N-arm panel (repo's "2x1" convention -- see paper_plots.
plot_main_figure_2x1), shared y-axis, so the with-traces/no-traces arms are
directly comparable side by side: figures/main/
robustness_seed_variance_2x1.png/.pdf. No separate CSV -- the two per-arm
CSVs above already have the underlying numbers.

Run as `python -m src.eval.robustness_plots` from repo root (same
convention as paper_plots.py -- this module does `from src...` imports so a
plain `python src/eval/robustness_plots.py` won't resolve them). Regenerates
every arm in ARMS by default (which also produces the combined figure);
`--arm mimic-icd2` runs just one arm and skips the combined figure.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.eval.robustness_plot_config import (
    METRICS, BAR_COLOR, SEED_COLORS, SEED_MARKERS, PAPER_SEED,
    ARM_COLORS, GROUPED_SEED_MARKERS, GROUPED_SEED_MARKER_COLOR, GROUPED_SEED_MARKER_EDGE,
)

REPO = Path(__file__).resolve().parents[2]
IN_CSV = REPO / "data" / "eval_metrics_merlin-eval-1.4.csv"
OUT_FIG_DIR = REPO / "figures" / "main"
OUT_DATA_DIR = REPO / "data" / "results" / "evaluation"

ARMS = {
    "thrfull2icd": dict(
        rows={
            42: "test-8b-full-e3",
            43: "8b-full-thrfull-icd2-seed43-e3-seed43",
            44: "8b-full-thrfull-icd2-seed44-e3-seed44",
        },
        title="ft-8b-full-thrfull2icd-v1-4 (with traces) -- seed robustness "
              "(3 seeds, TEST split, epoch 3)",
        short_title="with traces (thrfull2icd, epoch 3)",
        legend_label="with traces (thrfull2icd, e3)",
        fig_stem="robustness_seed_variance",
        csv_stem="robustness",
    ),
    "mimic-icd2": dict(
        rows={
            42: "test-8b-mimic-icd2-e4",
            43: "test-8b-mimic-icd2-seed43-e4-seed43",
            44: "test-8b-mimic-icd2-seed44-e4-seed44",
        },
        title="ft-8b-full-mimic-2icd-v1-4 (no traces) -- seed robustness "
              "(3 seeds, TEST split, epoch 4)",
        short_title="no traces (mimic-icd2, epoch 4)",
        legend_label="no traces (mimic-icd2, e4)",
        fig_stem="robustness_seed_variance_mimic_icd2",
        csv_stem="robustness_mimic_icd2",
    ),
}

# Combined side-by-side figure (repo convention: "AxB" = A columns, B row(s)
# -- see paper_plots.plot_main_figure_2x1's plt.subplots(1, 2, ...)), one
# panel per arm sharing a y-axis so magnitudes are directly comparable.
COMBINED_ARM_ORDER = ["thrfull2icd", "mimic-icd2"]
COMBINED_FIG_STEM = "robustness_seed_variance_2x1"

# Single-panel grouped-bar alternative to the 2x1 (one Axes, clustered bars
# per metric instead of two side-by-side Axes) -- more compact, and puts the
# with/without-traces comparison directly adjacent per metric instead of
# split across two panels.
GROUPED_FIG_STEM = "robustness_seed_variance_grouped"


def load_per_seed(rows: dict, in_csv: Path = IN_CSV) -> pd.DataFrame:
    df = pd.read_csv(in_csv, index_col=0)
    sub = df.loc[list(rows.values())].copy()
    sub.index = list(rows.keys())
    sub.index.name = "seed"
    return sub


def write_csvs(sub: pd.DataFrame, csv_stem: str, out_dir: Path = OUT_DATA_DIR):
    """Per-seed raw values + summary stats (mean/std/min/max/range/cv_%) --
    the same shape as the original hand-produced robustness_per_seed_metrics
    .csv / robustness_summary.csv (2026-07-17), now generated here instead of
    out-of-band so a new arm doesn't need its own one-off notebook cell."""
    out_dir.mkdir(parents=True, exist_ok=True)
    per_seed_path = out_dir / f"{csv_stem}_per_seed_metrics.csv"
    sub.to_csv(per_seed_path)

    numeric = sub.select_dtypes(include=[np.number])
    summary = pd.DataFrame({
        "mean": numeric.mean(),
        "std": numeric.std(ddof=1),
        "min": numeric.min(),
        "max": numeric.max(),
    })
    summary["range"] = summary["max"] - summary["min"]
    summary["cv_%"] = (summary["std"] / summary["mean"]) * 100
    summary = summary.round(3)
    summary_path = out_dir / f"{csv_stem}_summary.csv"
    summary.to_csv(summary_path)

    print(f"Saved {per_seed_path}, {summary_path}")
    return per_seed_path, summary_path


def _draw_arm(ax, sub: pd.DataFrame, title: str, show_ylabel: bool = True, fontsize: int = 10):
    """Draw one arm's mean+-std bars / per-seed scatter onto an existing Axes.
    Shared by the single-arm and combined (2x1) figures so the two never
    drift apart visually."""
    names = [n for n, _ in METRICS]
    x = np.arange(len(names))

    means = sub[names].mean()
    stds = sub[names].std(ddof=1)

    # Solid FULL_COLOR fill (no alpha wash) so this bar matches every other
    # full-FT bar in the paper instead of reading as a separate, paler
    # palette. Error bar gets the same white-halo treatment as
    # methodology_bars_plot.py: a thick white line under a thin dark one, so
    # it stays legible wherever it crosses a seed marker or the bar edge.
    ax.bar(x, means, color=BAR_COLOR, edgecolor="#1F1F1F", linewidth=1.2, zorder=2)
    ax.errorbar(x, means, yerr=stds, fmt="none", ecolor="white",
                elinewidth=3.2, capsize=5, capthick=3.2, zorder=2.5)
    ax.errorbar(x, means, yerr=stds, fmt="none", ecolor="#1A1A1A",
                elinewidth=1.1, capsize=4, capthick=1.1, zorder=2.6)
    for seed in sub.index:
        jitter = (np.random.RandomState(seed).rand(len(names)) - 0.5) * 0.12
        ax.scatter(x + jitter, sub.loc[seed, names], color=SEED_COLORS[seed],
                   marker=SEED_MARKERS[seed], s=90 if seed == PAPER_SEED else 60,
                   edgecolor="black", linewidth=0.5, zorder=3,
                   label=f"seed {seed}" + (" (paper)" if seed == PAPER_SEED else ""))
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=20, ha="right")
    if show_ylabel:
        ax.set_ylabel("%")
    ax.set_title(title, fontsize=fontsize)
    ax.grid(axis="y", alpha=0.3, zorder=0)
    return names


def plot_robustness(sub: pd.DataFrame, title: str, fig_stem: str, out_dir: Path = OUT_FIG_DIR):
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    names = _draw_arm(ax, sub, title)
    ax.legend(loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.16), frameon=False, fontsize=9)
    fig.tight_layout()

    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / f"{fig_stem}.png", dpi=150, bbox_inches="tight")
    fig.savefig(out_dir / f"{fig_stem}.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_dir / f'{fig_stem}.png'} (+ .pdf), metrics: {names}")


def plot_robustness_combined(subs: dict, arm_order: list = None,
                             fig_stem: str = COMBINED_FIG_STEM, out_dir: Path = OUT_FIG_DIR):
    """Side-by-side (1 row x N arms) comparison, one panel per arm, sharing a
    y-axis so the two arms' magnitudes/error bars are directly comparable --
    same convention as paper_plots.plot_main_figure_2x1 (1 row, 2 columns)."""
    arm_order = arm_order or list(subs)
    n = len(arm_order)
    fig, axes = plt.subplots(1, n, figsize=(6.5 * n / 1.4, 4.5), sharey=True)
    if n == 1:
        axes = [axes]

    handles_labels = None
    for i, arm_name in enumerate(arm_order):
        arm = ARMS[arm_name]
        names = _draw_arm(axes[i], subs[arm_name], arm["short_title"], show_ylabel=(i == 0))
        if handles_labels is None:
            handles_labels = axes[i].get_legend_handles_labels()

    fig.suptitle("8B full-FT -- seed robustness, with vs. without reasoning traces "
                 "(3 seeds each, TEST split)", fontsize=11, y=1.06)
    fig.legend(*handles_labels, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.0),
               frameon=False, fontsize=9)
    fig.tight_layout()

    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / f"{fig_stem}.png", dpi=150, bbox_inches="tight")
    fig.savefig(out_dir / f"{fig_stem}.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_dir / f'{fig_stem}.png'} (+ .pdf), arms: {arm_order}")


def plot_robustness_grouped(subs: dict, arm_order: list = None,
                            fig_stem: str = GROUPED_FIG_STEM, out_dir: Path = OUT_FIG_DIR):
    """Single-panel clustered bar chart: for each metric, one bar per arm
    side by side. Color now encodes ARM (ARM_COLORS), not seed -- with only
    one bar color available in plot_robustness's per-arm figure, color was
    free for seed; here two arms share the panel, so color goes to arm and
    seed drops to marker SHAPE only (GROUPED_SEED_MARKERS), rendered in one
    neutral fill/edge (GROUPED_SEED_MARKER_COLOR/_EDGE) so a dot reads the
    same whichever bar it sits over. Two separate legends as a result: arm
    (colored patches) and seed (shape only, monochrome)."""
    names = [n for n, _ in METRICS]
    arm_order = arm_order or list(subs)
    n_arms = len(arm_order)
    x = np.arange(len(names))
    width = 0.8 / n_arms

    fig, ax = plt.subplots(figsize=(2.2 * len(names) + 1.5, 4.5))

    arm_patches = []
    seed_handles = {}
    for i, arm_name in enumerate(arm_order):
        sub = subs[arm_name]
        offset = (i - (n_arms - 1) / 2) * width
        xi = x + offset
        means = sub[names].mean()
        stds = sub[names].std(ddof=1)
        color = ARM_COLORS[arm_name]

        bars = ax.bar(xi, means, width=width * 0.92, color=color,
                      edgecolor="#1F1F1F", linewidth=1.0, zorder=2)
        ax.errorbar(xi, means, yerr=stds, fmt="none", ecolor="white",
                    elinewidth=2.6, capsize=4, capthick=2.6, zorder=2.5)
        ax.errorbar(xi, means, yerr=stds, fmt="none", ecolor="#1A1A1A",
                    elinewidth=1.0, capsize=3.5, capthick=1.0, zorder=2.6)
        arm_patches.append(bars[0])

        for seed in sub.index:
            jitter = (np.random.RandomState(seed * 100 + i).rand(len(names)) - 0.5) * width * 0.5
            sc = ax.scatter(xi + jitter, sub.loc[seed, names],
                           facecolor=GROUPED_SEED_MARKER_COLOR, edgecolor=GROUPED_SEED_MARKER_EDGE,
                           marker=GROUPED_SEED_MARKERS[seed], s=90 if seed == PAPER_SEED else 55,
                           linewidth=0.9, zorder=3)
            seed_handles[seed] = sc

    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=20, ha="right")
    ax.set_ylabel("%")
    ax.grid(axis="y", alpha=0.3, zorder=0)

    # Title + two legends stacked above the axes (title text via ax.text, not
    # ax.set_title, so its vertical position is controlled the same way as
    # the legends' bbox_to_anchor instead of matplotlib's separate title-pad
    # system -- avoids the two systems' spacing fighting each other).
    ax.text(0.5, 1.42, "8B full-FT -- seed robustness, with vs. without reasoning "
            "traces (3 seeds each, TEST split)", transform=ax.transAxes,
            ha="center", fontsize=10)

    arm_legend = ax.legend(
        arm_patches, [ARMS[a]["legend_label"] for a in arm_order],
        loc="upper center", ncol=n_arms, bbox_to_anchor=(0.5, 1.30), frameon=False, fontsize=9,
        title="arm (bar color)", title_fontsize=8)
    ax.add_artist(arm_legend)
    seeds_sorted = sorted(seed_handles)
    ax.legend(
        [seed_handles[s] for s in seeds_sorted],
        [f"seed {s}" + (" (paper)" if s == PAPER_SEED else "") for s in seeds_sorted],
        loc="upper center", ncol=len(seeds_sorted), bbox_to_anchor=(0.5, 1.14), frameon=False,
        fontsize=9, title="seed (marker shape)", title_fontsize=8)

    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / f"{fig_stem}.png", dpi=150, bbox_inches="tight")
    fig.savefig(out_dir / f"{fig_stem}.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_dir / f'{fig_stem}.png'} (+ .pdf), arms: {arm_order}")


def run_arm(arm_name: str):
    arm = ARMS[arm_name]
    sub = load_per_seed(arm["rows"])
    write_csvs(sub, arm["csv_stem"])
    plot_robustness(sub, arm["title"], arm["fig_stem"])
    return sub


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=list(ARMS), default=None,
                    help="Regenerate just one arm (default: all arms in ARMS). "
                         "The combined 2x1 figure only runs when every arm in "
                         "COMBINED_ARM_ORDER is included, so --arm skips it.")
    args = ap.parse_args()
    arm_names = [args.arm] if args.arm else list(ARMS)
    subs = {}
    for arm_name in arm_names:
        print(f"=== {arm_name} ===")
        subs[arm_name] = run_arm(arm_name)

    if all(a in subs for a in COMBINED_ARM_ORDER):
        print("=== combined (2x1) ===")
        plot_robustness_combined(subs, COMBINED_ARM_ORDER)
        print("=== combined (grouped, one panel) ===")
        plot_robustness_grouped(subs, COMBINED_ARM_ORDER)


if __name__ == "__main__":
    main()
