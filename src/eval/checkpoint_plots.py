#!/usr/bin/env python3
"""Bar chart(s) from data/results/evaluation/checkpoint_metrics.csv
(produced by checkpoint_metrics.py). Run after that script.

Only 8b gets a quick-look bar chart (dataset_ablation_8b_dev.png) -- it's
the only size that trained more than 1-2 datasets, so it's the only size
where a per-dataset comparison is actually meaningful (see FAMILY_INFO in
checkpoint_metrics.py). Untuned base models have no epoch, so instead of a
bar they're drawn as a horizontal dashed reference line on the chart.

(Cleaned up 2026-07-24,: the per-size best-epoch bar charts for
0.6b/14b/32b weren't useful -- each of those sizes only trains 1-2 datasets,
so the bars didn't show a real ablation, just noise -- removed, only 8b's
stays. The epoch-lines charts (one connected line per dataset/mode across
epochs) were removed entirely, plots and code -- with only 3-4 checkpoints
per run a connected line implies a continuous trend that isn't really there;
src/eval/paper_plots.py's epoch_analyses bar charts are the intended way to
look at the epoch story.)

Metrics shown (in this order): ICD F1 Macro, ICD F1 Micro, V2 F1 Macro@1,
V2 F1 Micro@1. To change which metrics get plotted, either edit
DEFAULT_METRIC_SELECTION below (picks from ALL_METRICS -- every column
checkpoint_metrics.py carries over from evaluate_experiment()),
or override per-run with --metrics, e.g.:

    python scripts/eval_analysis.py create_plots --plots quick --metrics "V2 MRR" DotProduct "ICD F1 Macro"

Any number of metrics works -- the subplot grid (see _subplot_grid) sizes
itself to however many are selected, it isn't locked to 2x2.

Checkpoint selection: the bar chart picks ONE checkpoint per dataset -- the
epoch that maximizes --primary-metric (default "ICD F1 Macro") -- and then
plots that same epoch's value across all four metrics, since in practice you
have to ship a single checkpoint, not a different one per metric. Use
--primary-metric to choose which metric drives that selection, e.g.:

    python scripts/eval_analysis.py create_plots --plots quick --primary-metric "V2 MRR"

In addition to the dataset-ablation figure, one extra summary figure
(checkpoint_best_per_size_dev.png) puts all model sizes on one set of axes:
for each size, a grouped bar pair -- that size's untuned base-model block
right next to its single best checkpoint overall (by --primary-metric,
across all of that size's datasets/epochs). Everything here is DEV-split
(see run()'s is_test filtering).

These are still the "quick look" analysis plots -- less polished than the
sibling src/eval/paper_plots.py figures, which imports several helpers/
constants from this module -- but as of 2026-07-24 they're written straight
to figures/ (dataset_analyses/, main/) rather than data/, alongside
those publication figures: plots belong in the paper section, not under
data/,. Only non-plot data (checkpoint_best_epochs.csv, the reshaped
checkpoint_metrics.csv) stays under data/results/evaluation/ (moved there
from data/checkpoint_analysis/ on 2026-07-24, alongside encoder_results/ ->
data/results/encoder_results/ -- that whole directory no longer exists).

Lives in src/eval/ (moved from scripts/ on 2026-07-15) alongside the rest of
the post-hoc eval/analysis code. `scripts/eval_analysis.py create_plots` is the
CLI entry point -- run that (or
`python -m src.eval.checkpoint_plots`) rather than this file
directly with a plain `python src/eval/...`, since this module has no
`from src...` imports of its own but its sibling paper_plots.py does,
and staying consistent about how these get invoked avoids confusion.
"""
import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

REPO = Path(__file__).resolve().parents[2]
OUT_DIR = REPO / "data" / "results" / "evaluation"  # non-plot data only (checkpoint_best_epochs.csv); IN_CSV lives here too
IN_CSV = OUT_DIR / "checkpoint_metrics.csv"

# Plots go under figures/ (not data/) alongside the polished
# paper_plots.py figures -- same subfolder names/purpose as that module's
# MAIN_DIR/DATASET_ANALYSES_DIR, kept as separate constants here (rather than
# imported) since paper_plots.py imports FROM this module, not the reverse.
FIGURES_ROOT = REPO / "figures"
MAIN_DIR = FIGURES_ROOT / "main"
DATASET_ANALYSES_DIR = FIGURES_ROOT / "dataset_analyses"

# Every metric column checkpoint_metrics.py carries over from
# evaluate_experiment() that's worth plotting (name -> axis label). This is
# the single place to add a new one.
ALL_METRICS = {
    "ICD F1 Macro": "ICD F1 Macro (%)",
    "ICD F1 Micro": "ICD F1 Micro (%)",
    "V2 F1 Macro@1": "V2 F1 Macro@1 (%)",
    "V2 F1 Micro@1": "V2 F1 Micro@1 (%)",
    "V2 MRR": "V2 MRR (%)",
    "V2 Accuracy@1": "V2 Accuracy@1 (%)",
    "V2 Recall@1": "V2 Recall@1 (%)",
    "V2 Recall@3": "V2 Recall@3 (%)",
    "V2 Recall@5": "V2 Recall@5 (%)",
    "V2 Recall@10": "V2 Recall@10 (%)",
    "CosSim": "CosSim (v1)",
    "DotProduct": "DotProduct (v1)",
    "V1 JSON Valid Rate": "V1 JSON Valid Rate (%)",
    "V2 JSON Valid Rate": "V2 JSON Valid Rate (%)",
    "V4 JSON Valid Rate": "V4 JSON Valid Rate (%)",
}
METRIC_NAMES = list(ALL_METRICS)  # valid --metrics / --primary-metric choices

# This is the easy-to-edit knob: which metrics show up by default, and in
# what order, across every figure in this module and in paper_plots.py.
# Override per-run with --metrics instead of editing this if you just want a
# one-off look at something else.
DEFAULT_METRIC_SELECTION = ["ICD F1 Macro", "ICD F1 Micro", "V2 F1 Macro@1", "V2 F1 Micro@1"]
DEFAULT_PRIMARY_METRIC = "ICD F1 Macro"


def build_metrics(names: list = None) -> list:
    """Turn a list of metric names into the (name, ylabel) pairs every plot
    function here takes, validating against ALL_METRICS. `names=None` uses
    DEFAULT_METRIC_SELECTION."""
    names = names if names is not None else DEFAULT_METRIC_SELECTION
    unknown = [n for n in names if n not in ALL_METRICS]
    if unknown:
        raise ValueError(f"Unknown metric(s) {unknown} -- choices are {METRIC_NAMES}")
    return [(n, ALL_METRICS[n]) for n in names]


# Module-level default, used by anything that doesn't get an explicit
# `metrics` argument (e.g. quick interactive use). main() always builds and
# passes its own list from --metrics, so a --metrics override never depends
# on this being mutated.
METRICS = build_metrics()


def _subplot_grid(n: int, cell_size=(6.0, 4.5)):
    """Fig + a flat array of exactly n axes, laid out at up to 2 columns
    (matches the original hardcoded 2x2 for the n=4 default) with any extra
    grid cells hidden. Scales to however many metrics are selected instead of
    assuming there are always exactly 4."""
    ncols = 2 if n > 1 else 1
    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(cell_size[0] * ncols, cell_size[1] * nrows))
    axes = np.atleast_1d(axes).flatten()
    for ax in axes[n:]:
        ax.axis("off")
    return fig, axes[:n]
COLORS = {
    "icd2": "#4C72B0", "thrfull-icd2": "#DD8452",
    "mimic": "#55A868", "mimic-icd2": "#C44E52",
    "thrfulldrop": "#8172B2",
    # Added 2026-07-16: three more 8B full-FT dataset ablations
    # (replace/thr_half/thr_full -- see FAMILY_INFO in
    # checkpoint_metrics.py for what each maps to). Colors keep
    # the same seaborn "muted" palette as the entries above.
    "replace": "#937860", "thrhalf": "#DA8BC3", "thrfull": "#CCB974",
}
BASE_PALETTE = ["#333333", "#9467BD", "#8C564B", "#7F7F7F", "#BCBD22", "#17BECF"]

# Only 8b trained enough datasets to make a per-dataset bar chart meaningful
# -- every other size only trained 1-2 datasets. Matches paper_plots.py's
# DATASET_ANALYSES_SIZES.
DATASET_ABLATION_SIZES = ["8b"]


def _base_color_map(base: pd.DataFrame) -> dict:
    """Assign each base-model collection a stable color, shared across all figures."""
    collections = sorted(base["collection"].dropna().unique())
    return {c: BASE_PALETTE[i % len(BASE_PALETTE)] for i, c in enumerate(collections)}


def _draw_base_lines(ax, base: pd.DataFrame, base_colors: dict, metric: str):
    for _, row in base.iterrows():
        ax.axhline(row[metric], color=base_colors[row["collection"]], linestyle="--",
                    linewidth=1.3, label=f"{row['collection']} (untuned)")


def _size_sort_key(size: str) -> float:
    """'0.6b' -> 0.6, '8b' -> 8.0, '14b' -> 14.0, '32b' -> 32.0 -- so sizes sort
    numerically instead of as strings (where '14b' < '32b' < '8b')."""
    return float(size.rstrip("b"))


def plot_dataset_ablation(size: str, size_df: pd.DataFrame, base: pd.DataFrame, base_colors: dict,
                           primary_metric: str, metrics: list = None):
    """Quick-look dataset-ablation bar chart -- only called for size="8b" (the
    only size with more than 1-2 datasets trained, so the only size where a
    per-dataset comparison means anything; see module docstring). One
    checkpoint per dataset, chosen by primary_metric; every panel shows that
    same checkpoint's value for its own metric. DEV-split only (see run()).
    Saved to figures/dataset_analyses/, not data/."""
    metrics = metrics if metrics is not None else METRICS
    datasets = [d for d in COLORS if (size_df["dataset"] == d).any()]

    chosen = (size_df.loc[size_df.groupby("dataset")[primary_metric].idxmax()]
              .set_index("dataset").loc[datasets].reset_index())

    fig, axes = _subplot_grid(len(metrics))
    for ax, (metric, ylabel) in zip(axes, metrics):
        bars = ax.bar(chosen["dataset"], chosen[metric], color=[COLORS[d] for d in chosen["dataset"]])
        ymax = max(chosen[metric].max(), base[metric].max() if not base.empty else 0)
        ax.set_ylim(0, ymax * 1.22 if ymax else 1)
        for bar, ep in zip(bars, chosen["epoch"]):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + ymax * 0.03,
                    f"ep{int(ep)}", ha="center", va="bottom", fontsize=9)
        _draw_base_lines(ax, base, base_colors, metric)
        ax.set_title(metric)
        ax.set_ylabel(ylabel)
        ax.set_xlabel("dataset")
        ax.tick_params(axis="x", rotation=20)
        ax.spines[["top", "right"]].set_visible(False)
    if not base.empty:
        axes[0].legend(fontsize=8)
    fig.suptitle(f"{size} — dataset ablation (DEV split), checkpoint chosen by best {primary_metric} (v1.4)",
                 fontsize=13, y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    DATASET_ANALYSES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(DATASET_ANALYSES_DIR / f"dataset_ablation_{size}_dev.png", dpi=150)
    plt.close(fig)

    tag = f"+ base ref lines {list(base['collection'])}" if not base.empty else "(no base models found)"
    print(f"{size}: saved dataset-ablation chart for datasets {datasets} {tag}")


BASE_COLOR = "#4D4D4D"
LORA_COLOR = "#4C72B0"
FULL_COLOR = "#DD8452"
BLOCK_COLORS = {"base": BASE_COLOR, "lora": LORA_COLOR, "full": FULL_COLOR}
SLOT_OFFSETS = {"base": -0.28, "lora": 0.0, "full": 0.28}
SLOT_WIDTH = 0.26


def _best_by_size_for_mode(ckpt: pd.DataFrame, mode: str, primary_metric: str) -> pd.DataFrame:
    sub = ckpt[ckpt["mode"] == mode]
    if sub.empty:
        return sub
    return sub.loc[sub.groupby("model_size")[primary_metric].idxmax()].set_index("model_size")


def plot_all_sizes(ckpt: pd.DataFrame, base: pd.DataFrame, primary_metric: str, metrics: list = None):
    """One figure, grouped bars per model size, up to three blocks per group
    at fixed slots in a fixed order -- base | lora | full -- so the same slot
    always means the same thing across every size, and each block's color
    (not the dataset) is what tells them apart:
      - base (gray): that size's untuned base-model score.
      - lora (blue): that size's best LoRA-adapter checkpoint (by
        primary_metric), if any LoRA run exists for that size.
      - full (orange): that size's best full-fine-tune checkpoint (by
        primary_metric), if any full run exists for that size.
    A slot is simply skipped (left blank) for sizes that didn't train that
    mode -- e.g. 14b/32b (LoRA only) just get base+lora, no full block, while
    0.6b/8b (trained both) get all three so you can directly compare full vs.
    LoRA at that size instead of only seeing whichever mode happened to win
    overall. Which dataset a bar's score came from is not shown here -- see
    the per-size figures for that breakdown."""
    metrics = metrics if metrics is not None else METRICS
    sizes = sorted(ckpt["model_size"].dropna().unique(), key=_size_sort_key)
    base_by_size = base.set_index("model_size") if not base.empty else base
    full_by_size = _best_by_size_for_mode(ckpt, "full", primary_metric)
    lora_by_size = _best_by_size_for_mode(ckpt, "lora", primary_metric)

    x = np.arange(len(sizes))

    fig, axes = _subplot_grid(len(metrics))
    for ax, (metric, ylabel) in zip(axes, metrics):
        # First pass: draw every bar and remember (bar, epoch) for the
        # full/lora ones so labels can be placed using the final, correct ymax.
        all_vals = []
        labeled_bars = []
        for i, size in enumerate(sizes):
            if size in base_by_size.index:
                v = base_by_size.loc[size, metric]
                ax.bar(x[i] + SLOT_OFFSETS["base"], v, SLOT_WIDTH, color=BASE_COLOR, zorder=3)
                all_vals.append(v)
            for mode, tbl in (("lora", lora_by_size), ("full", full_by_size)):
                if size not in tbl.index:
                    continue
                row = tbl.loc[size]
                v = row[metric]
                bar = ax.bar(x[i] + SLOT_OFFSETS[mode], v, SLOT_WIDTH,
                              color=BLOCK_COLORS[mode], zorder=3)[0]
                all_vals.append(v)
                labeled_bars.append((bar, int(row["epoch"])))
        ymax = max(all_vals) if all_vals else 0
        ax.set_ylim(0, ymax * 1.25 if ymax else 1)
        for bar, epoch in labeled_bars:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + ymax * 0.03,
                    f"ep{epoch}", ha="center", va="bottom", fontsize=7)
        ax.set_xticks(x)
        ax.set_xticklabels(sizes)
        ax.set_title(metric)
        ax.set_ylabel(ylabel)
        ax.set_xlabel("model size")
        ax.spines[["top", "right"]].set_visible(False)
    # Figure-level legend instead of an in-axes one, so it doesn't overlap the
    # epoch labels above the bars. One entry per block color -- dataset isn't
    # shown here, so no per-dataset legend entries.
    handles = [Patch(facecolor=BASE_COLOR, label="base (untuned)")]
    if not lora_by_size.empty:
        handles.append(Patch(facecolor=LORA_COLOR, label="lora"))
    if not full_by_size.empty:
        handles.append(Patch(facecolor=FULL_COLOR, label="full"))
    fig.suptitle(f"Best checkpoint per model size (DEV split, by {primary_metric}): base vs. lora vs. full (v1.4)",
                 fontsize=13, y=0.99)
    fig.legend(handles=handles, fontsize=8, ncol=3,
               loc="lower center", bbox_to_anchor=(0.5, 0.9))
    fig.tight_layout(rect=[0, 0, 1, 0.85])
    MAIN_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(MAIN_DIR / "checkpoint_best_per_size_dev.png", dpi=150)
    plt.close(fig)
    both_modes_sizes = sorted(set(full_by_size.index) & set(lora_by_size.index)) \
        if not full_by_size.empty and not lora_by_size.empty else []
    print(f"all-sizes: saved summary chart for sizes {sizes} (full+lora both shown for {both_modes_sizes})")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--primary-metric", choices=METRIC_NAMES, default=DEFAULT_PRIMARY_METRIC,
        help=f"Metric used to pick the one checkpoint per dataset shown in the bar charts "
             f"(default: {DEFAULT_PRIMARY_METRIC!r}).",
    )
    parser.add_argument(
        "--metrics", nargs="+", choices=METRIC_NAMES, default=None, metavar="METRIC",
        help=f"Which metrics to plot, and in what order (space-separated). "
             f"Default: {DEFAULT_METRIC_SELECTION}. Choices: {METRIC_NAMES}.",
    )
    return parser.parse_args()


def run(primary_metric: str = DEFAULT_PRIMARY_METRIC, metric_names: list = None,
        out_dir=None, in_csv=None, figures_root=None):
    """Programmatic entry point (used by scripts/eval_analysis.py).
    `out_dir`/`in_csv` override the module-level defaults for this run --
    the plot functions read the module globals, so we rebind them here.
    `figures_root` overrides where plots land (default figures/,
    matching paper_plots.py's OUT_ROOT) -- kept separate from `out_dir`,
    which now only controls the non-plot CSV outputs under
    data/results/evaluation/."""
    global OUT_DIR, IN_CSV, FIGURES_ROOT, MAIN_DIR, DATASET_ANALYSES_DIR
    if out_dir is not None:
        OUT_DIR = Path(out_dir)
    if in_csv is not None:
        IN_CSV = Path(in_csv)
    if figures_root is not None:
        FIGURES_ROOT = Path(figures_root)
        MAIN_DIR = FIGURES_ROOT / "main"
        DATASET_ANALYSES_DIR = FIGURES_ROOT / "dataset_analyses"
    metrics = build_metrics(metric_names)

    if not IN_CSV.exists():
        print(f"{IN_CSV} not found -- run `scripts/eval_analysis.py new_evals` first.")
        return
    df = pd.read_csv(IN_CSV)

    # is_test rows (scripts/the test-split config's held-out
    # TEST-split sweep) are excluded here, not just is_base -- they share the
    # same (model_size, dataset, mode) keys as the dev sweep, so leaving them
    # in would let a test-split score win the per-size "best epoch" selection
    # in plot_all_sizes/plot_size and silently mix test numbers into what's
    # supposed to be a dev-only view. This module has no test-specific
    # figures (see src/eval/paper_plots.py for those).
    #
    # `base` must be dev-split only too (is_base & ~is_test): since
    # checkpoint_metrics.py's TEST_BASE_RE (2026-07-21) started tagging the
    # real test-{size}-base rows as is_base=True/is_test=True, filtering on
    # is_base alone gives two rows per size (dev base + test base) and
    # base_by_size.loc[size] downstream becomes ambiguous (crashes
    # plot_all_sizes). This module is dev-only, so drop the test-base rows.
    ckpt = df[~df["is_base"] & ~df["is_test"]].copy()
    base = df[df["is_base"] & ~df["is_test"]].copy()
    if ckpt.empty:
        print("No fine-tuned checkpoint rows found -- nothing to plot.")
        return

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    base_colors = _base_color_map(base)

    best_rows = []
    for size in sorted(ckpt["model_size"].dropna().unique()):
        size_df = ckpt[ckpt["model_size"] == size]
        # Dataset-ablation bar chart: only 8b trained enough datasets for a
        # real per-dataset comparison (see module docstring / DATASET_ABLATION_SIZES).
        if size in DATASET_ABLATION_SIZES:
            plot_dataset_ablation(size, size_df, base, base_colors, primary_metric, metrics)

        # checkpoint_best_epochs.csv covers every size regardless -- it's a
        # data table, not a plot, so it stays useful even for sizes with no
        # dedicated chart.
        datasets = [d for d in COLORS if (size_df["dataset"] == d).any()]
        for dataset in datasets:
            sub = size_df[size_df["dataset"] == dataset]
            chosen = sub.loc[sub[primary_metric].idxmax()]
            row = {"model_size": size, "dataset": dataset, "primary_metric": primary_metric,
                   "chosen_epoch": int(chosen["epoch"]), "collection": chosen["collection"]}
            for metric, _ in metrics:
                row[metric] = chosen[metric]
            best_rows.append(row)

    pd.DataFrame(best_rows).to_csv(OUT_DIR / "checkpoint_best_epochs.csv", index=False)

    plot_all_sizes(ckpt, base, primary_metric, metrics)

    if base.empty:
        print("Note: no base-model evals found -- charts have no reference lines "
              "(run scripts/eval_base_models.py, then re-run scripts/eval_analysis.py new_evals).")

    print(f"saved v1.4 charts (checkpoint chosen by {primary_metric}) to {FIGURES_ROOT} "
          f"+ best-epoch table to {OUT_DIR}")


def main():
    args = parse_args()
    run(args.primary_metric, args.metrics)


if __name__ == "__main__":
    main()
