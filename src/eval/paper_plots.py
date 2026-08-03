#!/usr/bin/env python3
"""Publication-ready figures for the EACL 2027 submission, built on top of
data/results/evaluation/checkpoint_metrics.csv (see
checkpoint_metrics.py) and reusing the metric/color constants
from the "quick look" module src/eval/checkpoint_plots.py.

None of these use a line/connected-dot chart over epoch: with only 3-4
checkpoints per run, a connected line implies a continuous trend between
measurements that isn't actually there (that's real for something like a
model-size scaling curve on a log axis, but not for training epoch). Bars
and heatmaps only.

Nine figure sets, written to paper/figures/ (six over the DEV
sweep, plus a TEST-split pair and two TEST-split compact alternates
(3-panel and 2-panel) at the end):

1. main/checkpoint_best_per_size_2x2_val.{png,pdf} -- the DEV-split headline
   results figure: for each model size, base vs. best-LoRA vs. best-full-
   fine-tune grouped bars, one subplot per metric (2x2). Polished version of
   checkpoint_plots.plot_all_sizes: no title (papers get a LaTeX
   caption instead), narrower figure sized for a two-column ACL/EACL layout,
   saved as both a vector PDF (for \\includegraphics) and a PNG (for quick
   viewing here). (Renamed from checkpoint_best_per_size.{png,pdf} on
   2026-07-21 to make the DEV/TEST split explicit alongside
   checkpoint_best_per_size_2x2 below -- update any \\includegraphics still
   pointing at the old name.)

2. dataset_analyses/bars_8b_{full,lora}.{png,pdf} -- 2 figures, one per mode,
   2x2 metric grid with one bar per dataset trained at 8b in that mode, plus
   the 8b base score as a dashed reference line. 8b is the only size with a
   real dataset ablation to show (icd2, thrfull-icd2, mimic, mimic-icd2,
   thrfulldrop, replace, thrhalf, thrfull for full; thrfull-icd2, thrfulldrop
   for lora) -- every other size only trained 1-2 datasets per mode, so isn't
   included here.

3. epoch_analyses/epoch_bars_{size}_{mode}.{png,pdf} -- 8 figures (4 sizes x
   {full, lora}), 2x2 metric grid, bars over [base, e1, e2, ..., eN] for that
   size/mode's best dataset (the one plot_heatmap_size_x_epoch also picks by
   --primary-metric). Bars instead of a connected line over epoch, for the
   same reason as everything else here -- no implied trend between 3-4
   discrete points.

4. heatmaps/heatmap_size_x_mode.{png,pdf} and
   heatmaps/heatmap_size_x_epoch.{png,pdf} -- annotated heatmaps, 2x2
   metric grid each, for the same two comparisons in compact grid form:
     - size x mode: rows=size, cols=[base, lora, full], cell=that combo's
       best-epoch score (same numbers as figure 1).
     - size x epoch: rows=size+mode (up to 8), cols=epoch 1..N, cell=score
       at that epoch for that size/mode's best dataset (same numbers as
       figure 3, condensed into one grid instead of 8 separate charts).

5. heatmaps/heatmap_dataset_x_epoch_{size}_{mode}.{png,pdf} -- one
   annotated heatmap (2x2 metric grid) per (size, mode) that trained MORE
   THAN ONE dataset -- currently 8b-full (8 datasets) and 8b-lora/32b-lora (2
   datasets each). Rows=dataset, cols=epoch 1..N, cell=score at that epoch
   for that dataset. Same idea as dataset_analyses (only meaningful where a
   real dataset ablation exists) crossed with the epoch story from figure
   3/4, but per-dataset instead of collapsing to the single best one.

6. main/checkpoint_best_per_size_2x2.{png,pdf} (renamed 2026-07-21 from
   checkpoint_best_per_size_test.{png,pdf} -- update any \\includegraphics
   still pointing at the old name) and heatmaps/heatmap_size_x_mode_test.
   {png,pdf} -- the TEST-split counterparts of figures 1 and 4's size x mode
   heatmap: the dev-selected best checkpoint per size/mode, evaluated on
   held-out test exactly once. No "best epoch" selection here -- there's
   only ever one test row per size/mode, so (also as of 2026-07-21, per Jan)
   no epoch label is drawn above the bars either. The base bar/column reuses
   the real TEST-split base-model eval where available.

7. main/checkpoint_best_per_size_3x1.{png,pdf} -- a more compact TEST-split
   alternative to checkpoint_best_per_size_2x2 above: same base/lora/full
   bars, but only 3 metrics (ICD F1 Macro, ICD F1 Micro, and Diagnoses
   Recall@3 -- swapped from Diagnosis MRR on 2026-07-24, per Jan; see
   MAIN_3X1_METRICS) in this figure only) laid out as a single row
   of 3 panels instead of a 2x2 grid, so it's shorter vertically. Added
   2026-07-21 per Jan as a candidate replacement for the 2x2 version as the
   paper's main-results figure (saves vertical space) -- see
   paper/EACL_2026_revision_notes.md for which one body.tex currently uses.
   Also written as main/checkpoint_best_per_size_3x1_no_lora.{png,pdf} (same
   call, include_lora=False) -- base vs. full FT only, re-centered 2-slot
   bars instead of a gap where LoRA used to sit. Added the same day per Jan:
   LoRA's ranking vs. full FT isn't fully stable across metrics/sizes, so
   dropping it from the main figure and moving the LoRA-vs-full comparison
   to an ablation is an open question -- see the revision notes.
   Error bars added 2026-07-24, same seed-averaging as figure 8 below.

8. main/checkpoint_best_per_size_2x1.{png,pdf} -- added 2026-07-23 per Jan,
   superseding the 3x1 figure above as the main-results candidate: same
   TEST-split base/lora/full bars, but only 2 panels (ICD F1 Macro, ICD F1
   Micro -- MAIN_2X1_METRICS). V2 MRR is dropped outright, not swapped for
   another metric -- per EACL_2026_revision_notes.md's "drop-V2" decision,
   the other V2 metrics (F1@1, Accuracy@1) are being cut from the headline
   figure for the same reason, so there's nothing else in that family left
   to promote into the freed slot. Legend moved to the right side of the
   figure instead of top-center (reads better with only 2, wider panels).
   Also written as main/checkpoint_best_per_size_2x1_no_lora.{png,pdf}
   (include_lora=False), mirroring the 3x1_no_lora cut above.

   Error bars added 2026-07-24: each bar is the mean over every seed
   replicate available for that (model_size, mode) -- see
   scripts/eval_error_bars_config_seed{43,44}[.yaml|_8b.yaml] (re-evals the
   seed-42 checkpoints/base models at eval seeds 43/44, no retraining) and
   _agg_seeds_by_size below. Sizes/modes without a downloaded seed-43/44
   rerun still show a plain bar (n_seeds=1 -> no whisker). Figure 6
   (checkpoint_best_per_size_2x2 / plot_main_figure_test) gets the same
   treatment, sharing the same helper.

9. diagnostics/v2_recall_at_k.{png,pdf} -- added 2026-07-24 per Jan, renamed
   the same day from v2_ranking_recovery_2x2 when the metric set dropped
   MRR/F1 Macro@1 (both rank-1-only reads, same information as Recall@1)
   in favor of V2 Recall@1 / V2 Recall@3 / V2 Recall@5 -- one metric family
   at three k instead of mixing MRR/F1@1 with Recall@k (see
   V2_RANKING_DIAGNOSTIC_METRICS / plot_v2_recall_at_k for the full
   reasoning: V2's own candidate-acceptance threshold in
   src/pipeline/verifier.py is effectively a Recall@1 gate already, not a
   continuous MRR target, so Recall@1 is the metric that actually matches
   what curation optimizes for). 3 panels now, not 4 -- the "_2x2" stem was
   dropped since it no longer describes the layout. Checks whether the "V2
   gets worse on bigger models" pattern visible at Recall@1 survives past
   rank 1 -- for 32b (full+lora) and 14b-lora it mostly doesn't (Recall@3/@5
   catch back up), for 14b-full and 8b-lora it does (they stay behind at
   every k). One-off diagnostic, not a headline result -- written to
   diagnostics/ (same directory src/eval/json_validity_impact.py already
   uses for its own one-off figure), not main/. Also written as
   diagnostics/v2_recall_at_k_no_lora.{png,pdf} (include_lora=False) --
   base vs. full FT only, same 2-slot re-centered layout as
   checkpoint_best_per_size_3x1_no_lora/2x1_no_lora.

Which metrics appear (and in what order) is controlled the same way as in
checkpoint_plots.py -- edit DEFAULT_METRIC_SELECTION there, or
pass --metrics here for a one-off override, e.g.:
    python scripts/eval_analysis.py create_plots --metrics "V2 MRR" DotProduct

Every subplot grid in this module sizes itself to however many metrics are
selected (see _subplot_grid), so this isn't locked to exactly 4/a 2x2 layout.

This module imports shared constants/helpers from its sibling
checkpoint_plots, so it needs the repo root on sys.path to run.
`scripts/eval_analysis.py create_plots --plots paper` handles that for you --
use it instead of calling this file directly:
    python scripts/eval_analysis.py create_plots --plots paper
If you do want to invoke this module directly instead of via that wrapper,
run it as `python -m src.eval.paper_plots` or with PYTHONPATH=. set,
both from repo root -- a plain `python src/eval/paper_plots.py` will
fail with ModuleNotFoundError.

Run after scripts/eval_analysis.py new_evals (which runs
checkpoint_metrics.py itself, so usually nothing else is needed
first):
    python scripts/eval_analysis.py new_evals
    python scripts/eval_analysis.py create_plots --plots paper
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from src.eval.checkpoint_plots import (
    METRICS, ALL_METRICS, METRIC_NAMES, DEFAULT_METRIC_SELECTION, DEFAULT_PRIMARY_METRIC,
    build_metrics, COLORS,
    BASE_COLOR, LORA_COLOR, FULL_COLOR, BLOCK_COLORS,
    SLOT_OFFSETS, SLOT_WIDTH,
    _size_sort_key, _best_by_size_for_mode, _subplot_grid, IN_CSV,
)

REPO = Path(__file__).resolve().parents[2]
OUT_ROOT = REPO / "paper" / "figures"
MAIN_DIR = OUT_ROOT / "main"
DATASET_ANALYSES_DIR = OUT_ROOT / "dataset_analyses"
EPOCH_ANALYSES_DIR = OUT_ROOT / "epoch_analyses"
HEATMAP_DIR = OUT_ROOT / "heatmaps"
# Same directory src/eval/json_validity_impact.py already writes its one-off
# diagnostic figure to -- for figures that investigate an anomaly rather than
# report a headline/ablation result, so they don't clutter main/.
DIAGNOSTICS_DIR = OUT_ROOT / "diagnostics"

# Only 8b trained enough datasets per mode for a real ablation comparison
# (icd2, thrfull-icd2, mimic, mimic-icd2, thrfulldrop for full; thrfulldrop
# for lora) -- every other size only has 1-2 datasets per mode.
DATASET_ANALYSES_SIZES = ["8b"]

DPI = 300


def plot_main_figure(ckpt: pd.DataFrame, base: pd.DataFrame, primary_metric: str, metrics: list = None):
    """Headline results figure: base vs. lora vs. full, grouped bars, one
    subplot per metric. Same selection logic as
    checkpoint_plots.plot_all_sizes, restyled for print."""
    metrics = metrics if metrics is not None else METRICS
    sizes = sorted(ckpt["model_size"].dropna().unique(), key=_size_sort_key)
    base_by_size = base.set_index("model_size") if not base.empty else base
    full_by_size = _best_by_size_for_mode(ckpt, "full", primary_metric)
    lora_by_size = _best_by_size_for_mode(ckpt, "lora", primary_metric)

    x = np.arange(len(sizes))

    fig, axes = _subplot_grid(len(metrics), cell_size=(3.5, 2.7))
    for ax, (metric, ylabel) in zip(axes, metrics):
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
        ax.set_ylim(0, ymax * 1.28 if ymax else 1)
        for bar, epoch in labeled_bars:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + ymax * 0.025,
                    f"e{epoch}", ha="center", va="bottom", fontsize=6.5)
        ax.set_xticks(x)
        ax.set_xticklabels(sizes, fontsize=9)
        ax.set_title(metric, fontsize=10)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.tick_params(axis="both", labelsize=8)
        ax.spines[["top", "right"]].set_visible(False)

    handles = [Patch(facecolor=BASE_COLOR, label="base")]
    if not lora_by_size.empty:
        handles.append(Patch(facecolor=LORA_COLOR, label="LoRA"))
    if not full_by_size.empty:
        handles.append(Patch(facecolor=FULL_COLOR, label="full FT"))
    fig.legend(handles=handles, fontsize=9, ncol=3, loc="upper center",
               bbox_to_anchor=(0.5, 1.02), frameon=False)
    fig.tight_layout(rect=[0, 0, 1, 0.94])

    MAIN_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(MAIN_DIR / f"checkpoint_best_per_size_2x2_val.{ext}", dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"main: saved polished headline figure to {MAIN_DIR}")


def _test_table(test: pd.DataFrame, mode: str) -> pd.DataFrame:
    """One row per model_size for `mode` on the TEST split, canonical seed-42
    checkpoint only. Each size/mode trains exactly one checkpoint on test by
    design (a single dev-selected epoch per manifest -- see TEST_FAMILY_INFO
    in checkpoint_metrics.py), but a rerun could in principle leave two
    collections for the same size/mode in the input CSV -- keep the
    higher-epoch one rather than letting a non-unique index break the .loc
    lookups below.

    Pinning seed==42 (added 2026-07-24, alongside the eval-time seed-
    robustness sweep -- see EVAL_SEED_TEST_FAMILY_RE/EVAL_SEED_TEST_BASE_RE
    in checkpoint_metrics.py) keeps every figure OTHER than
    plot_main_figure_2x1/plot_main_figure_test reading exactly the one
    dev-selected checkpoint's score, same as before that sweep started adding
    seed-43/44 replicate rows under the same (model_size, mode) key -- without
    this filter, drop_duplicates(keep="last") would pick whichever seed's row
    happens to sort last (arbitrary, not necessarily seed 42) instead of the
    canonical one. `seed` defaults to 42 for every pre-existing row (see
    checkpoint_metrics.py), so this is a no-op on data that predates the
    sweep -- but see run() below, which backfills the column for CSVs written
    before it existed at all."""
    sub = test[(test["mode"] == mode) & (test["seed"] == 42)]
    sub = sub.sort_values("epoch").drop_duplicates("model_size", keep="last")
    return sub.set_index("model_size")


def _agg_seeds_by_size(df: pd.DataFrame) -> pd.DataFrame:
    """Collapse `df` (already filtered to one mode, or to the base-model
    rows, all TEST split) to one row per model_size by averaging over every
    seed replicate present: the canonical seed-42 checkpoint, plus any
    seed-43/44 reruns from scripts/eval_error_bars_config_seed{43,44}
    [.yaml|_8b.yaml] (see EVAL_SEED_TEST_FAMILY_RE/EVAL_SEED_TEST_BASE_RE in
    checkpoint_metrics.py). Used only by plot_main_figure_2x1/
    plot_main_figure_test -- every other test-split figure uses _test_table
    above instead, which stays pinned to seed 42 alone.

    Returns, per metric in ALL_METRICS: the mean under the metric's own
    unsuffixed column name (so `.loc[size, metric]` reads exactly like the
    old single-row tables), the sample std under f"{metric}_std" (ddof=1,
    filled to 0.0 for a lone seed rather than NaN), and `n_seeds` so callers
    can skip drawing an error bar where there's only one seed to average --
    which is every size/mode that hasn't had its seed-43/44 rerun
    downloaded yet, so those bars render identically to before this change."""
    if df.empty:
        return df
    metric_cols = [c for c in ALL_METRICS if c in df.columns]
    grouped = df.groupby("model_size")
    mean = grouped[metric_cols].mean()
    std = grouped[metric_cols].std(ddof=1).fillna(0.0)
    std.columns = [f"{c}_std" for c in std.columns]
    n_seeds = grouped.size().rename("n_seeds")
    return pd.concat([mean, std, n_seeds], axis=1)


# Shared matplotlib error-bar styling for plot_main_figure_2x1/plot_main_figure_test.
ERROR_BAR_KW = {"ecolor": "black", "elinewidth": 1.0, "capsize": 3, "capthick": 1.0}


def _bar_mean_err(agg: pd.DataFrame, size: str, metric: str):
    """(mean, err) for `size`/`metric` out of an _agg_seeds_by_size table, or
    None if `size` isn't in it. `err` is None (not 0) when only one seed is
    present, so callers can skip passing yerr entirely rather than drawing a
    zero-length error bar."""
    if agg.empty or size not in agg.index:
        return None
    row = agg.loc[size]
    std_col = f"{metric}_std"
    # std_col can be absent if `metric` itself was never a column in the
    # source rows for this group (shouldn't happen on real
    # checkpoint_metrics.csv output -- every row carries every ALL_METRICS
    # column -- but fail soft rather than KeyError on partial/hand-edited data).
    err = row[std_col] if std_col in row.index and row["n_seeds"] > 1 else None
    return row[metric], err


def _test_base_table(base_test: pd.DataFrame) -> pd.DataFrame:
    """One row per model_size out of `base_test`, canonical seed-42 run only
    -- the base-model counterpart of _test_table, used by every test-split
    figure that ISN'T error-bar-aware (plot_main_figure_3x1,
    plot_heatmap_size_x_mode_test). Needed as of 2026-07-24: once the
    eval-time seed-robustness sweep's base-model reruns
    (test-{size}-base-seed{43,44}-seed{43,44}, see EVAL_SEED_TEST_BASE_RE in
    checkpoint_metrics.py) are downloaded, `base_test` legitimately has
    MORE THAN ONE row per model_size -- a plain `base_test.set_index
    ("model_size")` then has a non-unique index and `.loc[size, metric]`
    returns a Series instead of a scalar, breaking every arithmetic/plotting
    call downstream. Only plot_main_figure_2x1/plot_main_figure_test want the
    full seed spread (via _agg_seeds_by_size) -- everything else should keep
    reading the one dev-selected/canonical number, same as before this sweep
    existed."""
    if base_test.empty:
        return base_test
    sub = base_test[base_test["seed"] == 42]
    return sub.set_index("model_size")


def plot_main_figure_test(test: pd.DataFrame, base_test: pd.DataFrame, metrics: list = None,
                           out_dir: Path = None, stem: str = "checkpoint_best_per_size_2x2",
                           include_lora: bool = True):
    """Headline figure for the held-out TEST split: for each model size,
    base vs. lora vs. full bars, same slot layout as plot_main_figure.
    Unlike that dev figure there's no "best epoch" selection for lora/full
    -- each size/mode evaluated exactly ONE checkpoint on test (the epoch
    already selected on dev by ICD F1 Macro), so this just plots whatever's
    in `test` directly. The base bar uses the REAL test-split base-model
    eval (`base_test` -- test-{size}-base collections from
    eval_base_models_config_test.yaml, added 2026-07-21). Falls back to a
    caller-supplied dev-split base table for any size missing a same-split
    number, but as of 2026-07-21 all four sizes have real test-split base
    scores, so that fallback shouldn't currently trigger.

    No epoch labels above the bars (2026-07-21 -- dropped per Jan: the dev-
    selected epoch is documented in checkpoint_best_epochs.csv/the appendix
    instead, cleaner for the print figure).

    `out_dir`/`stem` (added 2026-07-24) let a caller reuse this exact
    base/lora/full grouped-bar layout for a different metric selection
    written somewhere other than main/ -- see plot_v2_recall_at_k
    below, which is the same figure with a diagnostic metric set saved to
    diagnostics/ instead of overwriting the headline checkpoint_best_per_size_2x2.

    `include_lora=False` (added 2026-07-24, same convention as
    plot_main_figure_3x1/plot_main_figure_2x1's no-lora variants) drops the
    LoRA bars entirely and re-centers to NO_LORA_OFFSETS/NO_LORA_WIDTH's
    2-slot (base/full) layout instead of leaving a gap where LoRA used to
    sit. Caller is responsible for picking a stem that reflects this (e.g.
    appending "_no_lora") -- this function doesn't rename the stem itself,
    since not every caller wants that suffix convention.

    Error bars (added 2026-07-24, per Jan's scripts/eval_error_bars_config_
    seed{43,44}[.yaml|_8b.yaml] sweep): every bar is the MEAN across all seed
    replicates available for that (model_size, mode) -- just the canonical
    seed-42 checkpoint where no error-bar rerun has been downloaded yet, or
    seed 42 + 43 + 44 wherever it has (see _agg_seeds_by_size). The error bar
    is the sample std across those seeds and is only drawn where n_seeds > 1,
    so bars for any size/mode still on a single seed look exactly like
    before this change -- no zero-length whiskers."""
    metrics = metrics if metrics is not None else METRICS
    out_dir = out_dir if out_dir is not None else MAIN_DIR
    if test.empty:
        print(f"main (test, {stem}): no test-split rows found -- skipping.")
        return
    sizes = sorted(test["model_size"].dropna().unique(), key=_size_sort_key)
    base_agg = _agg_seeds_by_size(base_test)
    modes = ("lora", "full") if include_lora else ("full",)
    mode_agg = {mode: _agg_seeds_by_size(test[test["mode"] == mode]) for mode in modes}
    offsets = SLOT_OFFSETS if include_lora else NO_LORA_OFFSETS
    width = SLOT_WIDTH if include_lora else NO_LORA_WIDTH

    x = np.arange(len(sizes))
    fig, axes = _subplot_grid(len(metrics), cell_size=(3.5, 2.7))
    for ax, (metric, ylabel) in zip(axes, metrics):
        all_vals = []
        for i, size in enumerate(sizes):
            hit = _bar_mean_err(base_agg, size, metric)
            if hit is not None:
                v, err = hit
                bar_kw = {"color": BASE_COLOR, "zorder": 3}
                if err:
                    bar_kw.update(yerr=err, error_kw=ERROR_BAR_KW)
                ax.bar(x[i] + offsets["base"], v, width, **bar_kw)
                all_vals.append(v + (err or 0))
            for mode, agg in mode_agg.items():
                hit = _bar_mean_err(agg, size, metric)
                if hit is None:
                    continue
                v, err = hit
                bar_kw = {"color": BLOCK_COLORS[mode], "zorder": 3}
                if err:
                    bar_kw.update(yerr=err, error_kw=ERROR_BAR_KW)
                ax.bar(x[i] + offsets[mode], v, width, **bar_kw)
                all_vals.append(v + (err or 0))
        ymax = max(all_vals) if all_vals else 0
        ax.set_ylim(0, ymax * 1.15 if ymax else 1)
        ax.set_xticks(x)
        ax.set_xticklabels(sizes, fontsize=9)
        ax.set_title(metric, fontsize=10)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.tick_params(axis="both", labelsize=8)
        ax.spines[["top", "right"]].set_visible(False)

    handles = [Patch(facecolor=BASE_COLOR, label="base")]
    if include_lora:
        handles.append(Patch(facecolor=LORA_COLOR, label="LoRA"))
    handles.append(Patch(facecolor=FULL_COLOR, label="full FT"))
    fig.legend(handles=handles, fontsize=9, ncol=len(handles), loc="upper center",
               bbox_to_anchor=(0.5, 1.02), frameon=False)
    fig.tight_layout(rect=[0, 0, 1, 0.94])

    out_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(out_dir / f"{stem}.{ext}", dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"main (test, {stem}): saved held-out test-split figure to {out_dir}")


# Diagnostic metric set, rewritten 2026-07-24 (per Jan) to drop MRR/F1 Macro@1
# entirely rather than pair them against Recall@3/@5. Reason: V2's own
# candidate-acceptance gate (src/pipeline/verifier.py's get_mrr_score, used
# via handle_diagnoses -> calculate_scores, gated by
# filter_success_candidates against v_args.threshold) is *effectively*
# Recall@1, not a continuous MRR target -- get_mrr_score only returns
# 1/(rank+1) in {1.0, 0.5, 0.333, ...}, and the V2/V3 thresholds (0.5, 0.8 in
# scripts/pipeline_config.yaml's `thresholds: [0.6, 0.5, 0.8, 0.55]`) both
# only let a rank-1 hit (score 1.0) clear the strict `score > threshold` bar
# in pipeline.py's filter_success_candidates -- a rank-2 hit (score exactly
# 0.5) does NOT pass at threshold 0.5. So "MRR" was never really the
# curation target; it's Recall@1 with MRR only used to break ties among
# rejected (sub-rank-1) candidates once the resampling budget runs out. Given
# that, reporting V2 MRR/F1 Macro@1 (both rank-1-only reads, same as the
# curation gate) next to Recall@3/@5 mixes a name ("MRR") that overstates
# what the pipeline does with metrics that are actually the same
# information as Recall@1. Cleaner: one metric family (Recall@k) throughout
# -- Recall@1 (what curation targets), Recall@3, Recall@5 (whether that
# target's rank-1 misses are still present lower in the list).
V2_RANKING_DIAGNOSTIC_METRICS = [
    ("V2 Recall@1", ALL_METRICS["V2 Recall@1"]),
    ("V2 Recall@3", ALL_METRICS["V2 Recall@3"]),
    ("V2 Recall@5", ALL_METRICS["V2 Recall@5"]),
]


def plot_v2_recall_at_k(test: pd.DataFrame, base_test: pd.DataFrame, include_lora: bool = True):
    """diagnostics/v2_recall_at_k.{png,pdf} (or _no_lora.{png,pdf} with
    include_lora=False) -- same TEST-split base/lora/full grouped-bar figure
    as checkpoint_best_per_size_2x2 (plot_main_figure_test), but with
    V2_RANKING_DIAGNOSTIC_METRICS (Recall@1/3/5) instead of the headline
    ICD-focused metric set. Renamed 2026-07-24 from
    plot_v2_ranking_diagnostic / v2_ranking_recovery_2x2 when the metric set
    dropped MRR/F1 Macro@1 (see V2_RANKING_DIAGNOSTIC_METRICS comment above
    for why) -- 3 panels now, not 4, so the old "_2x2" stem no longer
    describes the layout.

    Why: Recall@1 is what V2's curation threshold actually targets (see
    above), so it's the fairest "does fine-tuning hit the training target"
    read. Recall@3/@5 show that for 32b (full FT + LoRA) and 14b-LoRA, a
    Recall@1 miss is usually still in the list lower down -- those
    checkpoints catch back up to (or pass) 8b by Recall@3, while 14b-full
    and 8b-LoRA stay behind at every k, i.e. a real miss rather than a
    ranking one. One-off diagnostic (not a headline/ablation result), hence
    diagnostics/ rather than main/ -- same convention as
    json_validity_impact.py's figure.

    include_lora=False (added 2026-07-24, per Jan, same convention as
    plot_main_figure_3x1/2x1's no-lora variants) drops the LoRA bars --
    base vs. full FT only -- since LoRA's ranking vs. full FT isn't stable
    across metrics/sizes (see EACL_2026_revision_notes.md), a base/full-only
    read of the Recall@k story is useful on its own."""
    stem = "v2_recall_at_k" if include_lora else "v2_recall_at_k_no_lora"
    plot_main_figure_test(test, base_test, metrics=V2_RANKING_DIAGNOSTIC_METRICS,
                           out_dir=DIAGNOSTICS_DIR, stem=stem, include_lora=include_lora)


# Fixed 3-metric selection for plot_main_figure_3x1 -- always ICD F1 Macro,
# ICD F1 Micro, and Diagnoses Recall@3 (V2 Recall@3 in ALL_METRICS -- the
# diagnosis/differential-diagnosis stage, i.e. the \Vtwo{} stage in the
# paper). Swapped from V2 MRR/"Diagnosis MRR" to V2 Recall@3/"Diagnoses
# Recall@3" on 2026-07-24 per Jan -- MRR only credits the top-ranked
# diagnosis, Recall@3 is the more forgiving "is the right diagnosis anywhere
# in the top 3" read, consistent with why Recall@k got added alongside V2 MRR
# in the first place (see V2_RANKING_DIAGNOSTIC_METRICS above).
MAIN_3X1_METRICS = [
    ("ICD F1 Macro", ALL_METRICS["ICD F1 Macro"]),
    ("ICD F1 Micro", ALL_METRICS["ICD F1 Micro"]),
    ("V2 Recall@3", ALL_METRICS["V2 Recall@3"]),
]
MAIN_3X1_TITLES = {
    "ICD F1 Macro": "ICD F1 Macro",
    "ICD F1 Micro": "ICD F1 Micro",
    "V2 Recall@3": "Diagnoses Recall@3",
}


# 2-slot layout (base + full only) for the no-lora variant of the 3x1 figure
# below -- SLOT_OFFSETS/SLOT_WIDTH assume 3 slots (base/lora/full) and would
# leave an empty gap where the lora bar used to sit if reused as-is.
NO_LORA_OFFSETS = {"base": -0.18, "full": 0.18}
NO_LORA_WIDTH = 0.32

# 2-metric selection for plot_main_figure_2x1 -- ICD F1 Macro/Micro only.
# Added 2026-07-23 per Jan/revision-notes decision to drop V2 (diagnosis
# ranking) from the main figure entirely (not just MRR -- V2 F1@1 and V2
# Accuracy@1 are in the same "drop" bucket per
# EACL_2026_revision_notes.md's "Figure 1 + main table + drop-V2" section,
# so there's no other headline-worthy metric to swap into the freed slot).
# That leaves exactly 2 metrics, hence a single row of 2 panels instead of
# 3x1's fixed 3.
MAIN_2X1_METRICS = [
    ("ICD F1 Macro", ALL_METRICS["ICD F1 Macro"]),
    ("ICD F1 Micro", ALL_METRICS["ICD F1 Micro"]),
]


def plot_main_figure_3x1(test: pd.DataFrame, base_test: pd.DataFrame, include_lora: bool = True):
    """Compact single-row alternative to checkpoint_best_per_size_2x2: same
    TEST-split base/lora/full bars, but only 3 metrics (ICD F1 Macro, ICD F1
    Micro, Diagnoses Recall@3 -- see MAIN_3X1_METRICS) laid out as one row of
    3 panels instead of a 2x2 grid, so it's shorter vertically. Added
    2026-07-21 per Jan as a candidate replacement for the 2x2 version as the
    main results figure (2026-07-21, per Jan). No epoch labels, same as
    checkpoint_best_per_size_2x2.

    include_lora=False (added 2026-07-21, per Jan) drops the LoRA bars
    entirely -- base vs. full FT only, re-centered to a 2-slot layout instead
    of leaving a gap where the LoRA bar used to sit -- saved as
    checkpoint_best_per_size_3x1_no_lora. Motivation: LoRA's ranking vs. full
    FT isn't fully stable across metrics/sizes (see the open question added
    to EACL_2026_revision_notes.md), so a main-results figure arguably reads
    cleaner without it, with the LoRA-vs-full comparison moved to an
    ablation instead -- open question, not yet decided which cut is used in
    the paper.

    Error bars (added 2026-07-24, same seed-averaging as plot_main_figure_2x1/
    plot_main_figure_test -- see _agg_seeds_by_size): each bar is the mean
    across every seed replicate available for that (model_size, mode), with
    a std-based error bar drawn only where more than one seed's data exists."""
    if test.empty:
        print("main (3x1): no test-split rows found -- skipping.")
        return
    sizes = sorted(test["model_size"].dropna().unique(), key=_size_sort_key)
    base_agg = _agg_seeds_by_size(base_test)
    modes = ("lora", "full") if include_lora else ("full",)
    offsets = SLOT_OFFSETS if include_lora else NO_LORA_OFFSETS
    width = SLOT_WIDTH if include_lora else NO_LORA_WIDTH
    mode_agg = {mode: _agg_seeds_by_size(test[test["mode"] == mode]) for mode in modes}

    x = np.arange(len(sizes))
    fig, axes = plt.subplots(1, 3, figsize=(3.5 * 3, 2.7))
    for ax, (metric, ylabel) in zip(axes, MAIN_3X1_METRICS):
        all_vals = []
        for i, size in enumerate(sizes):
            hit = _bar_mean_err(base_agg, size, metric)
            if hit is not None:
                v, err = hit
                bar_kw = {"color": BASE_COLOR, "zorder": 3}
                if err:
                    bar_kw.update(yerr=err, error_kw=ERROR_BAR_KW)
                ax.bar(x[i] + offsets["base"], v, width, **bar_kw)
                all_vals.append(v + (err or 0))
            for mode, agg in mode_agg.items():
                hit = _bar_mean_err(agg, size, metric)
                if hit is None:
                    continue
                v, err = hit
                bar_kw = {"color": BLOCK_COLORS[mode], "zorder": 3}
                if err:
                    bar_kw.update(yerr=err, error_kw=ERROR_BAR_KW)
                ax.bar(x[i] + offsets[mode], v, width, **bar_kw)
                all_vals.append(v + (err or 0))
        ymax = max(all_vals) if all_vals else 0
        ax.set_ylim(0, ymax * 1.15 if ymax else 1)
        ax.set_xticks(x)
        ax.set_xticklabels(sizes, fontsize=9)
        ax.set_title(MAIN_3X1_TITLES[metric], fontsize=10)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.tick_params(axis="both", labelsize=8)
        ax.spines[["top", "right"]].set_visible(False)

    handles = [Patch(facecolor=BASE_COLOR, label="base")]
    if include_lora:
        handles.append(Patch(facecolor=LORA_COLOR, label="LoRA"))
    handles.append(Patch(facecolor=FULL_COLOR, label="full FT"))
    fig.legend(handles=handles, fontsize=9, ncol=len(handles), loc="upper center",
               bbox_to_anchor=(0.5, 1.08), frameon=False)
    fig.tight_layout(rect=[0, 0, 1, 0.88])

    stem = "checkpoint_best_per_size_3x1" if include_lora else "checkpoint_best_per_size_3x1_no_lora"
    MAIN_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(MAIN_DIR / f"{stem}.{ext}", dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    tag = "3x1" if include_lora else "3x1 no-lora"
    print(f"main ({tag}): saved compact 3-panel headline figure to {MAIN_DIR}")


def plot_main_figure_2x1(test: pd.DataFrame, base_test: pd.DataFrame, include_lora: bool = True):
    """TEST-split headline figure, take 2 (2026-07-23): same base/lora/full
    bars as plot_main_figure_3x1, but only the 2 metrics that survive the
    drop-V2 decision (ICD F1 Macro, ICD F1 Micro -- see MAIN_2X1_METRICS and
    EACL_2026_revision_notes.md's "Figure 1 + main table + drop-V2" section).
    No epoch labels, same as _2x2/_3x1.

    Legend moved to the right side of the figure (vs. top-center in _2x2/
    _3x1) -- with only 2 panels a top-center legend eats a disproportionate
    share of vertical space relative to the plot area, and a side legend
    reads more naturally for a wide-but-short 2-panel row. Per Jan,
    2026-07-23.

    include_lora=False drops the LoRA bars (base vs. full FT only, re-
    centered 2-slot layout), same open question as _3x1_no_lora -- both
    cuts saved so it's easy to compare before deciding.

    Error bars (added 2026-07-24, per Jan's scripts/eval_error_bars_config_
    seed{43,44}[.yaml|_8b.yaml] sweep -- this is the figure that sweep's file
    headers name explicitly): same seed-averaging as plot_main_figure_test,
    see that function's docstring / _agg_seeds_by_size for details. Bars for
    a size/mode without a downloaded seed-43/44 rerun are unaffected -- mean
    equals the seed-42 value, no error bar drawn."""
    if test.empty:
        print("main (2x1): no test-split rows found -- skipping.")
        return
    sizes = sorted(test["model_size"].dropna().unique(), key=_size_sort_key)
    base_agg = _agg_seeds_by_size(base_test)
    modes = ("lora", "full") if include_lora else ("full",)
    offsets = SLOT_OFFSETS if include_lora else NO_LORA_OFFSETS
    width = SLOT_WIDTH if include_lora else NO_LORA_WIDTH
    mode_agg = {mode: _agg_seeds_by_size(test[test["mode"] == mode]) for mode in modes}

    x = np.arange(len(sizes))
    fig, axes = plt.subplots(1, 2, figsize=(3.5 * 2 + 1.4, 2.7))
    for ax, (metric, ylabel) in zip(axes, MAIN_2X1_METRICS):
        all_vals = []
        for i, size in enumerate(sizes):
            hit = _bar_mean_err(base_agg, size, metric)
            if hit is not None:
                v, err = hit
                bar_kw = {"color": BASE_COLOR, "zorder": 3}
                if err:
                    bar_kw.update(yerr=err, error_kw=ERROR_BAR_KW)
                ax.bar(x[i] + offsets["base"], v, width, **bar_kw)
                all_vals.append(v + (err or 0))
            for mode, agg in mode_agg.items():
                hit = _bar_mean_err(agg, size, metric)
                if hit is None:
                    continue
                v, err = hit
                bar_kw = {"color": BLOCK_COLORS[mode], "zorder": 3}
                if err:
                    bar_kw.update(yerr=err, error_kw=ERROR_BAR_KW)
                ax.bar(x[i] + offsets[mode], v, width, **bar_kw)
                all_vals.append(v + (err or 0))
        ymax = max(all_vals) if all_vals else 0
        ax.set_ylim(0, ymax * 1.15 if ymax else 1)
        ax.set_xticks(x)
        ax.set_xticklabels(sizes, fontsize=9)
        ax.set_title(metric, fontsize=10)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.tick_params(axis="both", labelsize=8)
        ax.spines[["top", "right"]].set_visible(False)

    handles = [Patch(facecolor=BASE_COLOR, label="base")]
    if include_lora:
        handles.append(Patch(facecolor=LORA_COLOR, label="LoRA"))
    handles.append(Patch(facecolor=FULL_COLOR, label="full FT"))
    fig.legend(handles=handles, fontsize=9, loc="center left",
               bbox_to_anchor=(0.85, 0.5), frameon=False)
    fig.tight_layout(rect=[0, 0, 0.86, 1])

    stem = "checkpoint_best_per_size_2x1" if include_lora else "checkpoint_best_per_size_2x1_no_lora"
    MAIN_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(MAIN_DIR / f"{stem}.{ext}", dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    tag = "2x1" if include_lora else "2x1 no-lora"
    print(f"main ({tag}): saved compact 2-panel headline figure (V2 MRR dropped) to {MAIN_DIR}")


def plot_dataset_analyses(ckpt: pd.DataFrame, base: pd.DataFrame, primary_metric: str, metrics: list = None):
    """Dataset-ablation figures, restricted to DATASET_ANALYSES_SIZES (just
    8b): one bar per dataset trained in that exact size+mode, 2x2 metric
    grid, same-size base line for context. 8b is the only size with more
    than 1-2 datasets per mode, so it's the only place a dataset comparison
    is actually meaningful."""
    metrics = metrics if metrics is not None else METRICS
    DATASET_ANALYSES_DIR.mkdir(parents=True, exist_ok=True)
    sizes = [s for s in sorted(ckpt["model_size"].dropna().unique(), key=_size_sort_key)
             if s in DATASET_ANALYSES_SIZES]
    base_by_size = base.set_index("model_size") if not base.empty else base

    for size in sizes:
        for mode in ("full", "lora"):
            sub = ckpt[(ckpt["model_size"] == size) & (ckpt["mode"] == mode)]
            if sub.empty:
                continue
            datasets = [d for d in COLORS if (sub["dataset"] == d).any()]
            chosen = (sub.loc[sub.groupby("dataset")[primary_metric].idxmax()]
                      .set_index("dataset").loc[datasets].reset_index())
            base_row = base_by_size.loc[size] if size in base_by_size.index else None

            fig, axes = _subplot_grid(len(metrics), cell_size=(3.0, 2.3))
            for ax, (metric, ylabel) in zip(axes, metrics):
                bars = ax.bar(chosen["dataset"], chosen[metric],
                               color=[COLORS[d] for d in chosen["dataset"]])
                ymax = max(chosen[metric].max(), base_row[metric] if base_row is not None else 0)
                ax.set_ylim(0, ymax * 1.25 if ymax else 1)
                for bar, ep in zip(bars, chosen["epoch"]):
                    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + ymax * 0.03,
                            f"e{int(ep)}", ha="center", va="bottom", fontsize=7)
                if base_row is not None:
                    ax.axhline(base_row[metric], color=BASE_COLOR, linestyle="--", linewidth=1.2,
                               label=f"{size} base")
                ax.set_title(metric, fontsize=10)
                ax.set_ylabel(ylabel, fontsize=9)
                ax.tick_params(axis="x", labelsize=8, rotation=15)
                ax.tick_params(axis="y", labelsize=8)
                ax.spines[["top", "right"]].set_visible(False)
            if base_row is not None:
                axes[0].legend(fontsize=7)
            fig.tight_layout()

            stem = f"bars_{size}_{mode}"
            for ext in ("png", "pdf"):
                fig.savefig(DATASET_ANALYSES_DIR / f"{stem}.{ext}", dpi=DPI, bbox_inches="tight")
            plt.close(fig)
            print(f"dataset-analyses: saved {stem} for datasets {datasets}")


def plot_epoch_analyses(ckpt: pd.DataFrame, base: pd.DataFrame, primary_metric: str, metrics: list = None):
    """8 figures (4 sizes x {full, lora}): bars over [base, e1, e2, ..., eN]
    for that size/mode's best dataset (same dataset plot_heatmap_size_x_epoch
    picks by primary_metric). This is the epoch-progression story as a bar
    chart instead of a connected line, which would misleadingly imply a
    continuous trend between only 3-4 discrete checkpoints."""
    metrics = metrics if metrics is not None else METRICS
    EPOCH_ANALYSES_DIR.mkdir(parents=True, exist_ok=True)
    sizes = sorted(ckpt["model_size"].dropna().unique(), key=_size_sort_key)
    base_by_size = base.set_index("model_size") if not base.empty else base
    full_by_size = _best_by_size_for_mode(ckpt, "full", primary_metric)
    lora_by_size = _best_by_size_for_mode(ckpt, "lora", primary_metric)

    for size in sizes:
        for mode, tbl in (("lora", lora_by_size), ("full", full_by_size)):
            if size not in tbl.index:
                continue
            dataset = tbl.loc[size, "dataset"]
            sub = ckpt[(ckpt["model_size"] == size) & (ckpt["mode"] == mode) &
                       (ckpt["dataset"] == dataset)].sort_values("epoch")
            base_row = base_by_size.loc[size] if size in base_by_size.index else None
            labels = (["base"] if base_row is not None else []) + [f"e{int(e)}" for e in sub["epoch"]]
            mode_color = BLOCK_COLORS[mode]

            fig, axes = _subplot_grid(len(metrics), cell_size=(3.0, 2.3))
            for ax, (metric, ylabel) in zip(axes, metrics):
                vals = ([base_row[metric]] if base_row is not None else []) + sub[metric].tolist()
                colors = ([BASE_COLOR] if base_row is not None else []) + [mode_color] * len(sub)
                bars = ax.bar(labels, vals, color=colors)
                ymax = max(vals) if vals else 0
                ax.set_ylim(0, ymax * 1.22 if ymax else 1)
                for bar, v in zip(bars, vals):
                    ax.text(bar.get_x() + bar.get_width() / 2, v + ymax * 0.03,
                            f"{v:.1f}", ha="center", va="bottom", fontsize=7)
                ax.set_title(metric, fontsize=10)
                ax.set_ylabel(ylabel, fontsize=9)
                ax.tick_params(axis="both", labelsize=8)
                ax.spines[["top", "right"]].set_visible(False)
            fig.suptitle(f"{size} {mode} ({dataset})", fontsize=9, y=1.0)
            fig.tight_layout(rect=[0, 0, 1, 0.96])

            stem = f"epoch_bars_{size}_{mode}"
            for ext in ("png", "pdf"):
                fig.savefig(EPOCH_ANALYSES_DIR / f"{stem}.{ext}", dpi=DPI, bbox_inches="tight")
            plt.close(fig)
            print(f"epoch-analyses: saved {stem} ({dataset}, {labels})")


def _annotate_heatmap(ax, grid: np.ndarray, fmt: str = "{:.1f}"):
    """Text color needs to track where a cell falls in THIS subplot's own
    color scale, not an absolute fraction of the max -- imshow normalizes
    each subplot to its own [min, max], so a cell holding the subplot's
    minimum value always renders as the lightest color even if that value is
    numerically large (e.g. DotProduct cells cluster ~50-62), and needs dark
    text regardless of its absolute size."""
    finite = grid[~np.isnan(grid)]
    if not finite.size:
        return
    vmin, vmax = finite.min(), finite.max()
    span = vmax - vmin
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            v = grid[i, j]
            if np.isnan(v):
                continue
            frac = (v - vmin) / span if span > 0 else 0.5
            color = "white" if frac > 0.6 else "black"
            ax.text(j, i, fmt.format(v), ha="center", va="center", fontsize=7.5, color=color)


def plot_heatmap_size_x_mode(ckpt: pd.DataFrame, base: pd.DataFrame, primary_metric: str, metrics: list = None):
    """Rows=size, cols=[base, lora, full], cell=that combo's best-epoch
    score -- the same numbers as the main figure, in compact grid form."""
    metrics = metrics if metrics is not None else METRICS
    sizes = sorted(ckpt["model_size"].dropna().unique(), key=_size_sort_key)
    modes = ["base", "lora", "full"]
    base_by_size = base.set_index("model_size") if not base.empty else base
    tbl_by_mode = {
        "base": base_by_size,
        "lora": _best_by_size_for_mode(ckpt, "lora", primary_metric),
        "full": _best_by_size_for_mode(ckpt, "full", primary_metric),
    }

    HEATMAP_DIR.mkdir(parents=True, exist_ok=True)
    fig, axes = _subplot_grid(len(metrics), cell_size=(3.75, 3.0))
    for ax, (metric, ylabel) in zip(axes, metrics):
        grid = np.full((len(sizes), len(modes)), np.nan)
        for i, size in enumerate(sizes):
            for j, mode in enumerate(modes):
                tbl = tbl_by_mode[mode]
                if size in tbl.index:
                    grid[i, j] = tbl.loc[size, metric]
        im = ax.imshow(grid, cmap="YlOrRd", aspect="auto")
        ax.set_xticks(range(len(modes)))
        ax.set_xticklabels(modes, fontsize=9)
        ax.set_yticks(range(len(sizes)))
        ax.set_yticklabels(sizes, fontsize=9)
        _annotate_heatmap(ax, grid)
        ax.set_title(metric, fontsize=10)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(HEATMAP_DIR / f"heatmap_size_x_mode.{ext}", dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"heatmap: saved size x mode ({len(sizes)} sizes x {modes}) to {HEATMAP_DIR}")


def plot_heatmap_size_x_mode_test(test: pd.DataFrame, base_test: pd.DataFrame, metrics: list = None):
    """Test-split counterpart of plot_heatmap_size_x_mode: rows=size,
    cols=[base, lora, full], cell=that size/mode's single dev-selected
    checkpoint's score on the held-out TEST split. The base column uses the
    REAL test-split base-model eval (`base_test`, test-{size}-base
    collections, added 2026-07-21) -- same-split numbers throughout, not a
    dev-reused stand-in."""
    metrics = metrics if metrics is not None else METRICS
    if test.empty:
        print("heatmap (test): no test-split rows found -- skipping size x mode heatmap.")
        return
    sizes = sorted(test["model_size"].dropna().unique(), key=_size_sort_key)
    modes = ["base", "lora", "full"]
    base_by_size = _test_base_table(base_test)
    tbl_by_mode = {
        "base": base_by_size,
        "lora": _test_table(test, "lora"),
        "full": _test_table(test, "full"),
    }

    HEATMAP_DIR.mkdir(parents=True, exist_ok=True)
    fig, axes = _subplot_grid(len(metrics), cell_size=(3.75, 3.0))
    for ax, (metric, ylabel) in zip(axes, metrics):
        grid = np.full((len(sizes), len(modes)), np.nan)
        for i, size in enumerate(sizes):
            for j, mode in enumerate(modes):
                tbl = tbl_by_mode[mode]
                if size in tbl.index:
                    grid[i, j] = tbl.loc[size, metric]
        im = ax.imshow(grid, cmap="YlOrRd", aspect="auto")
        ax.set_xticks(range(len(modes)))
        ax.set_xticklabels(modes, fontsize=9)
        ax.set_yticks(range(len(sizes)))
        ax.set_yticklabels(sizes, fontsize=9)
        _annotate_heatmap(ax, grid)
        ax.set_title(metric, fontsize=10)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(HEATMAP_DIR / f"heatmap_size_x_mode_test.{ext}", dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"heatmap (test): saved size x mode ({len(sizes)} sizes x {modes}) to {HEATMAP_DIR}")


def plot_heatmap_size_x_epoch(ckpt: pd.DataFrame, base: pd.DataFrame, primary_metric: str,
                               metrics: list = None):
    """Rows=size+mode (up to 8), cols=[base, epoch 1..N], cell=score at that
    epoch for that size/mode's best dataset (same dataset plot_all_sizes/
    plot_per_model_bars pick by primary_metric). The base column is that
    row's size's untuned score, giving every row a "before" reference at a
    glance instead of only showing the fine-tuning trajectory in isolation.
    Shows the epoch-progression story as discrete cells instead of a line
    implying a continuous trend between 3-4 points."""
    metrics = metrics if metrics is not None else METRICS
    sizes = sorted(ckpt["model_size"].dropna().unique(), key=_size_sort_key)
    base_by_size = base.set_index("model_size") if not base.empty else base
    full_by_size = _best_by_size_for_mode(ckpt, "full", primary_metric)
    lora_by_size = _best_by_size_for_mode(ckpt, "lora", primary_metric)

    rows = []  # (row_label, size, mode, dataset)
    for size in sizes:
        for mode, tbl in (("lora", lora_by_size), ("full", full_by_size)):
            if size not in tbl.index:
                continue
            dataset = tbl.loc[size, "dataset"]
            rows.append((f"{size} ({mode})", size, mode, dataset))

    if not rows:
        print("heatmap: no full/lora checkpoint rows found -- skipping size x epoch heatmap.")
        return

    max_epoch = int(ckpt["epoch"].max())
    epochs = list(range(1, max_epoch + 1))
    col_labels = ["base"] + [f"e{e}" for e in epochs]

    HEATMAP_DIR.mkdir(parents=True, exist_ok=True)
    fig, axes = _subplot_grid(len(metrics), cell_size=(3.75, 3.75))
    for ax, (metric, ylabel) in zip(axes, metrics):
        grid = np.full((len(rows), len(col_labels)), np.nan)
        for i, (label, size, mode, dataset) in enumerate(rows):
            if size in base_by_size.index:
                grid[i, 0] = base_by_size.loc[size, metric]
            sub = ckpt[(ckpt["model_size"] == size) & (ckpt["mode"] == mode) &
                       (ckpt["dataset"] == dataset)]
            for _, r in sub.iterrows():
                ep = int(r["epoch"])
                if ep in epochs:
                    grid[i, 1 + epochs.index(ep)] = r[metric]
        im = ax.imshow(grid, cmap="YlOrRd", aspect="auto")
        ax.set_xticks(range(len(col_labels)))
        ax.set_xticklabels(col_labels, fontsize=9)
        ax.set_yticks(range(len(rows)))
        ax.set_yticklabels([r[0] for r in rows], fontsize=8)
        _annotate_heatmap(ax, grid)
        ax.set_title(metric, fontsize=10)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(HEATMAP_DIR / f"heatmap_size_x_epoch.{ext}", dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"heatmap: saved size x epoch ({len(rows)} rows: {[r[0] for r in rows]}) to {HEATMAP_DIR}")


def plot_heatmap_dataset_x_epoch(ckpt: pd.DataFrame, base: pd.DataFrame, primary_metric: str,
                                  metrics: list = None):
    """One heatmap per (size, mode) that trained more than one dataset --
    only there is a dataset-level epoch comparison actually meaningful.
    Rows=dataset, cols=[base, epoch 1..N], cell=score at that epoch. The base
    column repeats that size's untuned score on every row (base doesn't vary
    by dataset), so each dataset's trajectory still has a "before" reference
    alongside it. Complements dataset_analyses (which collapses each dataset
    to its single best epoch) by keeping the full epoch trajectory per
    dataset, still as discrete heatmap cells rather than a connected line."""
    metrics = metrics if metrics is not None else METRICS
    HEATMAP_DIR.mkdir(parents=True, exist_ok=True)
    sizes = sorted(ckpt["model_size"].dropna().unique(), key=_size_sort_key)
    base_by_size = base.set_index("model_size") if not base.empty else base

    for size in sizes:
        for mode in ("lora", "full"):
            sub_all = ckpt[(ckpt["model_size"] == size) & (ckpt["mode"] == mode)]
            datasets = [d for d in COLORS if (sub_all["dataset"] == d).any()]
            if len(datasets) < 2:
                continue  # nothing to compare -- only one dataset trained here

            max_epoch = int(sub_all["epoch"].max())
            epochs = list(range(1, max_epoch + 1))
            col_labels = ["base"] + [f"e{e}" for e in epochs]

            fig, axes = _subplot_grid(len(metrics), cell_size=(3.5, 2.75))
            for ax, (metric, ylabel) in zip(axes, metrics):
                grid = np.full((len(datasets), len(col_labels)), np.nan)
                for i, dataset in enumerate(datasets):
                    if size in base_by_size.index:
                        grid[i, 0] = base_by_size.loc[size, metric]
                    sub = sub_all[sub_all["dataset"] == dataset]
                    for _, r in sub.iterrows():
                        ep = int(r["epoch"])
                        if ep in epochs:
                            grid[i, 1 + epochs.index(ep)] = r[metric]
                im = ax.imshow(grid, cmap="YlOrRd", aspect="auto")
                ax.set_xticks(range(len(col_labels)))
                ax.set_xticklabels(col_labels, fontsize=9)
                ax.set_yticks(range(len(datasets)))
                ax.set_yticklabels(datasets, fontsize=9)
                _annotate_heatmap(ax, grid)
                ax.set_title(metric, fontsize=10)
                fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            fig.tight_layout()

            stem = f"heatmap_dataset_x_epoch_{size}_{mode}"
            for ext in ("png", "pdf"):
                fig.savefig(HEATMAP_DIR / f"{stem}.{ext}", dpi=DPI, bbox_inches="tight")
            plt.close(fig)
            print(f"heatmap: saved dataset x epoch for {size} {mode} ({datasets}) to {HEATMAP_DIR}")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--primary-metric", choices=METRIC_NAMES, default=DEFAULT_PRIMARY_METRIC,
        help=f"Metric used to pick the one checkpoint per group in every figure "
             f"(default: {DEFAULT_PRIMARY_METRIC!r}).",
    )
    parser.add_argument(
        "--metrics", nargs="+", choices=METRIC_NAMES, default=None, metavar="METRIC",
        help=f"Which metrics to plot, and in what order, across every figure "
             f"(space-separated). Default: {DEFAULT_METRIC_SELECTION}. Choices: {METRIC_NAMES}.",
    )
    return parser.parse_args()


def run(primary_metric: str = DEFAULT_PRIMARY_METRIC, metric_names: list = None,
        out_root=None, in_csv=None):
    """Programmatic entry point (used by scripts/eval_analysis.py).
    `out_root`/`in_csv` override the module-level defaults for this run --
    the plot functions read the module globals, so we rebind them (and the
    subdirectories derived from OUT_ROOT) here."""
    global OUT_ROOT, MAIN_DIR, DATASET_ANALYSES_DIR, EPOCH_ANALYSES_DIR, HEATMAP_DIR, DIAGNOSTICS_DIR, IN_CSV
    if out_root is not None:
        OUT_ROOT = Path(out_root)
        MAIN_DIR = OUT_ROOT / "main"
        DATASET_ANALYSES_DIR = OUT_ROOT / "dataset_analyses"
        EPOCH_ANALYSES_DIR = OUT_ROOT / "epoch_analyses"
        HEATMAP_DIR = OUT_ROOT / "heatmaps"
        DIAGNOSTICS_DIR = OUT_ROOT / "diagnostics"
    if in_csv is not None:
        IN_CSV = Path(in_csv)
    metrics = build_metrics(metric_names)

    if not IN_CSV.exists():
        print(f"{IN_CSV} not found -- run `scripts/eval_analysis.py new_evals` first.")
        return
    df = pd.read_csv(IN_CSV)

    # Backfill `seed` for checkpoint_metrics.csv files written before the
    # eval-time seed-robustness sweep added the column (2026-07-24) -- every
    # row in an old CSV is, by construction, the canonical seed-42 run, same
    # default checkpoint_metrics.py itself now assigns. Without this, _test_table
    # /_agg_seeds_by_size (which key off `seed`) would KeyError on a
    # not-yet-regenerated CSV instead of just behaving as if no seed-43/44
    # replicates exist yet.
    if "seed" not in df.columns:
        df["seed"] = 42

    # is_test rows (scripts/eval_checkpoints_config_test.yaml's held-out
    # TEST-split sweep -- one dev-selected checkpoint per size/mode,
    # evaluated on test exactly once) share the same (model_size, dataset,
    # mode) keys as the dev sweep, so they're excluded from `ckpt` here --
    # otherwise a test-split score could win the "best epoch" selection in
    # every dev figure above and silently mix split results together. They
    # get their own figures below instead.
    #
    # is_base rows now come in two flavors (checkpoint_metrics.py's
    # TEST_BASE_RE, added 2026-07-21): dev-split base (is_test=False, from
    # BASE_RE / eval_config.yaml's `models:` list) and REAL test-split base
    # (is_test=True, test-{size}-base from eval_base_models_config_test.yaml).
    # Split them so `base` (dev figures above) and `base_test` (test figures
    # below) each only ever see same-split numbers -- naively filtering on
    # is_base alone would put both flavors in one table and give
    # `.loc[size]` two rows for the same size.
    ckpt = df[~df["is_base"] & ~df["is_test"]].copy()
    base = df[df["is_base"] & ~df["is_test"]].copy()
    base_test = df[df["is_base"] & df["is_test"]].copy()
    test = df[~df["is_base"] & df["is_test"]].copy()
    if ckpt.empty:
        print("No fine-tuned checkpoint rows found -- nothing to plot.")
        return

    plot_main_figure(ckpt, base, primary_metric, metrics)
    plot_dataset_analyses(ckpt, base, primary_metric, metrics)
    plot_epoch_analyses(ckpt, base, primary_metric, metrics)
    plot_heatmap_size_x_mode(ckpt, base, primary_metric, metrics)
    plot_heatmap_size_x_epoch(ckpt, base, primary_metric, metrics)
    plot_heatmap_dataset_x_epoch(ckpt, base, primary_metric, metrics)
    plot_main_figure_test(test, base_test, metrics)
    plot_heatmap_size_x_mode_test(test, base_test, metrics)
    plot_main_figure_3x1(test, base_test)
    plot_main_figure_3x1(test, base_test, include_lora=False)
    plot_main_figure_2x1(test, base_test)
    plot_main_figure_2x1(test, base_test, include_lora=False)
    plot_v2_recall_at_k(test, base_test)
    plot_v2_recall_at_k(test, base_test, include_lora=False)

    print(f"saved all paper figures (primary metric: {primary_metric}, "
          f"metrics: {[m for m, _ in metrics]}) to {OUT_ROOT}")


def main():
    args = parse_args()
    run(args.primary_metric, args.metrics)


if __name__ == "__main__":
    main()
