#!/usr/bin/env python3
"""Shared plotting primitives: the metric catalogue, the base/LoRA/full-FT
palette, and the seed-aggregation helpers every figure module builds on.

One place to change a metric label or a bar colour, so all figures stay
mutually consistent.
"""
from __future__ import annotations

import pandas as pd

# ---------------------------------------------------------------------------
# Metric catalogue
# ---------------------------------------------------------------------------
# Every metric column checkpoint_metrics.py carries over from
# evaluate_experiment() that is worth plotting (name -> axis label).
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
METRIC_NAMES = list(ALL_METRICS)

DEFAULT_METRIC_SELECTION = [
    "ICD F1 Macro", "ICD F1 Micro", "V2 F1 Macro@1", "V2 F1 Micro@1",
]
DEFAULT_PRIMARY_METRIC = "ICD F1 Macro"


def build_metrics(names: list | None = None) -> list:
    """Turn a list of metric names into the (name, ylabel) pairs the plot
    functions take, validating against ALL_METRICS. `names=None` uses
    DEFAULT_METRIC_SELECTION."""
    names = names if names is not None else DEFAULT_METRIC_SELECTION
    unknown = [n for n in names if n not in ALL_METRICS]
    if unknown:
        raise ValueError(f"Unknown metric(s) {unknown} -- choices are {METRIC_NAMES}")
    return [(n, ALL_METRICS[n]) for n in names]


METRICS = build_metrics()


# ---------------------------------------------------------------------------
# Palette and bar geometry
# ---------------------------------------------------------------------------
BASE_COLOR = "#4D4D4D"
LORA_COLOR = "#4C72B0"
FULL_COLOR = "#DD8452"
BLOCK_COLORS = {"base": BASE_COLOR, "lora": LORA_COLOR, "full": FULL_COLOR}
SLOT_OFFSETS = {"base": -0.28, "lora": 0.0, "full": 0.28}
SLOT_WIDTH = 0.26

# Shared matplotlib error-bar styling.
ERROR_BAR_KW = {"ecolor": "black", "elinewidth": 1.0, "capsize": 3, "capthick": 1.0}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _size_sort_key(size: str) -> float:
    """'0.6b' -> 0.6, '8b' -> 8.0, '14b' -> 14.0, '32b' -> 32.0, so sizes sort
    numerically instead of as strings (where '14b' < '32b' < '8b')."""
    return float(size.rstrip("b"))


def _agg_seeds_by_size(df: pd.DataFrame) -> pd.DataFrame:
    """Collapse `df` (already filtered to one mode, or to the base-model rows,
    all TEST split) to one row per model_size by averaging over every seed
    replicate present: the canonical seed-42 checkpoint plus any seed-repeat
    eval reruns (see EVAL_SEED_TEST_FAMILY_RE / EVAL_SEED_TEST_BASE_RE in
    checkpoint_metrics.py).

    Returns, per metric in ALL_METRICS: the mean under the metric's own
    unsuffixed column name, the sample std under f"{metric}_std" (ddof=1,
    filled to 0.0 for a lone seed rather than NaN), and `n_seeds` so callers
    can skip drawing an error bar where there is only one seed to average.
    """
    if df.empty:
        return df
    metric_cols = [c for c in ALL_METRICS if c in df.columns]
    grouped = df.groupby("model_size")
    mean = grouped[metric_cols].mean()
    std = grouped[metric_cols].std(ddof=1).fillna(0.0)
    std.columns = [f"{c}_std" for c in std.columns]
    n_seeds = grouped.size().rename("n_seeds")
    return pd.concat([mean, std, n_seeds], axis=1)


def _bar_mean_err(agg: pd.DataFrame, size: str, metric: str):
    """(mean, err) for `size`/`metric` out of an _agg_seeds_by_size table, or
    None if `size` is not in it. `err` is None (not 0) when only one seed is
    present, so callers can skip passing yerr entirely rather than drawing a
    zero-length error bar."""
    if agg.empty or size not in agg.index:
        return None
    row = agg.loc[size]
    std_col = f"{metric}_std"
    # std_col can be absent if `metric` was never a column in the source rows
    # for this group; fail soft rather than KeyError on partial data.
    err = row[std_col] if std_col in row.index and row["n_seeds"] > 1 else None
    return row[metric], err
