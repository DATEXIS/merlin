#!/usr/bin/env python3
"""Diagnostic: how much of the TEST-split score is lost to invalid JSON.

We flagged two odd main results:
  1. 8B LoRA (test-8b-r128-e3) scores badly -- worse than even the untuned
     8B base model on ICD F1.
  2. V2 (differential diagnosis) MRR degrades on the bigger full-FT models
     (32b/14b) relative to smaller ones (8b/0.6b).

checkpoint_metrics.csv already carries "V1/V2/V4 JSON Valid Rate" columns
(calculate_json_validity_metrics in src/eval/eval_experiments.py), and both
anomalies co-occur with low validity rates on their respective JSON field:
8b-lora has V4 JSON Valid Rate 69.78%, and 14b-full has V2 JSON Valid Rate
84.29%. But a low valid-rate alone doesn't prove causation: the *current*
metrics already count every invalid-JSON row as an empty prediction (0
score), which mechanically drags the aggregate metric down regardless of
whether the model is otherwise "good." This script isolates that effect by
recomputing each metric a second way -- restricted to only the rows that
parsed -- so the two numbers can be compared directly:

  "current"    : today's metric, exactly as checkpoint_metrics.csv has it
                 (invalid JSON -> [] prediction -> coun면ed as a miss).
  "valid-only" : same metric, recomputed with invalid-JSON rows dropped
                 entirely from the denominator, i.e. "how good is the model
                 when it actually produces parseable output."

If valid-only ~= current, the invalid JSON isn't the story -- the model is
just bad even when it parses. If valid-only is much higher, invalid JSON is
doing most of the damage.

Recomputes from the raw per-item .pq files under
data/results/evaluation/1_4/ rather than re-deriving from checkpoint_metrics.csv,
since the aggregate CSV has no row-level JSON-validity info to filter by.
Reimplements (rather than imports) the small pure-Python pieces of
src/eval/classification_metrics.py + src/eval/eval_experiments.py needed here
(ICD F1 micro/macro, MRR, safe_parse_list, short-code conversion) to avoid
pulling in torch/sentence_transformers, which this recompute doesn't need
(no V1 embedding metrics here) and may not be installed in every environment
this is run from.

Usage (from repo root):
    python -m src.eval.json_validity_impact
or, since this has no other `from src...` imports:
    python src/eval/json_validity_impact.py

Writes:
    data/results/evaluation/json_validity_impact.csv
    figures/diagnostics/json_validity_impact_3x1.{png,pdf}
"""
import ast
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

REPO = Path(__file__).resolve().parents[2] if "__file__" in dir() else Path.cwd()
EVAL_DIR = REPO / "data" / "results" / "evaluation" / "1_4"
OUT_CSV = REPO / "data" / "results" / "evaluation" / "json_validity_impact.csv"
OUT_DIR = REPO / "figures" / "diagnostics"

# (size, mode) -> collection name, same TEST-split grid as
# checkpoint_metrics.py's TEST_FAMILY_INFO / TEST_BASE_RE, hardcoded here
# (same convention as src/eval/robustness_plots.py's ROWS dict) since this
# is a one-off diagnostic, not a new pipeline stage.
TEST_COLLECTIONS = {
    ("0.6b", "base"): "test-06b-base",
    ("0.6b", "full"): "test-06b-full-e4",
    ("0.6b", "lora"): "test-06b-r128-e4",
    ("8b",   "base"): "test-8b-base",
    ("8b",   "full"): "test-8b-full-e3",
    ("8b",   "lora"): "test-8b-r128-e3",
    ("14b",  "base"): "test-14b-base",
    ("14b",  "full"): "test-14b-full-e3",
    ("14b",  "lora"): "test-14b-r128-e3",
    ("32b",  "base"): "test-32b-base",
    ("32b",  "full"): "test-32b-full-e3",
    ("32b",  "lora"): "test-32b-r256-e4",
}
SIZES = ["0.6b", "8b", "14b", "32b"]
MODES = ["base", "lora", "full"]

ROUND_DIGITS = 4


# ---- reimplemented pure-Python pieces (see module docstring for why) ----

def safe_parse_list(value) -> list:
    """Verbatim copy of src/eval/eval_experiments.py's safe_parse_list."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return []
    if isinstance(value, str):
        stripped = value.strip()
        if stripped in ('', 'None', 'nan', 'NaN', 'null'):
            return []
        try:
            parsed = ast.literal_eval(stripped)
        except (ValueError, SyntaxError):
            return []
        return parsed if isinstance(parsed, list) else []
    if isinstance(value, (list, tuple, np.ndarray)):
        return list(value)
    return []


def is_invalid_json(x) -> bool:
    """Same check calculate_json_validity_metrics uses."""
    return pd.isna(x) or x == "None"


def convert_code_to_short_code(code: str, icd_version: int = 10) -> str:
    """Verbatim copy of src/utils.py's convert_code_to_short_code."""
    return code[:4] if icd_version == 9 and code.lower().startswith('e') else code[:3]


def convert_codes_to_short_codes(codes, icd_version: int = 10) -> list:
    return list(set(convert_code_to_short_code(c, icd_version) for c in codes))


def calculate_icd_metrics(y_pred, y_true) -> dict:
    """Verbatim copy of src/eval/classification_metrics.py's
    calculate_icd_metrics_cpu (pure Python, no torch -- copied here only to
    avoid importing that module, which pulls in torch at the top for an
    unrelated function)."""
    tp, fp, fn = defaultdict(int), defaultdict(int), defaultdict(int)
    for pred, true in zip(y_pred, y_true):
        pred_set, true_set = set(pred), set(true)
        for code in pred_set & true_set:
            tp[code] += 1
        for code in pred_set - true_set:
            fp[code] += 1
        for code in true_set - pred_set:
            fn[code] += 1
    labels = set(tp) | set(fp) | set(fn)

    TP, FP, FN = sum(tp.values()), sum(fp.values()), sum(fn.values())
    precision_micro = TP / (TP + FP) if TP + FP else 0
    recall_micro = TP / (TP + FN) if TP + FN else 0
    f1_micro = (2 * precision_micro * recall_micro / (precision_micro + recall_micro)
                if precision_micro + recall_micro else 0)

    f1s = []
    for label in labels:
        p = tp[label] / (tp[label] + fp[label]) if tp[label] + fp[label] else 0
        r = tp[label] / (tp[label] + fn[label]) if tp[label] + fn[label] else 0
        f1s.append(2 * p * r / (p + r) if p + r else 0)
    f1_macro = sum(f1s) / len(f1s) if f1s else 0.0

    return {"ICD F1 Micro": round(f1_micro, ROUND_DIGITS),
            "ICD F1 Macro": round(f1_macro, ROUND_DIGITS)}


def calculate_mrr(y_pred, y_true) -> float:
    """Verbatim copy of src/eval/classification_metrics.py's calculate_mrr."""
    rr_sum = 0
    for pred_list, true_label in zip(y_pred, y_true):
        if true_label in pred_list:
            rr_sum += 1.0 / (pred_list.index(true_label) + 1)
    return rr_sum / len(y_true) if y_true else 0.0


# ---- per-collection recompute ----

def load_pq(collection: str) -> pd.DataFrame:
    path = EVAL_DIR / f"eval_results_{collection}" / f"eval_results_{collection}.pq"
    return pd.read_parquet(path)


def recompute_row(size: str, mode: str, collection: str) -> dict:
    df = load_pq(collection)

    v4_pred_all = df["v4_preds"].apply(safe_parse_list)
    v4_label = df["ICD_CODES"].apply(lambda arr: convert_codes_to_short_codes(list(arr)))
    v2_pred_all = df["v2_preds"].apply(safe_parse_list)
    v2_label = df["disease"].tolist()

    v4_valid = ~df["v4_json"].apply(is_invalid_json)
    v2_valid = ~df["v2_json"].apply(is_invalid_json)

    icd_current = calculate_icd_metrics(v4_pred_all.tolist(), v4_label.tolist())
    icd_valid_only = calculate_icd_metrics(
        v4_pred_all[v4_valid].tolist(), v4_label[v4_valid].tolist())

    mrr_current = calculate_mrr(v2_pred_all.tolist(), v2_label)
    v2_label_valid = df.loc[v2_valid, "disease"].tolist()
    mrr_valid_only = calculate_mrr(v2_pred_all[v2_valid].tolist(), v2_label_valid)

    return {
        "size": size, "mode": mode, "collection": collection, "n_samples": len(df),
        "ICD F1 Macro (current)": round(icd_current["ICD F1 Macro"] * 100, 2),
        "ICD F1 Macro (valid-json-only)": round(icd_valid_only["ICD F1 Macro"] * 100, 2),
        "ICD F1 Micro (current)": round(icd_current["ICD F1 Micro"] * 100, 2),
        "ICD F1 Micro (valid-json-only)": round(icd_valid_only["ICD F1 Micro"] * 100, 2),
        "V2 MRR (current)": round(mrr_current * 100, 2),
        "V2 MRR (valid-json-only)": round(mrr_valid_only * 100, 2),
        "V4 JSON Valid Rate": round(v4_valid.mean() * 100, 2),
        "V2 JSON Valid Rate": round(v2_valid.mean() * 100, 2),
        "n_valid_v4": int(v4_valid.sum()),
        "n_valid_v2": int(v2_valid.sum()),
    }


def build_table() -> pd.DataFrame:
    rows = [recompute_row(size, mode, coll) for (size, mode), coll in TEST_COLLECTIONS.items()]
    return pd.DataFrame(rows)


# ---- plotting: same 3x1 layout as src/eval/paper_plots.py's
# plot_main_figure_3x1, but each slot gets a second, narrower "valid-json-only"
# bar drawn on top of the full-width "current" bar. ----

BASE_COLOR = "#4D4D4D"
LORA_COLOR = "#4C72B0"
FULL_COLOR = "#DD8452"
BLOCK_COLORS = {"base": BASE_COLOR, "lora": LORA_COLOR, "full": FULL_COLOR}
SLOT_OFFSETS = {"base": -0.28, "lora": 0.0, "full": 0.28}
SLOT_WIDTH = 0.26
INSET_WIDTH = SLOT_WIDTH * 0.42

PANELS = [
    ("ICD F1 Macro (current)", "ICD F1 Macro (valid-json-only)", "V4 JSON Valid Rate",
     "ICD F1 Macro", "ICD F1 Macro (%)"),
    ("ICD F1 Micro (current)", "ICD F1 Micro (valid-json-only)", "V4 JSON Valid Rate",
     "ICD F1 Micro", "ICD F1 Micro (%)"),
    ("V2 MRR (current)", "V2 MRR (valid-json-only)", "V2 JSON Valid Rate",
     "V2 MRR", "V2 MRR (%)"),
]


def _size_sort_key(size: str) -> float:
    return float(size.rstrip("b"))


def plot_3x1(table: pd.DataFrame):
    sizes = sorted(table["size"].unique(), key=_size_sort_key)
    x = np.arange(len(sizes))

    fig, axes = plt.subplots(1, 3, figsize=(3.5 * 3, 2.9))
    for ax, (cur_col, valid_col, rate_col, title, ylabel) in zip(axes, PANELS):
        # First pass: gather every bar's data so ymax (and thus label
        # offsets) is known before anything is drawn.
        bars = []
        for i, size in enumerate(sizes):
            for mode in MODES:
                row = table[(table["size"] == size) & (table["mode"] == mode)]
                if row.empty:
                    continue
                row = row.iloc[0]
                bars.append((x[i] + SLOT_OFFSETS[mode], BLOCK_COLORS[mode],
                             row[cur_col], row[valid_col], row[rate_col]))
        ymax = max((max(cur_v, valid_v) for _, _, cur_v, valid_v, _ in bars), default=0)

        for xpos, color, cur_v, valid_v, rate in bars:
            # full-width solid bar: today's metric (invalid JSON counted
            # as a miss).
            ax.bar(xpos, cur_v, SLOT_WIDTH, color=color, zorder=2)
            # narrower hatched bar on top, same x: metric recomputed with
            # invalid-JSON rows dropped entirely -- the "ceiling" if every
            # generation had parsed.
            ax.bar(xpos, valid_v, INSET_WIDTH, facecolor=color, edgecolor="black",
                   linewidth=0.6, hatch="////", alpha=0.85, zorder=3)

            # JSON valid rate, so the gap has an explicit number attached --
            # only label where validity actually dips below ~100%, otherwise
            # every single bar gets a "100%" tag and the figure is
            # unreadable (most bars ARE ~100% valid; the point of this
            # figure is to flag the handful that aren't).
            if rate < 99.5:
                ax.text(xpos, max(cur_v, valid_v) + ymax * 0.025, f"{rate:.0f}%",
                        ha="center", va="bottom", fontsize=7.5, color="#333333",
                        fontweight="bold")

        ax.set_ylim(0, ymax * 1.3 if ymax else 1)
        ax.set_xticks(x)
        ax.set_xticklabels(sizes, fontsize=9)
        ax.set_title(title, fontsize=10)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.tick_params(axis="both", labelsize=8)
        ax.spines[["top", "right"]].set_visible(False)

    handles = [
        Patch(facecolor=BASE_COLOR, label="base"),
        Patch(facecolor=LORA_COLOR, label="LoRA"),
        Patch(facecolor=FULL_COLOR, label="full FT"),
        Patch(facecolor="white", edgecolor="black", hatch="////", label="valid-JSON-only (recomputed)"),
    ]
    fig.legend(handles=handles, fontsize=8, ncol=4, loc="upper center",
               bbox_to_anchor=(0.5, 1.1), frameon=False)
    fig.suptitle("Solid = current metric (invalid JSON scored as a miss)  |  "
                  "Hatched = same metric on valid-JSON rows only  |  "
                  "% label shown only where JSON valid rate < 99.5%",
                  fontsize=8, y=1.0)
    fig.tight_layout(rect=[0, 0, 1, 0.86])

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(OUT_DIR / f"json_validity_impact_3x1.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {OUT_DIR / 'json_validity_impact_3x1.png'}")


def main():
    table = build_table()
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(OUT_CSV, index=False)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 250)
    print(table.to_string(index=False))
    print(f"\nWrote {OUT_CSV}")
    plot_3x1(table)


if __name__ == "__main__":
    main()
