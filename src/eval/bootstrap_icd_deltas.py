"""Case-level paired bootstrap CIs for 8B ICD macro-F1 deltas.

Compares MERLIN (test-8b-full-e3) against:
  - label-only ablation (test-8b-mimic-icd2-e4)
  - base / untuned Qwen3-8B (test-8b-base)

Replicates calculate_icd_metrics_cpu from src/eval/classification_metrics.py
exactly (labels = union of TP/FP/FN codes recomputed per resample) and
convert_codes_to_short_codes from src/utils.py for gold codes.
"""
import ast
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent.parent
BASE_DIR = REPO / "data" / "results" / "evaluation" / "1_4"

def _pq(name):
    return BASE_DIR / f"eval_results_{name}" / f"eval_results_{name}.pq"


FILES = {
    "merlin": _pq("test-8b-full-e3"),
    "label_only": _pq("test-8b-mimic-icd2-e4"),
    "base": _pq("test-8b-base"),
}

N_BOOT = 2000
SEED = 42


def convert_code_to_short_code(code, icd_version=10):
    return code[:4] if icd_version == 9 and code.lower().startswith("e") else code[:3]


def convert_codes_to_short_codes(codes, icd_version=10):
    return list(set(convert_code_to_short_code(c, icd_version) for c in codes))


def safe_parse_list(value):
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return []
    if isinstance(value, str):
        stripped = value.strip()
        if stripped in ("", "None", "nan", "NaN", "null"):
            return []
        try:
            parsed = ast.literal_eval(stripped)
        except (ValueError, SyntaxError):
            return []
        return parsed if isinstance(parsed, list) else []
    if isinstance(value, (list, tuple, np.ndarray)):
        return list(value)
    return []


def macro_f1(y_pred, y_true, labels=None):
    """Macro-F1 over `labels`.

    If `labels` is None, the label universe is derived from whatever TP/FP/FN
    codes appear in this particular y_pred/y_true call (the original
    behaviour). When bootstrapping, this recomputation-per-resample is a bias
    source: a code with support in only one case has a (1 - 1/n)^n ~= 37%
    chance of being absent from any given n-out-of-n resample, and when that
    happens the code silently drops out of the macro average instead of
    surviving as an F1=0 entry -- inflating the resampled statistic relative
    to the full-sample point estimate. Pass a frozen `labels` set (computed
    once from the full, non-resampled data) to both the point estimate and
    every bootstrap draw to remove this bias.
    """
    tp, fp, fn = defaultdict(int), defaultdict(int), defaultdict(int)
    for pred, true in zip(y_pred, y_true):
        pred_set, true_set = set(pred), set(true)
        for c in pred_set & true_set:
            tp[c] += 1
        for c in pred_set - true_set:
            fp[c] += 1
        for c in true_set - pred_set:
            fn[c] += 1
    if labels is None:
        labels = set(tp) | set(fp) | set(fn)
    if not labels:
        return 0.0
    f1s = []
    for label in labels:
        p = tp[label] / (tp[label] + fp[label]) if tp[label] + fp[label] else 0
        r = tp[label] / (tp[label] + fn[label]) if tp[label] + fn[label] else 0
        f1s.append(2 * p * r / (p + r) if p + r else 0)
    return sum(f1s) / len(f1s)


def load(path):
    df = pd.read_parquet(path, columns=["hadm_id", "ICD_CODES", "v4_preds"])
    df["gold"] = df["ICD_CODES"].apply(lambda x: convert_codes_to_short_codes(list(x)))
    df["pred"] = df["v4_preds"].apply(lambda x: convert_codes_to_short_codes(safe_parse_list(x)))
    return df.set_index("hadm_id")[["gold", "pred"]]


def main():
    data = {name: load(path) for name, path in FILES.items()}
    for name, df in data.items():
        print(f"{name}: {len(df)} rows")

    common = data["merlin"].index
    for name, df in data.items():
        common = common.intersection(df.index)
    common = common.sort_values()
    print(f"common hadm_ids across all three: {len(common)}")

    gold = {name: df.loc[common, "gold"].tolist() for name, df in data.items()}
    pred = {name: df.loc[common, "pred"].tolist() for name, df in data.items()}
    # sanity: gold codes should be identical across variants for the same hadm_id
    for name in ("label_only", "base"):
        mismatches = sum(1 for a, b in zip(gold["merlin"], gold[name]) if set(a) != set(b))
        print(f"gold mismatches merlin vs {name}: {mismatches} / {len(common)}")

    n = len(common)

    # Freeze the label universe from the full, non-resampled data (union of
    # gold codes and every model's predicted codes) so every bootstrap draw
    # is averaged over the *same* set of labels as the point estimate. See
    # macro_f1()'s docstring for why recomputing this per resample biases
    # the bootstrap distribution.
    labels = set()
    for g in gold["merlin"]:
        labels |= set(g)
    for name in data:
        for p in pred[name]:
            labels |= set(p)
    print(f"\nFrozen label universe: {len(labels)} codes")

    point = {name: macro_f1(pred[name], gold["merlin"], labels=labels) for name in data}
    print("\nPoint estimates (macro-F1, %):")
    for name, v in point.items():
        print(f"  {name}: {100*v:.2f}")

    rng = np.random.default_rng(SEED)
    deltas = {"merlin_minus_label_only": [], "merlin_minus_base": []}
    pred_arrs = {name: np.array(pred[name], dtype=object) for name in data}
    gold_arr = np.array(gold["merlin"], dtype=object)

    for _ in range(N_BOOT):
        idx = rng.integers(0, n, size=n)
        g = gold_arr[idx]
        f_merlin = macro_f1(pred_arrs["merlin"][idx], g, labels=labels)
        f_label = macro_f1(pred_arrs["label_only"][idx], g, labels=labels)
        f_base = macro_f1(pred_arrs["base"][idx], g, labels=labels)
        deltas["merlin_minus_label_only"].append(f_merlin - f_label)
        deltas["merlin_minus_base"].append(f_merlin - f_base)

    print(f"\n{N_BOOT}-resample paired bootstrap (case-level, seed={SEED}):")
    for key, vals in deltas.items():
        vals = np.array(vals) * 100
        point_delta = 100 * (
            point["merlin"] - point["label_only" if "label_only" in key else "base"]
        )
        lo, hi = np.percentile(vals, [2.5, 97.5])
        print(f"  {key}: point={point_delta:+.2f}  95% CI=[{lo:+.2f}, {hi:+.2f}]  (pts macro-F1)")


if __name__ == "__main__":
    main()
