"""Full-precision ICD code analysis -- same deterministic framework as
qa_deterministic.py (no LLM judge), comparing codes at two granularities:

  category  code[:3]   the level used everywhere else in the paper
  full      code       exact match, e.g. 'K5660'

(An earlier version of this script also had a 'chapter' (2-char) level; it
was cut -- 2026-07-27, Jan -- to keep the comparison to the two granularities
that matter: what the paper already reports (category) and the new thing
(full code). Re-add via LEVELS below if it's needed again; the truncation
machinery doesn't care how many levels there are.)

At each granularity we report BOTH aggregates, matching the main paper's own
distinction: micro-F1 (pool TP/FP/FN over all codes, so it's dominated by
frequent codes) and macro-F1 (F1 per individual code, then averaged across
codes unweighted, so rare codes count as much as common ones -- see
src/eval/json_validity_impact.py::calculate_icd_metrics for the paper's own
implementation of this split, which this mirrors exactly rather than
reinventing a different macro definition).

This answers a different question than the main-paper table: when a coarser
grouping is right, how often does that survive down to the exact code, i.e.
how specific is the model, not just how often is it in the right neighbourhood
-- and whether that specificity story holds up equally on rare codes (macro)
or only on frequent ones (micro).

Source of predictions: v4_json (raw generator output: a list of
{"icd_code": ..., "reason": ...} dicts), NOT v4_preds -- v4_preds is already
pre-truncated to 3 chars upstream (see scripts/qa_deterministic.py), so it
cannot answer a full-code question at all.

Gold ICD_CODES are already full MIMIC-style codes with no decimal point (e.g.
'K5660', 'Z23'). Predicted codes in v4_json are inconsistently formatted --
some models write 'K56.6', others 'K5660' or 'I10'; rarer malformed cells seen
in the data include stray internal spaces ('K 56.0'), underscores instead of a
dot ('I82_31'), comma-separated multi-code cells ('K86.0, K86.1'), and
hyphenated ranges ('C00-D04'). Every predicted code is normalized and validated
against a single-code ICD-10-CM pattern before comparison; anything that does
not match is dropped and counted (invalid_token_rate) rather than silently
corrupting the comparison.

Run:  python scripts/qa_full_code.py
      (also writes data/qa/full_code_by_granularity.csv and, if matplotlib
      is available, the bar figure via src.eval.qa_full_code_plot)
"""
from __future__ import annotations

import ast
import os
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

ROUND, DATA_ROOT = "1_4", "data"

# Same checkpoints as qa_deterministic.py (LoRA omitted here; add if needed).
MODELS = {
    "06b": {"base": "test-06b-base", "full": "test-06b-full-e4"},
    "8b":  {"base": "test-8b-base",  "full": "test-8b-full-e3"},
    "14b": {"base": "test-14b-base", "full": "test-14b-full-e3"},
    "32b": {"base": "test-32b-base", "full": "test-32b-full-e3"},
}

# Truncation length per granularity; None = no truncation (exact full code).
# Order matters -- it's the panel order in the figure, coarse to fine.
LEVELS: Dict[str, Optional[int]] = {"category": 3, "full": None}

# A single ICD-10-CM code: one letter, two digits (the category), then 0-4
# more alnum subcategory characters -- checked on the DOTLESS form, so it
# accepts both clinical notation ('K56.6') and MIMIC's own dotless storage
# format ('K5660', 'G4733'). Fine-tuned models overwhelmingly emit the latter
# (they were trained on MIMIC-formatted gold codes); an earlier version of
# this regex required a literal dot before any subcategory digits and
# therefore misclassified the majority of valid FT-model codes as garbage --
# verified against samples of what it was rejecting before shipping this.
# Known blind spot: a handful of ICD-10-CM categories put a letter in the
# category's digit slots (e.g. M1A, O9A, T-codes' 7th-character letter) --
# these are dropped as invalid. Rare enough not to matter here, but not zero.
CODE_RE = re.compile(r"^[A-Z]\d{2}[A-Z0-9]{0,4}$")


def load(name: str) -> pd.DataFrame:
    p = os.path.join(str(REPO_ROOT), DATA_ROOT, "results", "evaluation", ROUND,
                      f"eval_results_{name}", f"eval_results_{name}.pq")
    return pd.read_parquet(p)


def normalize_token(tok: str) -> Optional[str]:
    """One raw icd_code fragment -> a clean, dotless, validated code, or None
    if it doesn't look like a single valid ICD-10-CM code."""
    t = tok.strip().upper().replace("_", ".").replace(" ", "").replace(".", "")
    if not CODE_RE.match(t):
        return None
    return t


def split_and_normalize(raw: str) -> List[str]:
    """A single icd_code field is occasionally several comma/slash-separated
    codes ('K86.0, K86.1') rather than one; split first, then normalize each
    piece. A hyphenated range ('C00-D04') has no comma/slash so it is passed
    through whole and correctly rejected by CODE_RE (hyphen isn't valid)."""
    return [n for p in re.split(r"[,/]", raw) if (n := normalize_token(p))]


def full_pred_codes(v4_json_cell) -> Tuple[List[str], int, int]:
    """v4_json cell -> (ordered de-duplicated full codes, n_raw_entries,
    n_entries_that_yielded_no_valid_code)."""
    if not isinstance(v4_json_cell, str) or not v4_json_cell.strip():
        return [], 0, 0
    try:
        obj = ast.literal_eval(v4_json_cell.strip())
    except (ValueError, SyntaxError):
        return [], 0, 0
    if not isinstance(obj, list):
        return [], 0, 0
    seen: set = set()
    out: List[str] = []
    n_raw = n_dropped = 0
    for entry in obj:
        if not isinstance(entry, dict) or "icd_code" not in entry:
            continue
        raw = entry["icd_code"]
        if not isinstance(raw, str) or not raw.strip():
            continue
        n_raw += 1
        codes = split_and_normalize(raw)
        if not codes:
            n_dropped += 1
        for c in codes:
            if c not in seen:
                seen.add(c)
                out.append(c)
    return out, n_raw, n_dropped


def gold_full(arr) -> List[str]:
    return [str(c).strip().upper() for c in arr if str(c).strip()]


def trunc(code: str, n: Optional[int]) -> str:
    return code if n is None else code[:n]


def micro_macro(tp: dict, fp: dict, fn: dict) -> dict:
    """Per-code TP/FP/FN dicts (keyed by code at one granularity) -> both
    aggregates. Micro pools counts across all codes first, so it's dominated
    by whichever codes are frequent. Macro computes F1 (and precision/recall)
    per code, then averages unweighted across the label set, so a code seen
    twice counts as much as one seen 200 times -- identical definition to
    src/eval/json_validity_impact.py::calculate_icd_metrics, just extended to
    also report macro precision/recall, not only macro-F1."""
    TP, FP, FN = sum(tp.values()), sum(fp.values()), sum(fn.values())
    rec_mi = TP / (TP + FN) if TP + FN else 0.0
    prec_mi = TP / (TP + FP) if TP + FP else 0.0
    f1_mi = 2 * prec_mi * rec_mi / (prec_mi + rec_mi) if prec_mi + rec_mi else 0.0

    labels = set(tp) | set(fp) | set(fn)
    recs, precs, f1s = [], [], []
    for label in labels:
        t, f_p, f_n = tp[label], fp[label], fn[label]
        r = t / (t + f_n) if t + f_n else 0.0
        p = t / (t + f_p) if t + f_p else 0.0
        recs.append(r); precs.append(p)
        f1s.append(2 * p * r / (p + r) if p + r else 0.0)
    n_labels = len(labels)
    rec_ma = sum(recs) / n_labels if n_labels else 0.0
    prec_ma = sum(precs) / n_labels if n_labels else 0.0
    f1_ma = sum(f1s) / n_labels if n_labels else 0.0

    return {
        "recall_micro": rec_mi, "precision_micro": prec_mi, "f1_micro": f1_mi,
        "recall_macro": rec_ma, "precision_macro": prec_ma, "f1_macro": f1_ma,
        "n_labels": n_labels,
    }


def score_model(df: pd.DataFrame) -> dict:
    n = len(df)
    tp = {lvl: defaultdict(int) for lvl in LEVELS}
    fp = {lvl: defaultdict(int) for lvl in LEVELS}
    fn = {lvl: defaultdict(int) for lvl in LEVELS}
    cat_hit_exact = cat_hit_inexact = 0   # conditional on the 3-char category being covered
    n_tokens = n_invalid = 0

    for _, r in df.iterrows():
        g = gold_full(r["ICD_CODES"])
        gset = set(g)
        p, n_raw, n_drop = full_pred_codes(r["v4_json"])
        n_tokens += n_raw
        n_invalid += n_drop
        pset = set(p)

        for lvl, ln in LEVELS.items():
            gl = {trunc(c, ln) for c in gset}
            pl = {trunc(c, ln) for c in pset}
            for code in gl & pl:
                tp[lvl][code] += 1
            for code in pl - gl:
                fp[lvl][code] += 1
            for code in gl - pl:
                fn[lvl][code] += 1

        p3 = {c[:3] for c in pset}
        for gc in gset:
            if gc[:3] in p3:
                if gc in pset:
                    cat_hit_exact += 1
                else:
                    cat_hit_inexact += 1

    out = {"n": n}
    for lvl in LEVELS:
        mm = micro_macro(tp[lvl], fp[lvl], fn[lvl])
        for k, v in mm.items():
            out[f"{k}_{lvl}"] = v

    denom = cat_hit_exact + cat_hit_inexact
    out["specificity_given_cat_hit"] = cat_hit_exact / denom if denom else 0.0
    out["cat_hit_exact"] = cat_hit_exact
    out["cat_hit_inexact"] = cat_hit_inexact
    out["invalid_token_rate"] = (n_invalid / n_tokens) if n_tokens else 0.0
    return out


def to_long(tab: pd.DataFrame) -> pd.DataFrame:
    """Wide per-checkpoint table -> one row per (size, variant, granularity,
    metric_type), the shape src/eval/qa_full_code_plot.py expects."""
    rows = []
    for idx, row in tab.iterrows():
        size, variant = idx.rsplit("_", 1)
        for lvl in LEVELS:
            for mtype in ("micro", "macro"):
                rows.append({
                    "size": size, "variant": variant, "granularity": lvl,
                    "metric_type": mtype,
                    "recall": row[f"recall_{mtype}_{lvl}"],
                    "precision": row[f"precision_{mtype}_{lvl}"],
                    "f1": row[f"f1_{mtype}_{lvl}"],
                })
    return pd.DataFrame(rows)


def main():
    rows = {}
    for size, variants in MODELS.items():
        for variant, name in variants.items():
            rows[f"{size}_{variant}"] = score_model(load(name))
    tab = pd.DataFrame(rows).T

    count_cols = ["n", "cat_hit_exact", "cat_hit_inexact"] + \
        [f"n_labels_{lvl}" for lvl in LEVELS]
    pct_cols = [c for c in tab.columns if c not in count_cols]
    tab_pct = tab.copy()
    tab_pct[pct_cols] = (tab_pct[pct_cols] * 100).round(1)

    pd.set_option("display.width", 260)
    pd.set_option("display.max_columns", None)
    print("=== Per-checkpoint rates (%) ===")
    print(tab_pct.to_string())

    deltas = {}
    for size in MODELS:
        b, f = f"{size}_base", f"{size}_full"
        deltas[size] = (tab_pct.loc[f, pct_cols] - tab_pct.loc[b, pct_cols]).round(1)
    print("\n=== base -> full delta (pp; negative = fewer misses / lower rate) ===")
    print(pd.DataFrame(deltas).T.to_string())

    out_wide = os.path.join(str(REPO_ROOT), DATA_ROOT, "qa", "full_code_rates.csv")
    tab_pct.to_csv(out_wide)
    print(f"\n-> {out_wide}")

    # Long/plot-ready format keeps fractions (0-1), not the *100-rounded
    # display table, so the figure isn't built on doubly-rounded numbers.
    long_df = to_long(tab)
    out_long = os.path.join(str(REPO_ROOT), DATA_ROOT, "qa", "full_code_by_granularity.csv")
    long_df.to_csv(out_long, index=False)
    print(f"-> {out_long}")

    try:
        from src.eval.qa_full_code_plot import plot as plot_full_code
        plot_full_code()
    except ImportError as e:
        print(f"\n(skipping figure -- {e}; run `python -m src.eval.qa_full_code_plot` "
              "separately once matplotlib/deps are available)")


if __name__ == "__main__":
    main()
