"""Deterministic ICD error scoring -- every class that reduces to a set operation
on gold vs. predicted 3-char categories. No LLM judge involved.

Runs all 8 checkpoints ({06b,8b,14b,32b} x {base,full}) on the test split and
prints per-class rates (% of cases) plus the base->full delta at each size.

Principal diagnosis = gold[0] (MIMIC seq_num order; seq_num=1 is principal --
verified against discharge 'Primary Diagnosis' sections).

Classes computed here:
  micro recall / precision / F1        (model quality context)
  missed_primary        gold[0] category not covered by prediction
  missed_history        a gold Z80-Z99 category not covered
  missed_medication     a gold Z79 category not covered
  missed_chronic        a gold chronic-comorbidity category (Elixhauser) not covered
  missed_any            any gold category not covered (pure omission; ~always true)
  symptom_fp            predicts an R00-R99 symptom category absent from gold
  extra_codes           predicts more categories than gold (over-prediction)
  duplicate_subcodes    two predicted full codes share a 3-char category
  hallucinated_fp       predicts a category absent from gold (any false positive)
  recall/precision/f1_{history,medication,chronic}   see note below

Note on direction (2026-07-27): the missed_* rates above are framed so
LOWER is better, which sits awkwardly next to a table where every other
column (Recall, Prec., F1, head/body/tail) has HIGHER is better -- a reader
scanning the table has to remember which columns invert. For history/
medication/chronic (genuine multi-code subsets, unlike missed_primary, which
is a single code per case -- see below) we additionally compute a real
class-conditional precision/recall/F1, pooling TP/FP/FN over only the codes
that belong to that subset (e.g. for 'history': gold Z80-Z99 codes covered =
TP, predicted Z80-Z99 codes not in gold = FP, gold Z80-Z99 codes missed = FN).
This is a genuine P/R/F1 -- HIGHER is better throughout -- not just an
inverted restatement of missed_X, because it also captures precision within
that code family (does the model predict the *right* history code, not just
*a* history code), which missed_X never measured at all.

missed_primary has no equivalent F1: it's a single designated code per case
(gold[0]) -- confirmed 2026-07-27: every one of the 2,184 test cases has
>=1 gold ICD code (min=1, so "primary" is always exactly 1 code, never 0 or
ambiguous between several). Because it's a single code, "precision" restricted
to it is trivially ~100% (any predicted code that happens to equal gold[0] is
by definition correct -- there's no way to be "wrong" under a lens that only
looks at that one code), so a computed F1 would just be a monotonic reshuffling
of recall, not new information. We report recall_primary (= 100 - missed_primary,
direction-flipped for the same reason as the others).

recall_primary_at_{1,2,3}: instead of forcing an F1 for primary, we reuse the
Recall@k framing already established for diagnosis ranking (V2) elsewhere in
the paper, because it fits this case exactly: the V4 prompt explicitly asks
the model to rank predicted codes "in order of likelihood", and v4_preds
preserves that emission order, so "is gold[0] within the model's own top-k"
is a real, already-meaningful question -- not one we have to invent a new
metric to ask. hit@k counts a case as a hit if gold[0] appears among the
first k *distinct* predicted categories (duplicates collapsed, first
occurrence keeps its rank).

Run:  python scripts/qa_deterministic.py
"""
from __future__ import annotations

import ast
import os
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

ROUND, DATA_ROOT = "1_4", "data"

MODELS = {
    "06b": {"base": "test-06b-base", "full": "test-06b-full-e4", "lora": "test-06b-r128-e4"},
    "8b":  {"base": "test-8b-base",  "full": "test-8b-full-e3",  "lora": "test-8b-r128-e3"},
    "14b": {"base": "test-14b-base", "full": "test-14b-full-e3", "lora": "test-14b-r128-e3"},
    "32b": {"base": "test-32b-base", "full": "test-32b-full-e3", "lora": "test-32b-r256-e4"},
}
# Note: 8b lora (r128) is a known-degenerate checkpoint (~30% empty predictions);
# its inflated error rates are a JSON-validity artifact, not a real error profile.

# --- Elixhauser-derived chronic 3-char categories (chronic-focused; indicative) ---
def _rng(letter, lo, hi):
    return {f"{letter}{i:02d}" for i in range(lo, hi + 1)}

CHRONIC = set()
CHRONIC |= {"I50", "I11", "I13", "I09", "I42", "I43", "I25"}
CHRONIC |= {"I44", "I45", "I47", "I48", "I49"}
CHRONIC |= _rng("I", 5, 8) | {"I34", "I35", "I36", "I37", "I38", "I39"}
CHRONIC |= {"I26", "I27", "I28", "I70", "I71", "I73"}
CHRONIC |= {"I10", "I12", "I15"}
CHRONIC |= {"G20", "G35", "G40", "G41", "G81", "G82", "G83"}
CHRONIC |= {f"J{i:02d}" for i in range(40, 48)} | {f"J{i:02d}" for i in range(60, 68)}
CHRONIC |= {"E10", "E11", "E13", "E14", "E00", "E01", "E02", "E03", "E89"}
CHRONIC |= {"N18", "N19", "N25"}
CHRONIC |= {"K70", "K71", "K72", "K73", "K74", "K76", "B18", "I85"}
CHRONIC |= {"K25", "K26", "K27", "K28"}
CHRONIC |= {"B20", "B21", "B22", "B23", "B24"}
CHRONIC |= _rng("C", 0, 96)
CHRONIC |= {"M05", "M06", "M08", "M30", "M31", "M32", "M33", "M34", "M35", "M36", "L94"}
CHRONIC |= {"D65", "D66", "D67", "D68", "D69", "D50", "D51", "D52", "D53", "D63", "D64"}
CHRONIC |= {"E66"}
CHRONIC |= {f"F{i:02d}" for i in range(10, 20)}
CHRONIC |= {"F20", "F22", "F23", "F24", "F25", "F28", "F29", "F31", "F32", "F33", "F34"}

SYMPTOM = _rng("R", 0, 99)


def load(name):
    p = os.path.join(str(REPO_ROOT), DATA_ROOT, "results", "evaluation", ROUND,
                     f"eval_results_{name}", f"eval_results_{name}.pq")
    return pd.read_parquet(p)


def pred_lists(v):
    """v4_preds -> (set of 3-char cats, list of 3-char cats incl. duplicates)."""
    if v is None:
        return set(), []
    try:
        lst = ast.literal_eval(v) if isinstance(v, str) else list(v)
    except (ValueError, SyntaxError):
        return set(), []
    if not isinstance(lst, (list, tuple)):  # degenerate outputs (e.g. "None")
        return set(), []
    cats = [str(c).strip()[:3] for c in lst if str(c).strip()]
    return set(cats), cats


def gold_cats(arr):
    return [str(c).strip()[:3] for c in arr if str(c).strip()]


def _prf(inter, tg, tp):
    rec = inter / tg if tg else 0.0
    prec = inter / tp if tp else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return rec, prec, f1


PRIMARY_KS = (1, 2, 3)


def ordered_unique(cats):
    """First-occurrence-order de-dup -- a repeated category doesn't get to
    occupy two ranks, so Recall@k reflects the model's k *distinct* top
    guesses, matching how the existing V2 Recall@k is computed."""
    seen, out = set(), []
    for c in cats:
        if c not in seen:
            seen.add(c)
            out.append(c)
    return out


def score_model(df):
    n = len(df)
    inter = tg = tp = 0
    # Pooled TP/tg/tp per code-family, restricted to codes actually IN that
    # family on both sides -- this is what makes precision meaningful (a
    # predicted Z8x/Z9x code that ISN'T gold is a real FP here, which
    # missed_history could never see since it only checks coverage).
    class_counts = {"history": [0, 0, 0], "medication": [0, 0, 0], "chronic": [0, 0, 0]}
    acc = {k: 0 for k in ["missed_primary", "missed_history", "missed_medication",
                          "missed_chronic", "missed_any", "symptom_fp", "extra_codes",
                          "duplicate_subcodes", "hallucinated_fp"]}
    acc.update({f"recall_primary_at_{k}": 0 for k in PRIMARY_KS})
    for _, r in df.iterrows():
        gl = gold_cats(r["ICD_CODES"])
        g = set(gl)
        p, pcats = pred_lists(r["v4_preds"])
        inter += len(g & p); tg += len(g); tp += len(p)
        if gl:
            acc["missed_primary"] += (gl[0] not in p)
            ranked = ordered_unique(pcats)
            for k in PRIMARY_KS:
                acc[f"recall_primary_at_{k}"] += (gl[0] in ranked[:k])
        gh = {c for c in g if c.startswith(("Z8", "Z9"))}
        gm = {c for c in g if c == "Z79"}
        gc = {c for c in g if c in CHRONIC}
        ph = {c for c in p if c.startswith(("Z8", "Z9"))}
        pm = {c for c in p if c == "Z79"}
        pc = {c for c in p if c in CHRONIC}
        for key, gset_c, pset_c in (("history", gh, ph), ("medication", gm, pm),
                                    ("chronic", gc, pc)):
            cc = class_counts[key]
            cc[0] += len(gset_c & pset_c); cc[1] += len(gset_c); cc[2] += len(pset_c)
        acc["missed_history"] += bool(gh and (gh - p))
        acc["missed_medication"] += bool(gm and (gm - p))
        acc["missed_chronic"] += bool(gc and (gc - p))
        acc["missed_any"] += bool(g - p)
        fp = p - g
        acc["hallucinated_fp"] += bool(fp)
        acc["symptom_fp"] += bool({c for c in fp if c in SYMPTOM})
        acc["extra_codes"] += (len(p) > len(g))
        acc["duplicate_subcodes"] += (len(pcats) != len(set(pcats)))
    rec, prec, f1 = _prf(inter, tg, tp)
    out = {"recall": rec, "precision": prec, "micro_f1": f1}
    for key, (i_, t_g, t_p) in class_counts.items():
        r_, p_, f_ = _prf(i_, t_g, t_p)
        out[f"recall_{key}"] = r_
        out[f"precision_{key}"] = p_
        out[f"f1_{key}"] = f_
    out.update({k: v / n for k, v in acc.items()})
    out["recall_primary"] = 1 - out["missed_primary"]  # same direction as the rest
    return out


def main():
    rows = {}
    for size, variants in MODELS.items():
        for variant, name in variants.items():
            rows[f"{size}_{variant}"] = score_model(load(name))
    tab = pd.DataFrame(rows).T
    tab = (tab * 100).round(1)

    pd.set_option("display.width", 260)
    pd.set_option("display.max_columns", None)
    print("=== Per-checkpoint rates (%) ===")
    print(tab.to_string())

    # base->full delta at each size
    deltas = {}
    for size in MODELS:
        b, f = f"{size}_base", f"{size}_full"
        deltas[size] = (tab.loc[f] - tab.loc[b]).round(1)
    dtab = pd.DataFrame(deltas).T
    print("\n=== base -> full delta (pp; negative = improves) ===")
    print(dtab.to_string())

    out = os.path.join(str(REPO_ROOT), DATA_ROOT, "qa", "deterministic_rates.csv")
    tab.to_csv(out)
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
