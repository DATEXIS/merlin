"""Acute/current-diagnosis-only ICD macro-F1: does the MERLIN vs base gain
survive after excluding history (Z8x/Z9x), medication-status (Z79), and
chronic-comorbidity (Elixhauser-derived) codes?

Category rules copied verbatim from scripts/qa_deterministic.py (CHRONIC set,
history/medication prefix rules) so this matches the paper's existing
history/medication/chronic bucket definitions exactly.
"""
import numpy as np
import pandas as pd

from src.eval.bootstrap_icd_deltas import BASE_DIR, N_BOOT, SEED
from src.eval.bootstrap_icd_deltas import load as load_variant
from src.eval.bootstrap_icd_deltas import macro_f1


def _pq(name):
    return BASE_DIR / f"eval_results_{name}" / f"eval_results_{name}.pq"


FILES = {
    "merlin": _pq("test-8b-full-e3"),
    "base": _pq("test-8b-base"),
}


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


def is_excluded(code):
    c = code[:3]
    return c.startswith(("Z8", "Z9")) or c == "Z79" or c in CHRONIC


def acute_only(codes):
    return [c for c in codes if not is_excluded(c)]


def main():
    data = {name: load_variant(path) for name, path in FILES.items()}
    common = data["merlin"].index.intersection(data["base"].index).sort_values()
    print(f"common cases: {len(common)}")

    gold_full = data["merlin"].loc[common, "gold"].tolist()
    pred_merlin_full = data["merlin"].loc[common, "pred"].tolist()
    pred_base_full = data["base"].loc[common, "pred"].tolist()

    n_gold_codes = sum(len(g) for g in gold_full)
    gold_acute = [acute_only(g) for g in gold_full]
    n_gold_acute = sum(len(g) for g in gold_acute)
    pred_merlin_acute = [acute_only(p) for p in pred_merlin_full]
    pred_base_acute = [acute_only(p) for p in pred_base_full]

    print(f"gold codes total: {n_gold_codes}, acute-only: {n_gold_acute} "
          f"({100*n_gold_acute/n_gold_codes:.1f}% retained; "
          f"{100*(1-n_gold_acute/n_gold_codes):.1f}% excluded as history/medication/chronic)")

    n_cases_with_any_acute_gold = sum(1 for g in gold_acute if g)
    print(f"cases with >=1 acute gold code: {n_cases_with_any_acute_gold} / {len(common)}")

    # Freeze label universes from the full, non-resampled data -- one for the
    # full-code comparison, one for the acute-only comparison -- so bootstrap
    # resamples are averaged over the same labels as the point estimate
    # instead of recomputing (and shrinking) the label set per resample. See
    # macro_f1()'s docstring in bootstrap_icd_deltas.py for why this matters.
    labels_full = set()
    for g in gold_full:
        labels_full |= set(g)
    for p in (pred_merlin_full, pred_base_full):
        for codes in p:
            labels_full |= set(codes)

    labels_acute = set()
    for g in gold_acute:
        labels_acute |= set(g)
    for p in (pred_merlin_acute, pred_base_acute):
        for codes in p:
            labels_acute |= set(codes)

    f_merlin_full = macro_f1(pred_merlin_full, gold_full, labels=labels_full)
    f_base_full = macro_f1(pred_base_full, gold_full, labels=labels_full)
    f_merlin_acute = macro_f1(pred_merlin_acute, gold_acute, labels=labels_acute)
    f_base_acute = macro_f1(pred_base_acute, gold_acute, labels=labels_acute)

    print("\nMacro-F1 (%):")
    print(f"  full (all codes):   base={100*f_base_full:.2f}  merlin={100*f_merlin_full:.2f}  "
          f"delta={100*(f_merlin_full-f_base_full):+.2f}")
    print(f"  acute-only:         base={100*f_base_acute:.2f}  merlin={100*f_merlin_acute:.2f}  "
          f"delta={100*(f_merlin_acute-f_base_acute):+.2f}")

    # bootstrap CI on the acute-only delta
    n = len(common)
    rng = np.random.default_rng(SEED)
    gold_acute_arr = np.array(gold_acute, dtype=object)
    pred_merlin_acute_arr = np.array(pred_merlin_acute, dtype=object)
    pred_base_acute_arr = np.array(pred_base_acute, dtype=object)
    deltas = []
    for _ in range(N_BOOT):
        idx = rng.integers(0, n, size=n)
        g = gold_acute_arr[idx]
        fm = macro_f1(pred_merlin_acute_arr[idx], g, labels=labels_acute)
        fb = macro_f1(pred_base_acute_arr[idx], g, labels=labels_acute)
        deltas.append(fm - fb)
    deltas = np.array(deltas) * 100
    lo, hi = np.percentile(deltas, [2.5, 97.5])
    point = 100 * (f_merlin_acute - f_base_acute)
    print(f"\nAcute-only bootstrap ({N_BOOT} resamples, seed={SEED}): "
          f"delta={point:+.2f}  95% CI=[{lo:+.2f}, {hi:+.2f}]")


if __name__ == "__main__":
    main()
