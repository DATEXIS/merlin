"""
Chief-complaint breakdown of the paper's headline metric set --
Symptoms (DotProduct), Diagnoses (V2 Recall@3), ICD F1 Micro, ICD F1 Macro --
for ONE fixed checkpoint: test-32b-full-e3 (TEST split).

Model choice (Jan, 2026-07-28): 32b-full-e3 over 8b-full-e3 -- best ICD F1
Macro (the paper's stated primary metric) and ICD F1 Micro among all full-FT
checkpoints; has a mild 41/2184 (1.9%) V2 parse-failure rate on TEST (vs 0
for 8b-full-e3), not large enough to distort a per-complaint cut.

Metric formulas are the SAME functions the aggregate eval_metrics CSV uses
(src/eval/classification_metrics.py + eval_experiments.py), just scoped to
each Chief Complaint subset instead of the whole test set; reimplemented
here in numpy/pure Python (not imported) because the eval module pulls in
torch + sentence-transformers which aren't needed for these formulas and
weren't available in this environment. Verified byte-for-byte against
data/eval_metrics_merlin-eval-1.4.csv's test-32b-full-e3 row before running
per-complaint (whole-set numbers matched to within rounding: DotProduct
59.18 vs 59.17, ICD F1 Micro/Macro 50.49/20.99 exact, Recall@3 75.09 exact).

Notes on the per-complaint cut:
- disease_vector (symptom multi-hot vector) has a FIXED length per chief
  complaint (13..30 dims depending on complaint -- see the per-CC symptom
  scheme in data/medical_schemes), so DotProduct within one complaint is a
  flat mean, not a weighted average across length-groups.
- ICD F1 Macro is macro over the ICD codes that actually appear (as gold or
  prediction) WITHIN that chief complaint's rows -- a smaller, complaint-
  specific label universe than the whole-test-set macro-F1, so these numbers
  are not a decomposition of the single 20.99 headline figure (they can't be
  weighted back up to it) and shouldn't be read as parts of a whole -- they
  answer "how good is this complaint's diagnosis-to-ICD mapping", not
  "what part of the corpus number comes from this complaint".

Run from repo root: python scripts/analysis/cc_metrics_table_32b_full.py
"""
import ast
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

CKPT = "test-32b-full-e3"
EVAL_PQ = Path(f"data/results/evaluation/1_4/eval_results_{CKPT}/eval_results_{CKPT}.pq")
OUT_CSV = Path("data/results/evaluation/cc_metrics_table_32b_full.csv")
# Body-only, matching src/eval/paper_tables.py's convention: this script
# regenerates table_cc_32b_full_body.tex; the wrapper (table_cc_32b_full.tex,
# \input + \caption + \label + \end{table*}) is hand-maintained, same split
# as table_main.tex/table_main_body.tex.
OUT_TEX_BODY = Path("paper/EACL_2026_v2/table_cc_32b_full_body.tex")


def safe_parse_list(value) -> list:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return []
    if isinstance(value, str):
        s = value.strip()
        if s in ("", "None", "nan", "NaN", "null"):
            return []
        try:
            parsed = ast.literal_eval(s)
        except (ValueError, SyntaxError):
            return []
        return parsed if isinstance(parsed, list) else []
    if isinstance(value, (list, tuple, np.ndarray)):
        return list(value)
    return []


def short_codes(codes) -> list:
    """convert_codes_to_short_codes with the icd_version=10 default: code[:3]."""
    return list({c[:3] for c in codes})


def symptom_metrics(sub: pd.DataFrame) -> float:
    """DotProduct (%), matching calculate_batch_symptom_extraction_metrics."""
    preds_raw = sub["v1_preds"].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x).tolist()
    labels = sub["disease_vector"].tolist()
    clean_preds = [
        [0] * len(l) if (p is None or (isinstance(p, float) and np.isnan(p))) else p
        for p, l in zip(preds_raw, labels)
    ]
    tmp = pd.DataFrame({"p": clean_preds, "l": labels})
    tmp["len"] = tmp["l"].map(len)
    group_stats = []
    for _, group in tmp.groupby("len"):
        p_np = np.array(group["p"].tolist(), dtype=float)
        l_np = np.array(group["l"].tolist(), dtype=float)
        dot = (p_np * l_np).sum(axis=1)
        k = (l_np != 0).sum(axis=1)
        dp = np.where(k > 0, (dot + k) / (2 * k), 0.0)
        group_stats.append({"count": len(group), "dot": dp.mean()})
    total = sum(g["count"] for g in group_stats)
    return sum(g["dot"] * g["count"] for g in group_stats) / total * 100


def icd_f1(sub: pd.DataFrame) -> tuple:
    """(ICD F1 Micro %, ICD F1 Macro %), matching calculate_icd_metrics_cpu."""
    v4_pred = [short_codes(safe_parse_list(v)) for v in sub["v4_preds"]]
    v4_label = [short_codes(list(x)) for x in sub["ICD_CODES"]]
    tp, fp, fn = defaultdict(int), defaultdict(int), defaultdict(int)
    for pred, true in zip(v4_pred, v4_label):
        ps, ts = set(pred), set(true)
        for c in ps & ts:
            tp[c] += 1
        for c in ps - ts:
            fp[c] += 1
        for c in ts - ps:
            fn[c] += 1
    labels = set(tp) | set(fp) | set(fn)
    TP, FP, FN = sum(tp.values()), sum(fp.values()), sum(fn.values())
    prec_mi = TP / (TP + FP) if TP + FP else 0
    rec_mi = TP / (TP + FN) if TP + FN else 0
    f1_mi = 2 * prec_mi * rec_mi / (prec_mi + rec_mi) if prec_mi + rec_mi else 0
    f1s = []
    for l in labels:
        p = tp[l] / (tp[l] + fp[l]) if tp[l] + fp[l] else 0
        r = tp[l] / (tp[l] + fn[l]) if tp[l] + fn[l] else 0
        f1s.append(2 * p * r / (p + r) if p + r else 0)
    f1_ma = sum(f1s) / len(f1s) if f1s else 0
    return f1_mi * 100, f1_ma * 100


def _parse_preds(v):
    if not isinstance(v, str):
        return []
    try:
        p = ast.literal_eval(v)
        return p if isinstance(p, list) else []
    except Exception:
        return []


def recall_at_k(sub: pd.DataFrame, k: int) -> float:
    hits = [gold in _parse_preds(preds)[:k] for gold, preds in zip(sub["disease"], sub["v2_preds"])]
    return 100 * sum(hits) / len(hits)


def main():
    df = pd.read_parquet(EVAL_PQ)
    n_by_cc = df["Chief Complaint"].value_counts()
    cc_order = list(n_by_cc.sort_values(ascending=False).index)

    rows = []
    for cc in cc_order + ["All"]:
        sub = df if cc == "All" else df[df["Chief Complaint"] == cc]
        f1_mi, f1_ma = icd_f1(sub)
        rows.append({
            "Chief Complaint": cc,
            "n": len(sub),
            "Symptoms (DotProduct)": round(symptom_metrics(sub), 1),
            "Diagnoses (Recall@1)": round(recall_at_k(sub, 1), 1),
            "Diagnoses (Recall@3)": round(recall_at_k(sub, 3), 1),
            "ICD F1 Micro": round(f1_mi, 1),
            "ICD F1 Macro": round(f1_ma, 1),
        })
    table = pd.DataFrame(rows).set_index("Chief Complaint")

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(OUT_CSV)
    print(table.to_string())
    print(f"\nsaved table -> {OUT_CSV}")

    # --- LaTeX body, mirroring table_main's column layout (src/eval/paper_tables.py) ---
    lines = [
        "% GENERATED by scripts/analysis/cc_metrics_table_32b_full.py -- do not edit by hand.",
        "\\begin{table*}[t]",
        "\\centering",
        "\\small",
        "\\setlength{\\tabcolsep}{4pt}",
        "\\begin{tabular}{lcccccc}",
        "\\toprule",
        "\\textbf{Chief complaint} & \\textbf{n} & \\textbf{NormDot} & \\textbf{R@1} & \\textbf{R@3} & "
        "$\\mathbf{F1}_{\\mathrm{mic}}$ & $\\mathbf{F1}_{\\mathrm{mac}}$ \\\\",
        "\\midrule",
    ]
    for cc, row in table.iterrows():
        if cc == "All":
            lines.append("\\midrule")
        label = cc if cc == "All" else cc.title()
        lines.append(
            f"{label} & {int(row['n'])} & ${row['Symptoms (DotProduct)']:.1f}$ & "
            f"${row['Diagnoses (Recall@1)']:.1f}$ & ${row['Diagnoses (Recall@3)']:.1f}$ & "
            f"${row['ICD F1 Micro']:.1f}$ & ${row['ICD F1 Macro']:.1f}$ \\\\"
        )
    lines += ["\\bottomrule", "\\end{tabular}"]
    OUT_TEX_BODY.parent.mkdir(parents=True, exist_ok=True)
    OUT_TEX_BODY.write_text("\n".join(lines) + "\n")
    print(f"saved LaTeX body -> {OUT_TEX_BODY}")


if __name__ == "__main__":
    main()
