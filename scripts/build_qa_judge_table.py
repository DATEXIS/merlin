"""Build the per-case table an LLM judge fills in for the qualitative QA classes.

One row per test admission, holding the notes + gold codes once and every
model's predictions side by side, plus an EMPTY boolean column per (model x judge
class) for the judge (Sonnet) to populate. The coverage classes
(missed_history / missed_medication / missed_chronic) are computed
deterministically elsewhere and are NOT judged here -- see data/qa/judge_instruction.md.

Wide layout on purpose: the discharge note (the large field) is stored once per
case instead of duplicated per model. Melt to long later if the judge harness
prefers one row per (case, model).

Run:
    python scripts/build_qa_judge_table.py
Output: data/qa/judge_table.parquet  (+ a .csv preview of the first rows)
"""
from __future__ import annotations

import ast
import os
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.qa import data as qa_data  # noqa: E402

ROUND = "1_4"
DATA_ROOT = "data"

# size -> {variant: eval_results name}. Both base and full at each size.
MODELS = {
    "06b": {"base": "test-06b-base", "full": "test-06b-full-e4"},
    "8b":  {"base": "test-8b-base",  "full": "test-8b-full-e3"},
    "14b": {"base": "test-14b-base", "full": "test-14b-full-e3"},
    "32b": {"base": "test-32b-base", "full": "test-32b-full-e3"},
}

# Classes the judge decides (need the notes). Coverage classes are deterministic.
JUDGE_CLASSES = [
    "missed_primary_diagnosis",
    "unspecific_primary_diagnosis",
    "not_inferable_from_admission",
    "redundancy_symptom_misuse",
]

# Case-identifying / context columns taken from any one model's result file.
BASE_COLS = ["subject_id", "hadm_id", "Chief Complaint", "admission_note", "ICD_CODES"]


def preds_3char(v) -> list:
    """v4_preds is a stringified list of already-3char categories; parse -> list."""
    if v is None:
        return []
    try:
        lst = ast.literal_eval(v) if isinstance(v, str) else list(v)
    except (ValueError, SyntaxError):
        return []
    return sorted({str(c).strip()[:3] for c in lst if str(c).strip()})


def load_one(name: str) -> pd.DataFrame:
    path = qa_data.eval_results_path(DATA_ROOT, ROUND, name)
    if not os.path.isabs(path):
        path = os.path.join(str(REPO_ROOT), path)
    return pd.read_parquet(path)


def main() -> None:
    out_dir = os.path.join(str(REPO_ROOT), DATA_ROOT, "qa")
    os.makedirs(out_dir, exist_ok=True)

    # Anchor case set + context from the first model's file (all share hadm_ids).
    first = next(iter(next(iter(MODELS.values())).values()))
    base = load_one(first)[BASE_COLS].copy()
    base = base.rename(columns={"ICD_CODES": "gold_codes"})
    base["gold_codes"] = base["gold_codes"].map(lambda a: list(a) if a is not None else [])
    base["gold_3char"] = base["gold_codes"].map(
        lambda cs: sorted({str(c).strip()[:3] for c in cs if str(c).strip()}))

    # Attach discharge notes (union of icd-10 hosp splits, one row per hadm_id).
    notes = qa_data.load_discharge_notes(os.path.join(str(REPO_ROOT), DATA_ROOT))
    df = base.merge(notes, on="hadm_id", how="left")
    cov = df["discharge_note"].notna().mean()
    print(f"discharge-note coverage: {cov:.1%} ({df['discharge_note'].isna().sum()} missing)")

    # One prediction column per (size, variant), aligned on hadm_id.
    pred_cols = []
    for size, variants in MODELS.items():
        for variant, name in variants.items():
            col = f"{size}_{variant}_preds"
            m = load_one(name)[["hadm_id", "v4_preds"]].copy()
            m[col] = m["v4_preds"].map(preds_3char)
            df = df.merge(m[["hadm_id", col]], on="hadm_id", how="left")
            pred_cols.append(col)
            print(f"  [{col}] median #preds = "
                  f"{df[col].map(len).median():.0f}")

    # Empty nullable-boolean judge columns: (size, variant) x judge class.
    err_cols = []
    for size, variants in MODELS.items():
        for variant in variants:
            for cls in JUDGE_CLASSES:
                c = f"{size}_{variant}__{cls}"
                df[c] = pd.Series([pd.NA] * len(df), dtype="boolean")
                err_cols.append(c)

    ordered = (["subject_id", "hadm_id", "Chief Complaint",
                "admission_note", "discharge_note", "gold_codes", "gold_3char"]
               + pred_cols + err_cols)
    df = df[ordered]

    out_pq = os.path.join(out_dir, "judge_table.parquet")
    df.to_parquet(out_pq, index=False)
    df.head(20).drop(columns=["admission_note", "discharge_note"]).to_csv(
        os.path.join(out_dir, "judge_table_preview.csv"), index=False)

    print(f"\nrows: {len(df)}  pred cols: {len(pred_cols)}  judge cols: {len(err_cols)}")
    print(f"-> {out_pq}")


if __name__ == "__main__":
    main()
