"""Loading and joining data for the QA stages.

Eval result parquets live at
    data/results/evaluation/{round}/eval_results_{name}/eval_results_{name}.pq
and carry the prediction columns (v4_prompt, v4_text, v4_preds, v4_score, ...),
the gold ICD_CODES, and admission metadata -- but NOT the discharge note.

Discharge notes are joined on hadm_id from the MIMIC ICD-10 hosp splits. The eval
dev/test hadm_ids are distributed across the discharge-note train/dev/test files
(the two pipelines split independently), so we union all three ICD-10 hosp files
and de-duplicate on hadm_id to get full coverage.
"""
from __future__ import annotations

import glob
import os
from typing import List, Optional

import pandas as pd

# Prediction "version" the QA analyses -- v4 is the final verifier stage.
PRED_PREFIX = "v4"

_DISCHARGE_GLOB = "mimic-iv/discharge_notes/icd-10/hosp/*.pq"


def eval_results_path(data_root: str, round_name: str, name: str) -> str:
    return os.path.join(
        data_root, "results", "evaluation", round_name,
        f"eval_results_{name}", f"eval_results_{name}.pq",
    )


def load_discharge_notes(data_root: str) -> pd.DataFrame:
    """Union of ICD-10 hosp discharge notes, one row per hadm_id.

    Resolved under data_root (not the repo) so the same code runs on a laptop
    (data_root=<repo>/data) and in-cluster (data_root=/merlin_ddx/...).
    """
    pattern = os.path.join(data_root, _DISCHARGE_GLOB)
    paths = glob.glob(pattern)
    if not paths:
        raise FileNotFoundError(f"No discharge notes under {pattern}")
    frames = [pd.read_parquet(p, columns=["hadm_id", "text"]) for p in paths]
    notes = pd.concat(frames, ignore_index=True).drop_duplicates("hadm_id")
    return notes.rename(columns={"text": "discharge_note"})


def load_eval_results(repo_root: str, data_root: str, round_name: str, name: str) -> pd.DataFrame:
    """Load one eval result set and attach discharge notes.

    Adds a `model_name` column and drops rows without a discharge note (can't do
    RCA without ground truth). Prints coverage so silent join losses are visible.
    """
    path = eval_results_path(data_root, round_name, name)
    if not os.path.isabs(path):
        path = os.path.join(repo_root, path)
    df = pd.read_parquet(path)
    df["model_name"] = name

    notes = load_discharge_notes(data_root if os.path.isabs(data_root)
                                 else os.path.join(repo_root, data_root))
    merged = df.merge(notes, on="hadm_id", how="left")
    covered = merged["discharge_note"].notna().mean()
    print(f"[{name}] {len(df)} cases, discharge-note coverage {covered:.1%}")
    merged = merged[merged["discharge_note"].notna()].reset_index(drop=True)
    return merged


def _as_set(codes) -> set:
    if codes is None:
        return set()
    try:
        return {str(c).strip() for c in codes if str(c).strip()}
    except TypeError:
        return set()


def three_char(codes) -> set:
    """ICD-10 codes truncated to the 3-char category level, matching eval scoring."""
    return {c[:3] for c in _as_set(codes)}


def mispredicted_mask(df: pd.DataFrame) -> pd.Series:
    """True where gold and predicted 3-char categories disagree at all.

    Perfect cases carry no error signal, so stages 1 and 2 skip them by default.
    """
    gold = df["ICD_CODES"].map(three_char)
    pred = df[f"{PRED_PREFIX}_preds"].map(three_char)
    return [g != p for g, p in zip(gold, pred)]


def select_cases(
    df: pd.DataFrame,
    only_mispredicted: bool = True,
    sample: Optional[int] = None,
    seed: int = 42,
) -> pd.DataFrame:
    out = df
    if only_mispredicted:
        out = out[mispredicted_mask(out)].reset_index(drop=True)
    if sample is not None and sample < len(out):
        out = out.sample(n=sample, random_state=seed).reset_index(drop=True)
    return out


def required_columns() -> List[str]:
    return [
        f"{PRED_PREFIX}_prompt", f"{PRED_PREFIX}_text", f"{PRED_PREFIX}_preds",
        f"{PRED_PREFIX}_score", "ICD_CODES", "discharge_note",
        "subject_id", "hadm_id",
    ]
