"""WandB logging for the QA stages (project `merlin_qa`, like the old run_analyses).

Local files under data/qa/ stay the source of truth; wandb gets a mirrored copy in
a browsable format: preview wandb.Tables for eyeballing plus full artifacts for
download. Everything is behind QA.wandb so runs work offline too.
"""
from __future__ import annotations

from typing import List, Optional

import pandas as pd


def start(qa_cfg: dict, run_name: str):
    if not qa_cfg.get("wandb", True):
        return None
    from src.wandb.run import init_wandb
    return init_wandb(run_name, qa=True)


def log_table(run, key: str, df: pd.DataFrame, preview_rows: int = 200) -> None:
    if run is None or df.empty:
        return
    import wandb
    preview = df.head(preview_rows).astype(str)
    run.log({key: wandb.Table(dataframe=preview)})


def log_artifact_df(run, df: pd.DataFrame, name: str) -> None:
    if run is None or df.empty:
        return
    from src.wandb.data_loader import upload_df_to_wandb
    upload_df_to_wandb(run, df.astype(str), name, "dataset")


def log_summary(run, metrics: dict) -> None:
    if run is None:
        return
    run.summary.update(metrics)


def finish(run) -> None:
    if run is not None:
        run.finish()


def rca_records_to_df(rca_records: List[dict]) -> pd.DataFrame:
    """Flatten stage-1 RCAs to one row per error for table logging."""
    rows = []
    for rec in rca_records:
        for e in rec.get("errors", []):
            rows.append({
                "model_name": rec["model_name"],
                "subject_id": rec["subject_id"],
                "hadm_id": rec["hadm_id"],
                "case_summary": rec.get("case_summary", ""),
                "kind": e.get("kind", ""),
                "keyword": e.get("keyword", ""),
                "icd_codes": ", ".join(e.get("icd_codes", [])),
                "root_cause": e.get("root_cause", ""),
                "inferable_from_admission": e.get("inferable_from_admission"),
            })
    return pd.DataFrame(rows)


def taxonomy_to_df(classes: List[dict]) -> pd.DataFrame:
    return pd.DataFrame(classes)
