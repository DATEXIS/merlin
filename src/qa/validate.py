"""Stage-2 validation: judge-vs-human agreement.

Turns the QA from "the LLM said so" into a measured instrument. Two sub-commands
via the driver:

  export  -- write qa_validation_template.csv: a random sample of classified cases
             (admission prompt, prediction, gold codes, and one empty column per
             class) for a human to label 0/1.
  score   -- read the human-filled CSV back, align on (hadm_id, model_name), and
             report per-class Cohen's kappa + agreement against the judge's calls.

Report the resulting kappa table alongside the error-shift table so a reviewer can
see the judge is a reliable annotator, not a black box.
"""
from __future__ import annotations

import glob
import os
from typing import List

import pandas as pd

from src.qa import data as qa_data
from src.qa.schemas import Taxonomy
from src.qa.classify import load_approved_taxonomy

PREF = qa_data.PRED_PREFIX


def _class_names(repo_root: str, qa_cfg: dict) -> List[str]:
    tax: Taxonomy = load_approved_taxonomy(repo_root, qa_cfg)
    return tax.names()


def export_template(cfg: dict, repo_root: str) -> None:
    qa_cfg = cfg["QA"]
    out_dir = os.path.join(repo_root, qa_cfg["output_dir"])
    names = _class_names(repo_root, qa_cfg)
    data_root = os.path.join(repo_root, qa_cfg.get("data_root", "data"))
    round_name = qa_cfg["round"]
    n_per_model = qa_cfg["validation"].get("sample_per_model", 25)
    seed = qa_cfg.get("seed", 42)

    frames = []
    for name in qa_cfg["stage2"]["models"]:
        cls_path = os.path.join(out_dir, f"qa_classifications_{name}.parquet")
        if not os.path.exists(cls_path):
            print(f"skip {name}: run stage 2 first")
            continue
        classified = pd.read_parquet(cls_path)
        classified = classified[classified["parsed"]]
        sample = classified.sample(n=min(n_per_model, len(classified)), random_state=seed)

        df = qa_data.load_eval_results(repo_root, data_root, round_name, name)
        merged = sample.merge(
            df[["hadm_id", f"{PREF}_prompt", f"{PREF}_text", f"{PREF}_preds", "ICD_CODES"]],
            on="hadm_id", how="left",
        )
        for _, r in merged.iterrows():
            row = {
                "model_name": name,
                "hadm_id": int(r["hadm_id"]),
                "gold_icd": ", ".join(map(str, r["ICD_CODES"])),
                "pred_icd": ", ".join(map(str, r[f"{PREF}_preds"])),
                "admission_prompt": str(r[f"{PREF}_prompt"])[:4000],
                "model_reasoning": str(r[f"{PREF}_text"])[:4000],
            }
            for c in names:
                row[f"human__{c}"] = ""  # human fills 0/1
            frames.append(row)

    template = pd.DataFrame(frames)
    path = os.path.join(out_dir, "qa_validation_template.csv")
    template.to_csv(path, index=False)
    print(f"Wrote {len(template)} rows to {path}. Fill human__* columns with 0/1, then run "
          "`qa_analysis.py validate --score`.")


def _cohen_kappa(a: pd.Series, b: pd.Series) -> float:
    """Cohen's kappa for two binary label series (no sklearn dependency)."""
    a = a.astype(int).to_numpy()
    b = b.astype(int).to_numpy()
    n = len(a)
    if n == 0:
        return float("nan")
    po = (a == b).mean()
    pa1, pb1 = a.mean(), b.mean()
    pe = pa1 * pb1 + (1 - pa1) * (1 - pb1)
    return 1.0 if pe == 1.0 else (po - pe) / (1 - pe)


def score_agreement(cfg: dict, repo_root: str) -> None:
    qa_cfg = cfg["QA"]
    out_dir = os.path.join(repo_root, qa_cfg["output_dir"])
    names = _class_names(repo_root, qa_cfg)
    tmpl_path = os.path.join(out_dir, "qa_validation_template.csv")
    human = pd.read_csv(tmpl_path)

    judge_frames = []
    for name in qa_cfg["stage2"]["models"]:
        p = os.path.join(out_dir, f"qa_classifications_{name}.parquet")
        if os.path.exists(p):
            judge_frames.append(pd.read_parquet(p))
    judge = pd.concat(judge_frames, ignore_index=True)

    merged = human.merge(
        judge, on=["hadm_id", "model_name"], how="inner", suffixes=("", "_judge")
    )
    rows = []
    for c in names:
        hcol = f"human__{c}"
        if hcol not in merged.columns:
            continue
        sub = merged[[hcol, c]].dropna()
        sub = sub[sub[hcol].astype(str).str.strip() != ""]
        if sub.empty:
            continue
        h = sub[hcol].astype(float).round().astype(int)
        j = sub[c].astype(int)
        rows.append({
            "error_class": c,
            "n": len(sub),
            "agreement": round((h.to_numpy() == j.to_numpy()).mean(), 3),
            "cohen_kappa": round(_cohen_kappa(h, j), 3),
            "human_pos": int(h.sum()),
            "judge_pos": int(j.sum()),
        })
    report = pd.DataFrame(rows)
    path = os.path.join(out_dir, "qa_validation_agreement.csv")
    report.to_csv(path, index=False)
    print(report.to_string(index=False))
    print(f"\nWrote {path}")
