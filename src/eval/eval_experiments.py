import ast

import numpy as np
import pandas as pd

from src.eval.classification_metrics import calculate_icd_metrics_cpu, \
    calculate_disease_metrics, calculate_batch_symptom_extraction_metrics
from src.utils import detect_device, convert_codes_to_short_codes

ROUND_DIGITS = 4


def safe_parse_list(value) -> list:
    """Parse a cell into a list, tolerating None/NaN/'None'/empty/malformed values."""
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


def extract_icd_names(v4_json: list, potential_icds: set = None) -> list:
    if v4_json == 'None' or not v4_json:
        return []

    codes = convert_codes_to_short_codes([codes['icd_code'] for codes in v4_json])
    if potential_icds:
        return [code for code in codes if code in potential_icds]
    else:
        return codes

def calculate_json_validity_metrics(results: pd.DataFrame, verifier_steps: tuple = (1, 2, 4)) -> dict:
    """Fraction of rows per verifier step whose extracted JSON is valid (non-null, non-'None'),
    i.e. could actually be parsed into a prediction and thus be evaluated.

    Uses the same "invalid" check as `filter_success_candidates` in the pipeline
    (pd.isna(x) or x == 'None'), so the rate is 1 - (failure rate used there).
    """
    metrics = {}
    for v in verifier_steps:
        col = f'v{v}_json'
        if col not in results.columns or len(results) == 0:
            continue
        is_valid = results[col].apply(lambda x: not (pd.isna(x) or x == "None"))
        metrics[f'V{v} JSON Valid Rate'] = round(is_valid.mean(), ROUND_DIGITS)
    return metrics


def get_v1_tensors(results: pd.DataFrame):
    # 1. Parse strings to lists (v1_preds are often stored as strings in CSVs)
    preds_raw = results['v1_preds'].apply(
        lambda x: ast.literal_eval(x) if isinstance(x, str) else x).to_list()
    labels = results['disease_vector'].to_list()

    clean_preds = []

    for p, l in zip(preds_raw, labels):
        if p is None or (isinstance(p, float) and np.isnan(p)):
            # Fill with zero vector of the same length as the ground truth label
            clean_preds.append([0] * len(l))
        else:
            clean_preds.append(p)

    # 2. Group by length to ensure we can create valid numpy arrays
    # (Since you have 7 different complaint lengths)
    return clean_preds, labels


def evaluate_experiment(results: pd.DataFrame) -> dict:
    device = detect_device()
    print(device)

    v1_pred, v1_label = get_v1_tensors(results)

    v2_pred = results['v2_preds'].apply(safe_parse_list).to_list()
    v2_label = results['disease'].to_list()

    v4_pred = results['v4_preds'].apply(safe_parse_list).tolist()
    v4_label = results['ICD_CODES'].apply(convert_codes_to_short_codes).tolist()

    return {
        **calculate_icd_metrics_cpu(v4_pred, v4_label),
        **calculate_disease_metrics(v2_pred, v2_label, device),
        **calculate_batch_symptom_extraction_metrics(v1_pred, v1_label, device),
        **calculate_json_validity_metrics(results),
        'ICD len': round(np.mean([len(codes) for codes in v4_pred]), 2) / 100,
        'ICD label len': round(np.mean([len(codes) for codes in v4_label]), 2) / 100,
    }


# NOTE: the old data/sft_results-era helpers (get_metrics_df,
# calculate_chief_complaint_mrr, get_robustness_metrics) were removed
# Recover from git history if a per-chief-complaint or multi-seed robustness
# breakdown is needed for the paper.
