import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer, util

from src.pipeline.json_extraction import convert_symptom_dict_to_symptom_vector, DiagnosesModel, ICDsModel
from src.pipeline.verifier_args import VerifierArgs
from src.utils import convert_codes_to_short_codes

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def cosine_similarity_torch(prediction, label):
    prediction = torch.tensor(prediction, dtype=torch.float32, device=DEVICE)
    label = torch.tensor(label, dtype=torch.float32, device=DEVICE)

    pred_norm = torch.norm(prediction)
    label_norm = torch.norm(label)

    if pred_norm == 0 or label_norm == 0:
        return 0.0  # or define a fallback, e.g. 0.5

    cos_sim = torch.dot(prediction, label) / (pred_norm * label_norm)
    return round(cos_sim.item(), 3)


def normalized_dot_product_torch(prediction, label):
    prediction = torch.tensor(prediction, dtype=torch.float32, device=DEVICE)
    label = torch.tensor(label, dtype=torch.float32, device=DEVICE)

    if prediction.shape != label.shape:
        return 0.0

    dot = torch.dot(prediction, label)
    k = (label != 0).sum()

    if k == 0:
        return 0.5

    return round(((dot + k) / (2 * k)).item(), 3)


def get_mrr_score(pred_set: list[str], true_label: str) -> float:
    """Compute MRR where pred_set is a ranked list and true_label is a string."""
    if true_label not in pred_set:
        return 0.0
    return 1.0 / (pred_set.index(true_label) + 1)


def get_icd_f1_score(pred: list, true: list) -> float:
    intersection = set(pred) & set(true)
    precision = len(intersection) / len(pred) if pred else 0.0
    recall = len(intersection) / len(true) if true else 0.0
    return round(2 * precision * recall / (precision + recall) if (precision + recall) else 0.0, 3)


def extract_and_map_diagnose_name(model, diagnose_json: list[dict], labels: list[str],
                                  label_embeddings) -> (list[str] | None):
    if diagnose_json is None:
        return None

    pred_names = []
    for pred_dict in diagnose_json:
        pred = pred_dict['name']
        if pred in labels:
            pred_names.append(pred)
            continue

        pred_embedding = model.encode(pred, convert_to_tensor=True,
                                      normalize_embeddings=True,
                                      show_progress_bar=False)
        cosine_scores = util.cos_sim(pred_embedding, label_embeddings)
        best_idx = torch.argmax(cosine_scores).item()
        matched_name = labels[best_idx]
        pred_names.append(matched_name)

    return pred_names if pred_names else None


def extract_diagnose_names_from_choices(
        v_args: VerifierArgs,
        patients: pd.DataFrame,
        choices: list[list[list[dict]] | None],
        model: SentenceTransformer
) -> list[list[list[str]] | None]:
    """Extracts name from dict name/reason format.
    Maps predicted name to a potential diagnosis based on the primary complaint."""
    work_df = patients.copy()
    work_df['choices'] = choices
    matched_names = []

    # Iterate over all primary complaints
    for primary_complaint, potential_diagnoses in v_args.potential_diagnoses.items():
        sub_df = work_df[work_df['Chief Complaint'] == primary_complaint]
        label_embeddings = model.encode(
            potential_diagnoses,
            convert_to_tensor=True,
            normalize_embeddings=True,
            show_progress_bar=False
        )
        for choices in sub_df['choices'].tolist():
            if choices is not None:
                matched_names.append(
                    [extract_and_map_diagnose_name(model, choice, potential_diagnoses, label_embeddings)
                     for choice in choices]
                )
            else:
                matched_names.append([])

    return matched_names if matched_names else None


def extract_icd_names(icd_dicts: list, potential_codes: set) -> list:
    return [
        [
            [
                code
                for code in convert_codes_to_short_codes(icd["icd_code"] for icd in choice)
                if code in potential_codes
            ] if choice else None
            for choice in choices
        ] if choices else None
        for choices in icd_dicts
    ]


def handle_manifestations(v_args, work_df, extracted_choices, _):
    predictions = convert_symptom_dict_to_symptom_vector(extracted_choices)
    labels = work_df["disease_vector"].tolist()
    return predictions, labels, normalized_dot_product_torch


def handle_diagnoses(v_args, work_df, extracted_choices, mapping_model):
    predictions = extract_diagnose_names_from_choices(
        v_args, work_df, extracted_choices, mapping_model
    )
    labels = work_df["disease"].tolist()
    return predictions, labels, get_mrr_score


def handle_icds(v_args: VerifierArgs, work_df: pd.DataFrame, extracted_choices, _):
    potential_codes = set(v_args.potential_icds)
    predictions = extract_icd_names(extracted_choices, potential_codes)
    labels = [convert_codes_to_short_codes(codes) for codes in work_df["ICD_CODES"].tolist()]
    return predictions, labels, get_icd_f1_score


def calculate_scores(
        v_args: VerifierArgs,
        work_df: pd.DataFrame,
        extracted_choices: list[list[list[dict]]],
        mapping_model,
) -> tuple[list[list[float]], list]:
    """Calculate scores for the given verifier step."""
    scheme = v_args.get_pydantic_scheme()

    scheme_handlers = {
        "ManifestationsModel": handle_manifestations,
        DiagnosesModel: handle_diagnoses,
        ICDsModel: handle_icds,
    }

    handler = scheme_handlers.get(scheme.__name__) or scheme_handlers.get(scheme)
    if handler is None:
        raise NotImplementedError(f"Unknown scheme type: {scheme}")

    predictions, labels, verifier = handler(
        v_args, work_df, extracted_choices, mapping_model
    )
    scores = [
        [verifier(pred, label) if pred not in (None, 'None') else np.nan
         for pred in choices]
        if choices is not None else [np.nan]
        for choices, label in zip(predictions, labels)
    ]

    return scores, predictions

