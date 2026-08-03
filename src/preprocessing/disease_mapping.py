import logging
import re
from typing import List, Optional

import numpy as np
import pandas as pd
from sentence_transformers import util

from src.preprocessing.diagnosis_patterns import END_SECTION_FULL_LINES, END_SECTION_PREFIXES, \
    EQUALS_ONLY_RE, MINUS_ONLY_RE, HEADING_LINES, TRIM_PREFIX_RE, PREV_HDR, NEXT_HDR, TARGET_HDR
from src.utils import load_sbert_model


def _clean_primary_diagnosis_lines(lines: List[str]) -> List[str]:
    """
    Run a single cleaning pass over extracted diagnosis lines.
    """
    cleaned: List[str] = []

    for raw in lines:
        stripped = raw.strip()
        upper = stripped.upper()

        # ── early termination
        if (upper in END_SECTION_FULL_LINES
                or any(upper.startswith(p) for p in END_SECTION_PREFIXES)):
            break

        # ── decorative / heading noise
        if (EQUALS_ONLY_RE.fullmatch(stripped)
                or MINUS_ONLY_RE.fullmatch(stripped)
                or upper in HEADING_LINES):
            continue

        # ── trim known prefixes
        m = TRIM_PREFIX_RE.match(stripped)
        if m:
            remainder = stripped[m.end():].lstrip()
            if remainder:
                cleaned.append(remainder)
            continue

        # ── untouched line
        cleaned.append(stripped)

    return cleaned


def extract_primary_discharge_diagnoses(note: str) -> Optional[str]:
    tail = re.split(PREV_HDR, note, flags=re.I | re.S)[-1]

    m = re.search(rf'{TARGET_HDR}\s*(.*?)\s*(?={NEXT_HDR}|$)', tail,
                  flags=re.I | re.S)
    if not m:
        return None

    lines = m.group(1).splitlines()

    while True:
        cleaned = _clean_primary_diagnosis_lines(lines)
        if cleaned == lines:
            break
        lines = cleaned

    result = "\n".join(lines).lstrip(".-):+=").rstrip(":").strip()
    return result or None


def add_primary_diagnosis_column(df: pd.DataFrame) -> None:
    """Extract primary diagnosis string from discharge note and add as primary_diagnosis column."""
    df['diagnose_string'] = df['discharge_note'].apply(extract_primary_discharge_diagnoses)
    logging.info('Added diagnose_string column to DataFrame.')


def add_disease_columns(mimic_df: pd.DataFrame, wikidoc_df: pd.DataFrame):
    """Map primary diagnosis strings to wikidoc diseases using SBERT cosine similarity."""
    model = load_sbert_model()
    diagnose_strings = mimic_df["diagnose_string"].fillna("").tolist()
    labels = wikidoc_df['Disease'].tolist()

    encode_kwargs = dict(batch_size=16, normalize_embeddings=True, show_progress_bar=True)
    label_embeddings = model.encode(labels, **encode_kwargs)
    diagnose_embeddings = model.encode(diagnose_strings, **encode_kwargs)

    # Similarity matrix
    sim_mat = util.cos_sim(diagnose_embeddings, label_embeddings).cpu().numpy()

    # Turn each row into a ranking
    sort_idx = np.argsort(-sim_mat, axis=1)                     # shape (N, P)

    # helper to map sorted indices → names / scores
    def row_rank(i):
        idx = sort_idx[i]
        return [labels[j] for j in idx], [float(sim_mat[i, j]) for j in idx]  # cast to plain float

    ranked_names, ranked_scores = zip(*(row_rank(i) for i in range(sim_mat.shape[0])))

    mimic_df["diseases"] = ranked_names     # list[str] per row
    mimic_df["diseases_cossim_scores"] = ranked_scores    # list[float] per row
    mimic_df["disease"] = mimic_df["diseases"].apply(lambda x: x[0])
    logging.info('Added wikidoc columns diseases, diseases_cossim_scores, disease to DataFrame.')


def add_disease_symptom_column(mimic_df: pd.DataFrame, wikidoc_df: pd.DataFrame) -> None:
    disease_df = wikidoc_df.drop(['Synonyms', 'icd10_regex'], axis=1, errors='ignore')
    disease_df.set_index('Disease', inplace=True)
    mimic_df['disease_vector'] = [disease_df.loc[disease].values for disease in mimic_df['disease']]
    logging.info('Added wikidoc column disease_vector to DataFrame.')


def add_wikidoc_columns(mimic_df: pd.DataFrame, wikidoc_df: pd.DataFrame) -> None:
    """Add wikidoc disease and disease_vector columns based on primary diagnoses to DataFrame."""
    logging.info('Mapping primary diagnoses to wikidoc diseases...')
    add_disease_columns(mimic_df, wikidoc_df)
    add_disease_symptom_column(mimic_df, wikidoc_df)
