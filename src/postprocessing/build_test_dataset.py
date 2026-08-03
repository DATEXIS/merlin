from pathlib import Path

import pandas as pd

# Columns the downstream eval pipeline actually needs (src/pipeline client in
# eval_mode) plus enough patient context to be useful standalone. 'split' /
# 'Chief Complaint' / 'ICD_CODES' are required by the client; the rest mirrors
# what the ad-hoc eval.ipynb build used before this was automated.
TEST_DATASET_COLUMNS = [
    "subject_id", "hadm_id", "split", "admission_note",
    "Chief Complaint", "disease", "disease_vector", "ICD_CODES", "labs",
]


def build_test_dataset(combined_path: str, output_path: str) -> pd.DataFrame:
    """
    Build the downstream-eval test_dataset from combined.pq: one row per
    patient, train split excluded.

    combined.pq has ~3 rows per patient (one per teacher model, from
    combine_models); dedup on hadm_id (same convention as
    build_mimic_instructions.build_mimic_instructions) since every
    patient-level column here (admission_note, ICD_CODES, labs, ...) is
    identical across the model rows — only the v1-v4 generation columns
    (excluded here) differ.

    'val' is already folded into 'dev' upstream by combine_models, so the
    only splits present are 'dev' and 'test'. This is what scripts/
    the dataset-upload step uploads as the 'test_dataset' wandb artifact,
    and what *_config.yaml's eval-mode client loads via file_name:
    "test_dataset".

    Parameters
    ----------
    combined_path : str
        Path to combined.pq (output of merge_gen_data.combine_models).
    output_path : str
        Destination path for the eval parquet file.
    """
    df = pd.read_parquet(combined_path, columns=TEST_DATASET_COLUMNS)

    no_train = df[df["split"] != "train"]
    deduped = no_train.drop_duplicates(subset="hadm_id", keep="first")

    out = deduped[TEST_DATASET_COLUMNS].reset_index(drop=True)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(output_path, index=False)

    from collections import Counter
    print(f"  test_dataset: {len(out):,} patients -> {output_path} "
          f"| splits: {dict(Counter(out['split']))}")
    return out
