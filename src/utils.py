import logging
import os
from pathlib import Path

import pandas as pd
import torch
from sentence_transformers import SentenceTransformer

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def init_notebook(depth=0):
    """Initialize working directory and logging for Jupyter notebooks."""
    top_level = os.getenv("REPO_ROOT", str(Path(os.getcwd()).parents[depth]))
    os.environ["REPO_ROOT"] = top_level
    os.chdir(top_level)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    logging.info(f"Working directory set to: {top_level}")


def load_diseases_for_chief_complaint(exp_args, chief_complaint: str) -> pd.DataFrame:
    chief_complaint_key = chief_complaint.replace(" ", "_")
    path = f'{exp_args.data_dir}/medical_schemes/diagnoses/{chief_complaint_key}.csv'
    return pd.read_csv(path)


def load_sbert_model(model_name='neuml/pubmedbert-base-embeddings'):
    """Load a Sentence-BERT model from HuggingFace or local directory (PVC)."""
    # Model for SentenceTransformer 2.2.2 is stored locally as the p100 config requires an old one
    try:
        local_path = f'/models/{model_name}'
        model = SentenceTransformer(model_name, device=detect_device())
        logging.info(f'Loading local model from {local_path}')
        return model
    except Exception as e:
        logging.info(f'Loading model {model_name} from HuggingFace')
        return SentenceTransformer(model_name, device=detect_device())


def convert_code_to_short_code(code: str, icd_version: int = 10) -> str:
    """Converts ICD codes to short format."""
    return code[:4] if icd_version == 9 and code.lower().startswith('e') else code[:3]


def convert_codes_to_short_codes(codes: list, icd_version: int = 10) -> list:
    """Batch conversion of codes to short format."""
    return list(set([convert_code_to_short_code(code, icd_version) for code in codes]))


def is_model_chat_based(model_name: str) -> bool:
    """Models containing these strings used chat instead if prompt template"""
    chat_based_strings = ["chat", "mistral", 'qwen', 'Qwen', 'lora_module', 'meditron3']
    return any(name in model_name.lower() for name in chat_based_strings)


def detect_device() -> str:
    # MPS (Apple Silicon) is skipped: sentence-transformers segfaults on large
    # encoding jobs due to MPS memory/driver instability.
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"
