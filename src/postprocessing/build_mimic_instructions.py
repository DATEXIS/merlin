import json
from collections import Counter
from pathlib import Path

import pandas as pd
import yaml

from src.pipeline.prompts import ICD_MIMIC_PROMPT, DIAGNOSE_MIMIC_PROMPT, RERANK_MIMIC_PROMPT, \
    EXTRACT_PROMPT_DICT

MEDICAL_SCHEMES_DIR = Path("data/medical_schemes")

ICD_FORMAT_EXAMPLE = '''[
    {
        "icd_code": "icd_code_0",
        "reason": "Based on clinical evidence, icd_code_0 is suspected."
    },
    {
        "icd_code": "icd_code_1",
        "reason": "Based on clinical evidence, icd_code_1 is suspected."
    },
    ...
]'''

DIAGNOSE_FORMAT_EXAMPLE = '''[
    {
        "name": "diagnosis_0",
        "reason": "Based on clinical evidence, diagnosis_0 is suspected."
    },
    {
        "name": "diagnosis_1",
        "reason": "Based on clinical evidence, diagnosis_1 is suspected."
    },
    ...
]'''

OUTPUT_COLUMNS = ["hadm_id", "subject_id", "split", "chief_complaint", "Verifier", "Input", "Output"]


def _labs_str(labs) -> str:
    return "" if (labs is None or str(labs) == "None") else str(labs)


# ── V4: ICD prediction ──────────────────────────────────────────────────────


def _build_icd_output(icd_codes) -> str:
    codes = list(icd_codes) if hasattr(icd_codes, "tolist") else icd_codes
    return json.dumps([
        {"icd_code": str(code), "reason": f"Based on the admission note {code} is suspected."}
        for code in codes
    ])


def _build_icd_input(admission_note: str, labs) -> str:
    return ICD_MIMIC_PROMPT.format(
        admission_note=admission_note,
        laboratory_results=_labs_str(labs),
        icd_format_example=ICD_FORMAT_EXAMPLE,
    )


# ── V1: symptom extraction ──────────────────────────────────────────────────
# Ground truth comes from the `disease_vector` column, a positional array that
# must line up with the CC's manifestation yaml key order. Verified against
# real v1_json output in combined.pq (same key order) -- see conversation/
# investigation notes; fails loud below if the lengths ever disagree.

_SYMPTOM_SCHEMES: dict[str, dict] = {}


def _load_symptom_scheme(cc_key: str) -> dict:
    if cc_key not in _SYMPTOM_SCHEMES:
        path = MEDICAL_SCHEMES_DIR / "symptoms" / f"{cc_key}.yaml"
        with open(path) as f:
            _SYMPTOM_SCHEMES[cc_key] = yaml.safe_load(f)
    return _SYMPTOM_SCHEMES[cc_key]


def _symptom_format_example(scheme: dict) -> str:
    return json.dumps({
        name: {
            "value": 0,
            "reasoning": f"Based on the admission note, {name} is suspected to be present / "
                         f"absent / unmentioned.",
        }
        for name in scheme
    }, indent=4)


def _build_symptom_output(scheme: dict, disease_vector) -> str:
    names = list(scheme.keys())
    if len(names) != len(disease_vector):
        raise ValueError(
            f"disease_vector length {len(disease_vector)} != symptom scheme length "
            f"{len(names)} for scheme keys {names}"
        )
    # NOTE: key is "reasoning" (matches the schema requested in the prompt and
    # real v1_json output) -- the old dead src/pipeline/instruction_builder.py
    # used "reason" here, a mismatch with its own format example. Fixed here.
    return json.dumps({
        name: {"value": int(value), "reasoning": f"Extracted {name} from the admission note."}
        for name, value in zip(names, disease_vector)
    }, indent=4)


def _build_symptom_input(cc_key: str, admission_note: str, scheme: dict) -> str:
    template = EXTRACT_PROMPT_DICT.get(cc_key, EXTRACT_PROMPT_DICT["generic"])
    return template.format(
        admission_note=admission_note,
        symptom_format_example=_symptom_format_example(scheme),
    )


# ── V2/V3: diagnosis ranking + lab-informed reranking ───────────────────────
# Ground truth is the single `disease` label. "Potential Diagnoses" is the
# CC's static candidate label space (data/medical_schemes/diagnoses/<cc>.csv),
# NOT a prior stage's prediction -- these MIMIC-style prompts skip the
# reasoning-chain scaffolding entirely (never actually run through generation).

_DIAGNOSIS_LABELSPACE: dict[str, list[str]] = {}


def _load_diagnosis_labelspace(cc_key: str) -> list[str]:
    if cc_key not in _DIAGNOSIS_LABELSPACE:
        path = MEDICAL_SCHEMES_DIR / "diagnoses" / f"{cc_key}.csv"
        _DIAGNOSIS_LABELSPACE[cc_key] = pd.read_csv(path)["Disease"].tolist()
    return _DIAGNOSIS_LABELSPACE[cc_key]


def _potential_diagnoses_str(diagnoses: list[str]) -> str:
    return "\n".join(f"{i + 1}. {name}" for i, name in enumerate(diagnoses))


def _build_diagnose_output(disease: str) -> str:
    return json.dumps([{"name": disease, "reason": f"Based on the admission note {disease} is suspected."}])


def _build_diagnose_input(admission_note: str, diagnoses_str: str) -> str:
    return DIAGNOSE_MIMIC_PROMPT.format(
        admission_note=admission_note,
        potential_diagnoses=diagnoses_str,
        diagnose_format_example=DIAGNOSE_FORMAT_EXAMPLE,
    )


def _build_rerank_input(admission_note: str, labs, diagnoses_str: str) -> str:
    return RERANK_MIMIC_PROMPT.format(
        admission_note=admission_note,
        laboratory_results=_labs_str(labs),
        potential_diagnoses=diagnoses_str,
        diagnose_format_example=DIAGNOSE_FORMAT_EXAMPLE,
    )


# ── Row assembly ─────────────────────────────────────────────────────────────


def _build_patient_rows(row, duplicate_verifiers: tuple[int, ...]) -> list[dict]:
    cc_key = str(row["Chief Complaint"]).replace(" ", "_")
    admission_note = row["admission_note"]
    labs = row.get("labs")

    scheme = _load_symptom_scheme(cc_key)
    diagnoses_str = _potential_diagnoses_str(_load_diagnosis_labelspace(cc_key))

    stage_io = {
        1: (_build_symptom_input(cc_key, admission_note, scheme),
            _build_symptom_output(scheme, row["disease_vector"])),
        2: (_build_diagnose_input(admission_note, diagnoses_str),
            _build_diagnose_output(row["disease"])),
        3: (_build_rerank_input(admission_note, labs, diagnoses_str),
            _build_diagnose_output(row["disease"])),
        4: (_build_icd_input(admission_note, labs),
            _build_icd_output(row["ICD_CODES"])),
    }

    base = dict(hadm_id=row["hadm_id"], subject_id=row["subject_id"], split=row["split"],
                chief_complaint=cc_key)

    verifier_steps = list(range(1, 5)) + list(duplicate_verifiers)
    return [
        {**base, "Verifier": v, "Input": stage_io[v][0], "Output": stage_io[v][1]}
        for v in verifier_steps
    ]


def build_mimic_instructions(
    combined_path: str,
    output_path: str,
    duplicate_verifiers: tuple[int, ...] = (),
) -> pd.DataFrame:
    """
    Build the full-stage MIMIC instruction dataset: one row per (patient x
    verifier step), V1-V4, covering the SAME patients as MeRLIn -- but with
    MIMIC-style DIRECT prompts (EXTRACT/DIAGNOSE_MIMIC/RERANK_MIMIC/ICD_MIMIC)
    that skip the reasoning-chain scaffolding (no conditioning on a prior
    stage's prediction), and pure ground-truth outputs with no reasoning trace
    (no `Thinking` column at all -> the SFT formatter renders a bare answer,
    same convention as the old single-stage version). This is the
    reasoning-trace ablation arm, now spanning all four stages instead of just
    V4/ICD (see build_mimic_instructions_icd_only for the legacy single-stage
    version this replaces).

    Parameters
    ----------
    combined_path : str
        Path to combined.pq (output of merge_gen_data.combine_models).
    output_path : str
        Destination path for the instruction parquet file.
    duplicate_verifiers : tuple[int, ...]
        Verifier steps to emit twice per patient (e.g. (4,) for icd_2x-style
        oversampling of the ICD-prediction step).
    """
    cols = ["hadm_id", "subject_id", "split", "Chief Complaint", "admission_note",
            "ICD_CODES", "labs", "disease", "disease_vector"]
    df = pd.read_parquet(combined_path, columns=cols)
    df = df.drop_duplicates(subset="hadm_id", keep="first")

    rows: list[dict] = []
    for _, row in df.iterrows():
        rows.extend(_build_patient_rows(row, duplicate_verifiers))

    out = pd.DataFrame(rows, columns=OUTPUT_COLUMNS)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(output_path, index=False)
    stages = len(duplicate_verifiers) + 4
    print(f"  saved {len(out):,} rows ({len(df):,} patients x {stages} stages) -> {output_path} "
          f"| splits: {dict(Counter(out['split']))}")
    return out


# ── Legacy: single-stage (V4/ICD only) mimic dataset. Kept for reference; no
# longer wired into run_postprocessing.py (superseded by build_mimic_instructions
# above). ─────────────────────────────────────────────────────────────────────


def build_mimic_instructions_icd_only(combined_path: str, output_path: str) -> pd.DataFrame:
    cols = ["hadm_id", "subject_id", "split", "Chief Complaint",
            "admission_note", "ICD_CODES", "labs"]
    df = pd.read_parquet(combined_path, columns=cols)
    df = df.drop_duplicates(subset="hadm_id", keep="first")

    rows = []
    for _, row in df.iterrows():
        rows.append({
            "hadm_id": row["hadm_id"],
            "subject_id": row["subject_id"],
            "split": row["split"],
            "chief_complaint": str(row["Chief Complaint"]).replace(" ", "_"),
            "Input": _build_icd_input(row["admission_note"], row.get("labs")),
            "Output": _build_icd_output(row["ICD_CODES"]),
        })

    out = pd.DataFrame(rows, columns=["hadm_id", "subject_id", "split", "chief_complaint", "Input", "Output"])
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(output_path, index=False)
    print(f"  saved {len(out):,} patients → {output_path} | splits: {dict(Counter(out['split']))}")
    return out
