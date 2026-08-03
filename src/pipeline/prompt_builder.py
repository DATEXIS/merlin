import ast
import json
import logging

import pandas as pd

from src.pipeline.json_extraction import ICDsModel
from src.pipeline.verifier_args import VerifierArgs


def _get_symptom_format_example(symptom_dict) -> str:
    """Converts a symptom dict to a structured JSON format with value, evidence, and reasoning."""
    structured_data = {}
    for symptom, _ in symptom_dict.items():
        structured_data[symptom] = {
            "value": 0,
            "reasoning": f"Based on the admission note, {symptom} is suspected to be present / "
                         f"absent / unmentioned."
        }
    return json.dumps(structured_data, indent=4)


def _get_diagnose_format_example() -> str:
    return '''[
        {
            "name": "diagnosis_0",
            "reason": "Based on clinical evidence, diagnosis_0 is suspected."
        },
        {
            "name": "diagnosis_1",
            "reason": "Based on clinical evidence, diagnosis_1 is suspected."
        },
        ...
        {
            "name": "diagnosis_9",
            "reason": "Based on clinical evidence, diagnosis_9 is suspected."
        }
    ]'''


def _get_icd_format_example() -> str:
    return '''[
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


def _format_manifestations_from_string(dict_str: str) -> str:
    try:
        data = ast.literal_eval(dict_str)
    except (ValueError, SyntaxError):
        return f"Invalid dictionary string provided: {dict_str}"

    present_section = ["\n#### Present Manifestations"]
    absent_section = ["\n#### Absent Manifestations"]

    for manifestation, details in data.items():
        formatted_entry = (
            f"  - {manifestation}: {details.get('reasoning', 'N/A')}"
        )
        if details.get("value") == 1:
            present_section.append(formatted_entry)
        elif details.get("value") == -1:
            absent_section.append(formatted_entry)

    return "\n".join(present_section + absent_section)


def _get_diagnose_string(value, label, fallback):
    """Convert stringified JSON if valid, otherwise return fallback."""
    if pd.isna(value) or value in "None":
        return fallback
    else:
        diagnoses = ast.literal_eval(value)
        diagnose_string = ""
        for i, d in enumerate(diagnoses):
            diagnose_string += f"  {i+1}. {d['name']}: {d.get('reason', 'No reason provided.')}\n"
            if i > 4 and d['name'] == label:
                break
        return diagnose_string


def get_json_format_examples(schema) -> dict:
    return {
        'symptom_format_example': _get_symptom_format_example(schema),
        'diagnose_format_example': _get_diagnose_format_example(),
        'icd_format_example': _get_icd_format_example(),
    }


def extract_verifier_results(v_args: VerifierArgs, patient: dict) -> dict:
    """Extracts manifestations and diagnoses from previous verifiers."""
    # Manifestations
    v1_json = patient.get('v1_json', "None")
    manifestation_string = (
        _format_manifestations_from_string(v1_json)
        if v1_json not in ("None", None)
        else "Failed to extract manifestations"
    )
    v2_json = patient.get('v2_json', "None")
    v3_json = patient.get('v3_json', v2_json)

    diagnose_label = patient['disease']
    potential_diagnoses = v_args.potential_diagnoses[patient['Chief Complaint']]
    diagnose_string = _get_diagnose_string(v2_json, diagnose_label, potential_diagnoses)

    if v_args.get_pydantic_scheme() == ICDsModel:
        diagnose_string = _get_diagnose_string(v3_json, diagnose_label, "Not assigned")

    return {
        'potential_diagnoses': diagnose_string,
        'clinical_manifestations': manifestation_string,
        # For qualitative analysis
        'v4_prompt': patient.get('v4_prompt', "None"),
        'v4_json': patient.get('v4_json', "None"),
        'ICD_CODES': str(patient.get('ICD_CODES', "None")),
    }


def build_prompt(patient: dict, v_args: VerifierArgs) -> str:

    chief_complaint = patient['Chief Complaint']
    current_schema = v_args.get_manifestation_yaml(chief_complaint)
    prompt_data = get_json_format_examples(current_schema)

    # 2. Get the verifier results (manifestations, etc.)
    verifier_results = extract_verifier_results(v_args, patient)

    # Collect all prompt data to fill in the template
    prompt_data.update({
        'admission_note': patient['admission_note'],
        'laboratory_results': patient['labs'],
        **verifier_results,
    })

    chief_complaint = patient['Chief Complaint']
    template = v_args.get_prompt_template(chief_complaint)
    full_prompt = template.format(**prompt_data)

    # 25,000 chars is a safe ceiling for an 8192 token limit
    max_prompt_chars = 25000
    if len(full_prompt) > max_prompt_chars:
        logging.warning(f"Prompt too long for hadm_id: {patient.get('hadm_id')}. Truncating.")
        return full_prompt[:max_prompt_chars]

    return full_prompt


def extract_admission_note_from_prompt(text: str) -> str:
    while not isinstance(text, str):
        text = text[0]

    idx = text.find("### Admission Note")
    if idx != -1:
        return text[idx:]
    else:
        return text  # fallback if not found


def build_prompts(v_args: VerifierArgs, patients: pd.DataFrame) -> list[str]:
    patient_dicts = patients.to_dict(orient="records")
    return [build_prompt(patients_dict, v_args) for patients_dict in patient_dicts]
