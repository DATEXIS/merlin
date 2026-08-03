import logging
from pathlib import Path
from typing import Annotated

import yaml
from pydantic import BaseModel, RootModel, Field, create_model, conint

from src.exp_args import ExpArgs
from src.utils import load_diseases_for_chief_complaint


class DiagnosisModel(BaseModel):
    name: str
    reason: str


class DiagnosesModel(
    RootModel[Annotated[list[DiagnosisModel], Field(min_length=3, max_length=10)]]):
    pass


class DiagnosesExtractModel(RootModel[list[DiagnosisModel]]):
    pass


class ICDModel(BaseModel):
    icd_code: str
    reason: str


class ICDsModel(RootModel[Annotated[list[ICDModel], Field(min_length=3, max_length=30)]]):
    pass


class ICDsExtractModel(RootModel[list[ICDModel]]):
    pass


def build_extraction_schema(yaml_data: dict[str, list]) -> type[BaseModel]:
    """Recursive function to create the model from the YAML"""
    def make_submodel(field_name: str, allowed_values: list) -> type[BaseModel]:
        min_val, max_val = min(allowed_values), max(allowed_values)
        value_type = conint(strict=True, ge=min_val, le=max_val)

        return create_model(
            field_name.replace(" ", "_").replace("/", "_"),
            value=(value_type, Field(..., description=f"Allowed values: {allowed_values}")),
            reasoning=(str, Field(...))
        )

    fields_dict: dict[str, tuple] = {}

    for key, allowed_vals in yaml_data.items():
        submodel = make_submodel(key, allowed_vals)
        fields_dict[key] = (submodel, ...)

    # Create and return the final model class
    return create_model('ManifestationsModel', **fields_dict)


def get_yaml_scheme(args: ExpArgs, chief_complaint: str) -> (dict, BaseModel):

    yaml_path = Path(f"{args.data_dir}/medical_schemes/symptoms/{chief_complaint}.yaml")

    try:
        with open(yaml_path, 'r') as f:
            return yaml.safe_load(f)
            # yaml_schema = yaml.safe_load(f)
    except FileNotFoundError as e:
        logging.error(f'No yaml scheme found for {chief_complaint} at {yaml_path}')
        raise e


def get_potential_diagnoses_for_complaints(exp_args: ExpArgs, chief_complaints: list[str]
                                           ) -> dict[str, list[str]]:
    potential_diagnoses = {}
    logging.info(f'Loading potential diagnoses for complaints: {chief_complaints}')
    for chief_complaint in chief_complaints:
        diseases = load_diseases_for_chief_complaint(exp_args, chief_complaint)
        potential_diagnoses[chief_complaint] = diseases['Disease'].tolist()
    return potential_diagnoses
