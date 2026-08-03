import logging
from dataclasses import dataclass
from typing import List, Optional, Type

import yaml
from pandas import DataFrame

from src.exp_args import ExpArgs
from src.pipeline.prompts import PROMPT_TEMPLATES_GEN, PROMPT_TEMPLATES_EVAL_MERLIN, \
    PROMPT_TEMPLATES_EVAL_MIMIC, PROMPT_TEMPLATES_EVAL_NO_LAB, EXTRACT_PROMPT_DICT
from src.pipeline.pydantic_schemas import DiagnosesModel, ICDsModel, get_yaml_scheme, \
    build_extraction_schema, get_potential_diagnoses_for_complaints
from src.utils import convert_code_to_short_code


@dataclass
class VerifierArgs:
    start_verifier: int
    verifier_steps: List[int]
    current_verifier: int
    chief_complaint: str
    budgets_: List[int]
    temperatures_: List[float]
    max_tokens_: List[int]
    verifier_thresholds_: List[float]
    schemas_: List[Optional[Type]]
    potential_diagnoses: dict
    potential_icds: set[str]
    concurrency: int
    manifestations_yaml: any = None
    manifestation_extraction_schema: any = None
    current_budget: int = -1
    generated_prompts: int = 0

    def __init__(self,
                 exp_args: ExpArgs,
                 chief_complaint: str = "abdominal_pain",
                 start_verifier: int = 1,
                 budgets: List[int] = None,
                 temperatures: List[float] = None,
                 max_tokens: List[int] = None,
                 verifier_thresholds: List[float] = None,
                 num_choices: int = 10,
                 concurrency: int = 16,
                 temperature_step: float = 0.2,
                 max_tokens_step: int = 200,
                 ):

        self.chief_complaint = chief_complaint
        self.start_verifier = start_verifier
        self.current_verifier = self.start_verifier
        self.verifier_steps = [x for x in range(start_verifier, 5)]

        # Schemas and prompt types
        self.prompt_types = ["EXTRACT_PROMPT", "DIAGNOSE_PROMPT", "RERANK_PROMPT", "ICD_PROMPT"]
        chief_complaints = ['abdominal_pain', 'back_pain', 'chest_pain', 'cough', 'diarrhea',
                            'dyspnea', 'headache']

        # ToDo CleanUp
        manifestations_yaml = {}
        extraction_schemas = {}
        for cc in chief_complaints:
            manifestation_yaml = get_yaml_scheme(exp_args, cc)
            manifestations_yaml[cc] = manifestation_yaml
            extraction_schemas[cc] = build_extraction_schema(manifestation_yaml)

        self.manifestations_yaml = manifestations_yaml
        self.schemas_ = [extraction_schemas, DiagnosesModel, DiagnosesModel, ICDsModel]

        # params individual to each verifier
        self._set_prompt_template(exp_args)
        self._set_temperatures(exp_args, temperatures)
        self.budgets_ = budgets or [4, 4, 4, 4]
        self.max_tokens_ = max_tokens or [1500, 1500, 1500, 1500]
        self.verifier_thresholds_ = verifier_thresholds or [0.3, 0.3, 0.3, 0.3]

        # params shared across verifiers
        self.num_choices_ = num_choices
        self.concurrency = concurrency
        self.temperature_step = 0.0 if exp_args.guided_decoding else temperature_step
        self.max_tokens_step = max_tokens_step

        self.current_budget = self.budget
        logging.info(self._start_verifier_string)

    def set_labelspace(self, exp_args: ExpArgs, patients: DataFrame):
        """Sets the potential diagnoses and ICD codes based on chief complaint."""
        all_complaints = patients['Chief Complaint'].unique().tolist()
        # all_complaints = [complaint.replace(' ', '_') for complaint in all_complaints]
        self.potential_diagnoses = get_potential_diagnoses_for_complaints(exp_args, all_complaints)
        # ICD-10 Code Labelspace
        code_lists = patients['ICD_CODES'].tolist()
        self.potential_icds = set([convert_code_to_short_code(code) for codes in code_lists
                                   for code in codes])

    def _set_temperatures(self, exp_args: ExpArgs, temperatures: List[float] = None):
        """Sets the temperatures for each verifier step, no temperature with guided decoding."""
        # temperatures = [0.0, 0.0, 0.0, 0.0] if exp_args.guided_decoding else temperatures
        self.temperatures_ = temperatures or [0.2, 0.2, 0.2, 0.2]

    def _set_current_budget(self, current_budget: Optional[int]):
        """Check if current budget is given from checkpoint, set only once"""
        if self.current_budget == -1 and current_budget:
            self.current_budget = current_budget
        else:
            self.current_budget = self.budget

    def _set_prompt_template(self, exp_args: ExpArgs):
        """Sets the appropriate prompt templates based on experiment args."""
        if exp_args.eval_mode and exp_args.merlin_mode:
            self.prompt_templates = PROMPT_TEMPLATES_EVAL_MERLIN
        elif exp_args.eval_mode and exp_args.think_about_labs:
            self.prompt_templates = PROMPT_TEMPLATES_EVAL_NO_LAB
        elif exp_args.eval_mode and not exp_args.merlin_mode:
            self.prompt_templates = PROMPT_TEMPLATES_EVAL_MIMIC
        else:
            self.prompt_templates = PROMPT_TEMPLATES_GEN

        self.prompt_templates['EXTRACT_PROMPT'] = EXTRACT_PROMPT_DICT

    def set_next_verifier(self, eval_mode: bool):
        if eval_mode and self.current_verifier == 2:
            self.current_verifier = 4
            self.current_budget = self.budget
            logging.info('Skipping verifier step 3 in eval mode.')
            return

        if self.current_verifier < max(self.verifier_steps):
            self.current_verifier += 1
            self.current_budget = self.budget
            logging.info(self._start_verifier_string)

    @property
    def is_last_verifier(self) -> bool:
        return self.current_verifier >= max(self.verifier_steps)

    def adapt_params_for_next_generation(self, work_sample_len: int):
        self.generated_prompts += work_sample_len
        self.current_budget -= 1

        if self.current_budget != self.budget:
            if self.temperature <= 1.4:
                self.temperatures_[self.current_verifier - 1] += self.temperature_step
            self.max_tokens_[self.current_verifier - 1] += self.max_tokens_step

    @property
    def verifier_pos(self):
        return self.current_verifier - 1

    @property
    def num_choices(self):
        if self.temperature == 0.0:
            # n = 1 for temperature 0 (deterministic output)
            return 1
        return self.num_choices_

    @property
    def _prompt_type(self):
        return self.prompt_types[self.verifier_pos]

    def get_pydantic_scheme(self, chief_complaint: str = None):
        # Return the raw object (could be dict or Model)
        if isinstance(self.schemas_[self.verifier_pos], dict):
            manifestation_schemas_ = self.schemas_[self.verifier_pos]
            if chief_complaint is None:
                return manifestation_schemas_['abdominal_pain']
            else:
                return self.schemas_[self.verifier_pos][chief_complaint.replace(' ', '_')]
        else:
            return self.schemas_[self.verifier_pos]

    def get_manifestation_yaml(self, chief_complaint: str):
        return self.manifestations_yaml[chief_complaint.replace(' ', '_')]

    def get_prompt_template(self, chief_complaint: str):
        if self._prompt_type == 'EXTRACT_PROMPT':
            return self.prompt_templates[self._prompt_type].get(chief_complaint, EXTRACT_PROMPT_DICT['generic'])
        return self.prompt_templates[self._prompt_type]

    @property
    def budget(self):
        return self.budgets_[self.verifier_pos]

    @property
    def temperature(self):
        return self.temperatures_[self.verifier_pos]

    @property
    def max_tokens(self):
        return self.max_tokens_[self.verifier_pos]

    @property
    def threshold(self):
        return self.verifier_thresholds_[self.verifier_pos]

    @property
    def _start_verifier_string(self):
        # Safe check for the name attribute
        # scheme = self.pydantic_scheme
        # scheme_name = "DynamicExtractionMapping" if isinstance(scheme, dict) else scheme.__name__
        return (f'Start with verifier {self.current_verifier} using {self._prompt_type}')

    def store_checkpoint(self, path: str):
        step_dict = {
            "current_verifier": self.current_verifier,
            "current_budget": self.current_budget,
            "generated_prompts": self.generated_prompts,
            "start_verifier": self.start_verifier,
            "temperatures": self.temperatures_,
            "max_tokens": self.max_tokens_,
        }
        with open(f"{path}_data.yaml", "w") as f:
            yaml.dump(step_dict, f)

    def load_from_checkpoint(self, path: str):
        with open(f"{path}_data.yaml", "r") as f:
            step_dict = yaml.safe_load(f)

        self.current_verifier = step_dict["current_verifier"]
        self.current_budget = step_dict["current_budget"]
        self.generated_prompts = step_dict["generated_prompts"]
        self.start_verifier = step_dict["start_verifier"]
        self.temperatures_ = step_dict["temperatures"]
        self.max_tokens_ = step_dict["max_tokens"]

        logging.info(f'Loaded verifier checkpoint from step {self.current_verifier} '
                     f'with budget {self.current_budget}')
