"""Pydantic schemas + guided-decoding helpers for the QA stages.

Stage 2 needs a JSON schema with one boolean field per taxonomy class, built at
runtime from the (human-approved) taxonomy. We deliberately use a FLAT OBJECT of
booleans rather than an array of objects: it is trivial for the model, it maps
one-to-one onto the aggregation table, and it side-steps the vLLM guided-decoding
array-schema failure we hit on the eval side (see the guided_decoding_array_schema
memory) -- object-of-scalars schemas decode cleanly.
"""
from __future__ import annotations

import json
import re
from typing import List, Optional, Type

from pydantic import BaseModel, Field, ValidationError, create_model, field_validator


# --- Stage 1 validation ------------------------------------------------------

class RcaError(BaseModel):
    kind: str = ""
    icd_codes: List[str] = Field(default_factory=list)
    keyword: str = ""
    root_cause: str = ""
    inferable_from_admission: Optional[bool] = None


class RcaResult(BaseModel):
    """Accepts both RCA formats so checkpointed responses stay valid across
    prompt revisions: the current simple form {"errors": ["tag", ...]} and the
    older rich form {"errors": [{"kind": ..., "keyword": ..., ...}]}."""
    case_summary: str = ""
    errors: List[RcaError] = Field(default_factory=list)

    @field_validator("errors", mode="before")
    @classmethod
    def _coerce_tags(cls, v):
        if isinstance(v, list):
            return [{"keyword": e} if isinstance(e, str) else e for e in v]
        return v


class TaxonomyClass(BaseModel):
    name: str
    label: str = ""
    definition: str = ""
    positive_example: str = ""
    decision_rule: str = ""
    # Stage-1 provenance: which error keywords were clustered into this class.
    keywords: List[str] = Field(default_factory=list)


class Taxonomy(BaseModel):
    classes: List[TaxonomyClass]

    def names(self) -> List[str]:
        return [c.name for c in self.classes]


# --- Stage 2 dynamic schema --------------------------------------------------

def build_classification_model(class_names: List[str]) -> Type[BaseModel]:
    """A pydantic model with one required bool per class name (flat object)."""
    fields = {name: (bool, ...) for name in class_names}
    return create_model("CaseClassification", **fields)


def taxonomy_block(classes: List[TaxonomyClass]) -> str:
    """Render the taxonomy for injection into the stage-2 prompt."""
    lines = []
    for c in classes:
        lines.append(f"- {c.name} ({c.label}): {c.definition} Decision rule: {c.decision_rule}")
    return "\n".join(lines)


def example_block(class_names: List[str]) -> str:
    """A tiny JSON example so the model returns the exact keys we expect."""
    return json.dumps({n: False for n in class_names}, indent=2)


# --- Shared JSON extraction --------------------------------------------------

def extract_json_object(text: str) -> Optional[dict]:
    """Pull the first balanced {...} object out of a model response, tolerant of
    ```json fences and chatter around it. Returns None on failure."""
    if not text:
        return None
    # Drop any thinking block (Qwen3 emits <think>...</think> before the answer;
    # an unclosed <think> means the whole budget went to thoughts -> no JSON).
    text = re.sub(r"<think>.*?(</think>|$)", "", text, flags=re.DOTALL)
    # Strip code fences if present.
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    candidate = fenced.group(1) if fenced else None
    if candidate is None:
        m = re.search(r"\{.*\}", text, re.DOTALL)
        candidate = m.group(0) if m else None
    if candidate is None:
        return None
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        return None


def validate_stage2(text: str, model: Type[BaseModel]) -> Optional[dict]:
    obj = extract_json_object(text)
    if obj is None:
        return None
    try:
        return model.model_validate(obj).model_dump()
    except ValidationError:
        return None


def validate_stage1(text: str) -> Optional[dict]:
    obj = extract_json_object(text)
    if obj is None:
        return None
    try:
        return RcaResult.model_validate(obj).model_dump()
    except ValidationError:
        return None
