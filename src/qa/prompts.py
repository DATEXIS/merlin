"""Prompts for the two-stage QA error analysis.

Stage 1 has two prompts:
  * STAGE1_RCA_PROMPT  -- per-case free-text root-cause analysis.
  * STAGE1_CONSOLIDATE_PROMPT -- fold many RCAs into a candidate taxonomy.
Stage 2 has one:
  * STAGE2_CLASSIFY_PROMPT -- score a single case against the frozen taxonomy
    (one yes/no per class). The class list is injected from the taxonomy file so
    the prompt and the JSON schema always agree.

Design notes
------------
* The judge never sees the discharge note at *prediction* time, but it DOES see it
  here: at analysis time the discharge note is the ground-truth signal used to
  explain what went wrong. This is intentional and safe -- QA is post-hoc.
* We evaluate ICD-10 codes at the 3-character category level (K35 vs K36 matters;
  K35.2 vs K35.3 does not). The prompts say so explicitly to keep the judge from
  flagging spurious specificity mismatches.
"""

# --- Stage 1: per-case root-cause analysis -----------------------------------

STAGE1_RCA_PROMPT = """You are a Senior Medical Coding Auditor performing a root-cause analysis (RCA) \
on ICD-10 code predictions made by a language model.

You are given four inputs:

1. ADMISSION PROMPT (what the prediction model saw -- it did NOT see the discharge note):
{prompt}

2. MODEL PREDICTION (the model's reasoning and predicted ICD-10 categories):
{predicted_output}

3. GOLD ICD-10 CODES (the codes actually assigned to this admission):
{labels}

4. DISCHARGE NOTE (ground-truth clinical picture, for your analysis only):
{discharge_note}

Task
----
Compare the prediction against the gold codes, using the discharge note as the \
source of truth, and name the HIGH-LEVEL ERROR PATTERNS this case exhibits.

Rules
-----
- Judge at the ICD-10 three-character category level (e.g. K35 vs K36 matters; \
K35.2 vs K35.3 does not).
- Name abstract, recurring patterns -- the kind of tag that would apply to many \
similar cases -- NOT case-specific descriptions. Examples of the right granularity:
  too_unspecific_code, underpredicting_omission, history_error, medication_error.
  These are illustrations of style and granularity; use whatever patterns actually
  fit the case.
- 1 to 4 tags per case; only patterns that clearly apply.
- snake_case, 2-4 words per tag.

Return a JSON object exactly matching this shape:
{{
  "errors": ["missed_chronic_comorbidity", "symptom_overcoding"]
}}
Return only the JSON object."""


# --- Stage 1: cluster error keywords into a candidate taxonomy ----------------

STAGE1_CLUSTER_PROMPT = """You are a Senior Medical Coding Auditor building an error taxonomy.

An audit assigned a short KEYWORD to every individual error found in ICD-10 \
prediction cases, across several models of different sizes and training setups. \
Below is each unique keyword with its frequency and example root causes.

Your job: group ALL keywords into a set of abstract error classes -- aim for 12-16, \
finer-grained than a prior pass that collapsed to 7. When a keyword group could go \
either way, prefer splitting into two distinct classes over merging, as long as each \
side still meets the bar below. Each class must be:
- clinically meaningful and distinct from the others,
- decidable from admission note + prediction + gold codes + discharge note with a \
simple yes/no ("does this case exhibit this error?"),
- named and defined so a different auditor would apply it the same way.

Derive the classes purely from the keyword evidence below. If a group of keywords \
describes a failure mode that only some models exhibit (e.g. degenerate or \
repetitive outputs), give it its own class rather than folding it into a broader one.

Include one catch-all class named "other". Every keyword must be assigned to \
exactly one class (unassigned keywords will be counted as "other"). Copy keyword \
names into the output EXACTLY as they appear in the input, character for character.

Keywords (keyword | count | example root causes):
{keyword_data}

Return a JSON object exactly matching this shape:
{{
  "classes": [
    {{
      "name": "snake_case_identifier",
      "label": "Human Readable Label",
      "definition": "what counts as this error, in one or two sentences",
      "positive_example": "a short paraphrased case that exhibits it",
      "decision_rule": "the yes/no test an auditor applies",
      "keywords": ["every", "keyword", "assigned", "to", "this", "class"]
    }}
  ]
}}
Return only the JSON object."""


# --- Stage 2: classify one case against the frozen taxonomy -------------------

STAGE2_CLASSIFY_PROMPT = """You are a Senior Medical Coding Auditor. You apply a FIXED error taxonomy to a \
single ICD-10 prediction case and decide, independently for each class, whether the \
case exhibits that error.

Inputs:

1. ADMISSION PROMPT (what the prediction model saw):
{prompt}

2. MODEL PREDICTION:
{predicted_output}

3. GOLD ICD-10 CODES:
{labels}

4. DISCHARGE NOTE (ground truth, for your judgement only):
{discharge_note}

Error taxonomy (apply every class; classes are independent and NOT mutually exclusive):
{taxonomy_block}

POLARITY -- read first
----------------------
For EVERY class, the boolean means the SAME thing:
  true  = the error described by the class IS present in this case.
  false = the error is NOT present.
Never invert this. A model that coded well scores false; a model that made the
error scores true. When a class says "missed", true means something WAS missed.

Reference for the taxonomy
--------------------------
- PRINCIPAL DIAGNOSIS: the main reason for the admission. Find it in the discharge \
note's "Principal Diagnosis" / "Discharge Diagnosis" section (or, absent an explicit \
heading, the single condition the stay was chiefly about). Several classes are judged \
against this one code.
- ICD-10 Z-code families used by the taxonomy:
  * History / status codes = Z80-Z99 (e.g. Z85 personal history of malignancy, \
Z86/Z87 personal history of other conditions, Z95 presence of implanted device).
  * Long-term (current) medication codes = Z79 (e.g. Z79.4 long-term insulin use, \
Z79.01 long-term anticoagulant use).

How to decide the "missed_*" coverage classes (do this literally, step by step)
-------------------------------------------------------------------------------
Work with three-character categories throughout (I219 -> I21, Z794 -> Z79).
  1. From the GOLD codes, write down the categories that belong to this class's
     target group (e.g. for missed_history: the gold categories in Z80-Z99).
  2. Collect the set of categories the PREDICTION contains.
  3. Go through each target category from step 1 and check whether it appears in
     the prediction set from step 2.
  4. If AT LEAST ONE target category is present in gold but absent from the
     prediction -> the code was missed -> answer TRUE.
     If the target group is empty, or every target category is also in the
     prediction -> answer FALSE.
This is positive matching: you are looking FOR a present-in-gold, absent-in-
prediction category. Finding one means the error is present (true).

Worked micro-examples (missed_history)
  * gold has Z85, Z87; prediction has {{I21, E11, Z85}} -> Z87 is in gold but not
    predicted -> true.
  * gold has Z85; prediction has {{Z85, I21}} -> Z85 is covered -> false.
  * gold has no Z80-Z99 code -> false.

Rules
-----
- Judge at the ICD-10 three-character category level.
- Apply the polarity contract above to every class.
- redundancy_symptom_misuse and unspecific_primary_diagnosis are the false-positive / \
specificity checks: answer true only when the low-value or over-general coding is clear.
- A case may exhibit several classes, one, or none. not_inferable_from_admission is a \
qualifier that may co-occur with a missed_* class -- decide it independently.
- Decide each class on its own definition; do not force exactly one.

Return a JSON object with exactly one boolean field per class name, e.g.:
{example_block}
Return only the JSON object."""
