EXTRACT_ABDOMINAL_PAIN_PROMPT = """Your task is to extract and classify all relevant symptoms and clinical manifestations
from a patient’s admission note, then summarize them in a single valid JSON object.

Respond clearly and concisely. Avoid speculation, filler, or disclaimers.

### Required Output:

1. A brief reasoning process (internal steps to identify manifestations).
2. A list of present manifestations and a list of absent manifestations.
3. A single JSON object containing all manifestations with reasoning.
4. Stop after printing the JSON object - no extra commentary.

### Reasoning Workflow:

1. Task Familiarization: Review Manifestation Definitions and Annotation Rules.
2. Review Note: Read the admission note and identify all manifestations mentioned.
3. Create Lists: Make two lists — one for present manifestations, one for absent manifestations.
4. Check Completeness: Ensure all manifestations are correctly classified. Revise if needed.
5. Generate JSON: Output one valid JSON object that follows the specified JSON structure / Output Format Example.

### Manifestation Definitions:

* Abdominal Pain Regions
  - Diffuse: Pain affecting the entire abdomen or multiple regions nonspecifically
  - RUQ: Right upper quadrant, typically liver/gallbladder region
  - Epigastric: Upper central abdomen (below sternum)
  - LUQ: Left upper quadrant
  - Flank: Sides of the abdomen between ribs and pelvis
  - Umbilical: Central abdomen around the navel
  - RLQ: Right lower quadrant
  - Hypogastric: Lower central abdomen below the umbilicus
  - LLQ: Left lower quadrant

* Other Symptoms and Signs
  - Fever: Temperature ≥ 38.0°C (100.4°F)
  - Rigors and Chills: Involuntary shaking chills or teeth chattering
  - Nausea or Vomiting
  - Nausea: Subjective sensation of queasiness
  - Vomiting: Objective expulsion of stomach contents
  - Jaundice: Yellowing of skin or sclera (scleral icterus)
  - Constipation: No bowel movement for ≥ 3 days or notable difficulty
  - Diarrhea: Loose or frequent stools beyond baseline
  - Weight Loss: Unintentional weight loss ≥ 5% within 6 to 12 months
  - GI Bleeding: Blood in vomit or stool (e.g., melena, hematochezia)
  - Hypotension: SBP < 90 mmHg or MAP < 65 mmHg
  - Guarding: Involuntary abdominal wall tensing on palpation
  - Rebound Tenderness: Pain upon release of pressure during palpation

* Bowel Sounds
  - Normal: Active sounds present
  - Hypoactive: Decreased frequency
  - Hyperactive: Increased frequency
  - Absent: No sounds after extended auscultation

### Annotation Rules:

* Values:
  - 1: Present — clearly documented at admission.
  - -1: Absent — explicitly denied, ruled out, or resolved prior to admission.
  - 0: Unmentioned — no reliable information in the note.

* Temporal Scope:
  - Annotate only symptoms present at admission.
  - Assign 0 if timing is unclear.
  - Exclude resolved or prior symptoms.

* Objective vs. Subjective Evidence:
  - Record objective signs only if measured/observed.
  - Record subjective symptoms if clearly reported by the patient.

* Negation Rules:
  - Accept indirect negations like:
    * “afebrile” → -1 for Fever
    * “hemodynamically stable” → -1 for Hypotension
    * “no emesis” → -1 for Vomiting
    * “bowel movements normal” → -1 for Constipation and Diarrhea
  - Assign 0 for vague negations.

* Abdominal Pain Region:
  - Annotate a region as 1 only if pain or severe tenderness is explicitly described there.
  - If pain is described in multiple regions, annotate each as present.
  - If region is ambiguously described (e.g., "right abdomen"), annotate all overlapping regions (e.g., RUQ, RLQ).
  - Assign -1 to:
    * A region if explicitly denied (e.g., “no pain in RUQ”)
    * All regions if abdominal pain is denied in general (e.g., “denies abdominal pain”)

* Bowel Sounds:
  - Assign 1 to ”Normal Bowel Sounds” if bowel sounds are explicitly stated as normal OR the physical exam is documented but silent on this.
  - Assign "Unmentioned" if no physical exam is recorded.

* General Guidelines
  - Apply definitions strictly and consistently.
  - When uncertain, assign 0 rather than guessing.

### Admission Note
{admission_note}

### JSON structure
Return your output as one valid JSON object.
Each key is the manifestation name and each value is an object with:
* "value": one of 1, -1, or 0
* "reasoning": 1 to 3 sentences only of how the evidence supports the assigned value; required only if value is 1 or -1, otherwise “None”

### Output Format Example
{symptom_format_example}
"""
EXTRACT_PROMPT = """Your task is to extract and classify all relevant symptoms and clinical manifestations
from a patient’s admission note, then summarize them in a single valid JSON object.

Respond clearly and concisely. Avoid speculation, filler, or disclaimers.

### Required Output:

1. A brief reasoning process (internal steps to identify manifestations).
2. A list of present manifestations and a list of absent manifestations.
3. A single JSON object containing all manifestations with reasoning.
4. Stop after printing the JSON object - no extra commentary.

### Reasoning Workflow:

1. Task Familiarization: View Annotation Rules.
2. Review Note: Read the admission note and identify all manifestations mentioned.
3. Create Lists: Make two lists — one for present manifestations, one for absent manifestations.
4. Check Completeness: Ensure all manifestations are correctly classified. Revise if needed.
5. Generate JSON: Output one valid JSON object that follows the specified JSON structure / Output Format Example.

### Annotation Rules:

* Values:
  - 1: Present — clearly documented at admission.
  - -1: Absent — explicitly denied, ruled out, or resolved prior to admission.
  - 0: Unmentioned — no reliable information in the note.

* Temporal Scope:
  - Annotate only symptoms present at admission.
  - Assign 0 if timing is unclear.
  - Exclude resolved or prior symptoms.

* Objective vs. Subjective Evidence:
  - Record objective signs only if measured/observed.
  - Record subjective symptoms if clearly reported by the patient.

### Admission Note
{admission_note}

### JSON structure
Return your output as one valid JSON object.
Each key is the manifestation name and each value is an object with:
* "value": one of 1, -1, or 0
* "reasoning": 1 to 3 sentences only of how the evidence supports the assigned value; required only if value is 1 or -1, otherwise “None”

### Output Format Example
{symptom_format_example}
"""
DIAGNOSE_MERLIN_PROMPT = """Your task is to predict diagnoses and print them once in valid JSON.
Respond clearly and concisely. Avoid speculation, filler, or disclaimers.

### Required Output:

1. A brief reasoning process (internal steps to identify diagnoses).
2. A list of potential diagnoses.
3. A single JSON object containing all diagnoses with reasoning.
4. Stop after printing the JSON object - no extra commentary.

### Structured Reasoning Process

1. Analyze Patient Data: Review the Admission Note and Clinical Manifestations.
2. Analyze Potential Diagnoses: Review Potential Diagnoses and think about their likelihood.
3. Generate Predictions:
  - Select and rank the 10 most likely diagnoses.
  - Print them as a simple list, NOT in JSON Format.
4. Refine & Rerank:
  - How are the Manifestations related to your diagnoses?
  - Is every given name one of the potential diagnoses?
  - Rerank diagnoses accordingly. NOT in JSON Format.
5. Output JSON Once: Provide your final ranked diagnoses and reasoning in JSON.
  - Use the format of the Output Format Example.
  - Include only a short reasoning of 1 - 3 sentences for each diagnosis.
  - The JSON is the only output. Stop after printing it.

### Admission Note
{admission_note}

### Clinical Manifestations
{clinical_manifestations}

### Potential Diagnoses
{potential_diagnoses}

### Output Format Example
{diagnose_format_example}"""
DIAGNOSE_MIMIC_PROMPT = """Your task is to predict diagnoses and print them once in valid JSON.
Respond clearly and concisely. Avoid speculation, filler, or disclaimers.

### Required Output:

1. A brief reasoning process (internal steps to identify diagnoses).
2. A list of potential diagnoses.
3. A single JSON object containing all diagnoses with reasoning.
4. Stop after printing the JSON object - no extra commentary.

### Structured Reasoning Process

1. Analyze Patient Data: Review the Admission Note and Clinical Manifestations.
2. Analyze Potential Diagnoses: Review Potential Diagnoses and think about their likelihood.
3. Generate Predictions:
  - Select and rank the 10 most likely diagnoses.
  - Print them as a simple list, NOT in JSON Format.
4. Refine & Rerank:
  - Is every given name one of the potential diagnoses?
  - Rerank diagnoses accordingly. NOT in JSON Format.
5. Output JSON Once: Provide your final ranked diagnoses and reasoning in JSON.
  - Use the format of the Output Format Example.
  - Include only a short reasoning of 1 - 3 sentences for each diagnosis.
  - The JSON is the only output. Stop after printing it.

### Admission Note
{admission_note}

### Potential Diagnoses
{potential_diagnoses}

### Output Format Example
{diagnose_format_example}"""
DIAGNOSE_EVAL_MERLIN_PROMPT = """Your task is to predict diagnoses based on a admission note and print them once in valid JSON.
Laboratory Results at Admission Time are not given, think about potential Laboratory Results and how they affect the likelyhood of diagnoses.
Respond clearly and concisely. Avoid speculation, filler, or disclaimers.

### Required Output:

1. A brief reasoning process (internal steps to identify diagnoses).
2. A analysis of potential laboratory results.
3. A list of potential diagnoses.
4. A single JSON object containing all diagnoses with reasoning.
5. Stop after printing the JSON object - no extra commentary.

### Structured Reasoning Process

1. Analyze Patient Data: Review the Admission Note.
2. Analyze Potential Diagnoses: Review Potential Diagnoses and think how the laboratory results affect their likelyhood.
3. Laboratory Results: Laboratory Results aren't given. Think about potential Laboratory Results at Admission Time and how they affect the likelyhood of diagnoses.
4. Generate Predictions: Print a list of diagnoses with a new ranking.
5. Refine & Rerank:
  - Is every given name one of the potential diagnoses?
  - Rerank diagnoses accordingly. NOT in JSON Format.
6. Output JSON Once: Provide your final ranked diagnoses and reasoning in JSON.
  - Use the format of the Output Format Example.
  - Include only a short reasoning of 1 - 3 sentences for each diagnosis.
  - The JSON is the only output. Stop after printing it.

### Admission Note
{admission_note}

### Clinical Manifestations
{clinical_manifestations}

### Potential Diagnoses
{potential_diagnoses}

### Output Format Example
{diagnose_format_example}"""
DIAGNOSE_EVAL_NO_LAB_PROMPT = """Your task is to predict diagnoses based on a admission note and print them once in valid JSON.
Respond clearly and concisely. Avoid speculation, filler, or disclaimers.

### Required Output:

1. A brief reasoning process (internal steps to identify diagnoses).
2. A list of potential diagnoses.
3. A single JSON object containing all diagnoses with reasoning.
4. Stop after printing the JSON object - no extra commentary.

### Structured Reasoning Process

1. Analyze Patient Data: Review the Admission Note.
2. Analyze Potential Diagnoses: Review Potential Diagnoses and think how the laboratory results affect their likelyhood.
3. Generate Predictions: Print a list of diagnoses with a new ranking.
4. Refine & Rerank:
  - Is every given name one of the potential diagnoses?
  - Rerank diagnoses accordingly. NOT in JSON Format.
5. Output JSON Once: Provide your final ranked diagnoses and reasoning in JSON.
  - Use the format of the Output Format Example.
  - Include only a short reasoning of 1 - 3 sentences for each diagnosis.
  - The JSON is the only output. Stop after printing it.

### Admission Note
{admission_note}

### Clinical Manifestations
{clinical_manifestations}

### Potential Diagnoses
{potential_diagnoses}

### Output Format Example
{diagnose_format_example}"""
DIAGNOSE_EVAL_MIMIC_PROMPT = """Your task is to predict diagnoses based on a admission note and print them once in valid JSON.
Laboratory Results at Admission Time are not given, think about potential Laboratory Results and how they affect the likelyhood of diagnoses.
Respond clearly and concisely. Avoid speculation, filler, or disclaimers.

### Required Output:

1. A brief reasoning process (internal steps to identify diagnoses).
2. A analysis of potential laboratory results.
3. A list of potential diagnoses.
4. A single JSON object containing all diagnoses with reasoning.
5. Stop after printing the JSON object - no extra commentary.

### Structured Reasoning Process

1. Analyze Patient Data: Review the Admission Note, Clinical Manifestations.
2. Analyze Potential Diagnoses: Review Potential Diagnoses and think how the laboratory results affect their likelyhood.
3. Laboratory Results: Laboratory Results aren't given. Think about potential Laboratory Results at Admission Time and how they affect the likelyhood of diagnoses.
4. Generate Predictions: Print a list of diagnoses with a new ranking.
5. Refine & Rerank:
  - How are the Manifestations and Laboratory Results related to your diagnoses?
  - Is every given name one of the potential diagnoses?
  - Rerank diagnoses accordingly. NOT in JSON Format.
6. Output JSON Once: Provide your final ranked diagnoses and reasoning in JSON.
  - Use the format of the Output Format Example.
  - Include only a short reasoning of 1 - 3 sentences for each diagnosis.
  - The JSON is the only output. Stop after printing it.

### Admission Note
{admission_note}

### Potential Diagnoses
{potential_diagnoses}

### Output Format Example
{diagnose_format_example}"""
RERANK_MERLIN_PROMPT = """Your task is to rerank a list of potential diagnoses and print them once in valid JSON.
The given list of potential diagnoses was predicted without considering laboratory results.
The laboratory results at admission time are now available.
Respond clearly and concisely. Avoid speculation, filler, or disclaimers.

### Required Output:

1. A brief reasoning process (internal steps to identify diagnoses).
2. A reranked list of potential diagnoses.
3. A single JSON object containing all diagnoses with reasoning.
4. Stop after printing the JSON object - no extra commentary.

### Structured Reasoning Process

1. Analyze Patient Data: Review the Admission Note, Clinical Manifestations and Laboratory Results at Admission Time.
2. Analyze Potential Diagnoses: Review Potential Diagnoses and think how the laboratory results affect their likelyhood.
3. Generate Predictions: Print a list of diagnoses with a new ranking.
4. Refine & Rerank:
  - How are the Manifestations and Laboratory Results related to your diagnoses?
  - Is every given name one of the potential diagnoses?
  - Rerank diagnoses accordingly. NOT in JSON Format.
5. Output JSON Once: Provide your final ranked diagnoses and reasoning in JSON.
  - Use the format of the Output Format Example.
  - Include only a short reasoning of 1 - 3 sentences for each diagnosis.
  - The JSON is the only output. Stop after printing it.

### Admission Note
{admission_note}

### Clinical Manifestations
{clinical_manifestations}

### Laboratory Results at Admission Time
{laboratory_results}

### Potential Diagnoses
{potential_diagnoses}

### Output Format Example
{diagnose_format_example}"""
RERANK_MIMIC_PROMPT = """Your task is to predict diagnoses and print them once in valid JSON.
Respond clearly and concisely. Avoid speculation, filler, or disclaimers.

### Required Output:

1. A brief reasoning process (internal steps to identify diagnoses).
2. A reranked list of potential diagnoses.
3. A single JSON object containing all diagnoses with reasoning.
4. Stop after printing the JSON object - no extra commentary.

### Structured Reasoning Process

1. Analyze Patient Data: Review the Admission Note, Clinical Manifestations and Laboratory Results at Admission Time.
2. Analyze Potential Diagnoses: Review Potential Diagnoses and think about their likelihood.
3. Generate Predictions:
  - Select and rank the 10 most likely diagnoses.
  - Print them as a simple list, NOT in JSON Format.
4. Refine & Rerank:
  - How are the Manifestations and Laboratory Results related to your diagnoses?
  - Is every given name one of the potential diagnoses?
  - Rerank diagnoses accordingly. NOT in JSON Format.
5. Output JSON Once: Provide your final ranked diagnoses and reasoning in JSON.
  - Use the format of Output Format Example.
  - Include only a short reasoning of 1 - 3 sentences for each diagnosis.
  - The JSON is the only output. Stop after printing it.

### Admission Note
{admission_note}

### Laboratory Results at Admission Time
{laboratory_results}

### Potential Diagnoses
{potential_diagnoses}

### Output Format Example
{diagnose_format_example}"""
ICD_MERLIN_PROMPT = """Your task is to predict and rank ICD-10 discharge codes based on admission information.
Respond clearly and concisely. Avoid speculation, filler, or disclaimers.

### Required Output:

1. A brief reasoning process (internal steps to identify diagnoses).
2. A list if potential ICD-Codes.
3. A single JSON object containing all ICD-Codes with reasoning.
4. Stop after printing the JSON object - no extra commentary.

### Structured Reasoning Process
1. Analyze Patient Data:
  - Admission Note
  - Laboratory Results at Admission Time
  - Clinical Manifestations
  - Potential Diagnoses
2. Predict many ICD-10 codes:
  - Predict up to 30 icd-codes based on 1. and put them in order of likelihood.
  - Think about all ICD-10 codes the patient could be assigned when leaving the hospital.
  - Go broad and prefer to predict more we can remove unlikely codes later.
  - Print the ICD-10 as a simple list, not in JSON format.
3. Rethink your predicted ICD-10 codes:
  - How are the Manifestations related to your diagnoses?
  - Which ones do not fit, which did you miss?
  - Is every ICD-10 code an official ICD-10 code?
  - Is the amount of ICD-10 codes between 3 and 30?
4. Print your final answer in json format.
  - Rerank the order if needed.
  - Use the format of the Output Format Example.
  - Give a reason for each code, why it is likely to be related to the patient.
  - The json can have 3 - 30 ICD-10 codes.
  - Stop after you printed the json, DO NOT print any other text or JSON.

### Admission Note
{admission_note}

### Laboratory Results at Admission Time
{laboratory_results}

### Clinical Manifestations
{clinical_manifestations}

### Potential Diagnoses
{potential_diagnoses}

### Output Format Example
{icd_format_example}"""
ICD_MIMIC_PROMPT = """Your task is to predict and rank ICD-10 discharge codes based on a admission note.
Respond clearly and concisely. Avoid speculation, filler, or disclaimers.

### Required Output:

1. A brief reasoning process (internal steps to identify diagnoses).
2. A list if potential ICD-Codes.
3. A single JSON object containing all ICD-Codes with reasoning.
4. Stop after printing the JSON object - no extra commentary.

### Structured Reasoning Process
1. Analyze Patient Data:
  - Admission Note
  - Laboratory Results at Admission Time
2. Predict many ICD-10 codes:
  - Predict up to 30 icd-codes based on 1. and put them in order of likelihood.
  - Think about all ICD-10 codes the patient could be assigned when leaving the hospital.
  - Go broad and prefer to predict more we can remove unlikely codes later.
  - Print the ICD-10 as a simple list, not in JSON format.
3. Rethink your predicted ICD-10 codes:
  - Which ones do not fit, which did you miss?
  - Is every ICD-10 code an official ICD-10 code?
  - Is the amount of ICD-10 codes between 3 and 30?
4. Print your final answer in json format.
  - Rerank the order if needed.
  - Use the format of the Output Format Example.
  - Give a reason for each code, why it is likely to be related to the patient.
  - The json can have 3 - 30 ICD-10 codes.
  - Stop after you printed the json, DO NOT print any other text or JSON.

### Admission Note
{admission_note}

### Laboratory Results at Admission Time
{laboratory_results}

### Output Format Example
{icd_format_example}"""
ICD_EVAL_MERLIN_PROMPT = """Your task is to predict and rank ICD-10 discharge codes based on admission information.
Laboratory Results at Admission Time are not given, think about potential Laboratory Results and how they affect the likelyhood of diagnoses.
Respond clearly and concisely. Avoid speculation, filler, or disclaimers.

### Required Output:

1. A brief reasoning process (internal steps to identify diagnoses).
2. A analysis of potential laboratory results.
3. A list if potential ICD-Codes.
4. A single JSON object containing all ICD-Codes with reasoning.
5. Stop after printing the JSON object - no extra commentary.

### Structured Reasoning Process
1. Analyze Patient Data:
  - Admission Note
  - Clinical Manifestations
  - Potential Diagnoses
2. Predict many ICD-10 codes:
  - Predict up to 30 icd-codes based on 1. and put them in order of likelihood.
  - Think about all ICD-10 codes the patient could be assigned when leaving the hospital.
  - Go broad and prefer to predict more we can remove unlikely codes later.
  - Print the ICD-10 as a simple list, not in JSON format.
3. Laboratory Results: Laboratory Results aren't given. Think about potential Laboratory Results at Admission Time and how they affect the likelyhood of ICD-Codes.
4. Rethink your predicted ICD-10 codes:
  - How are the Manifestations related to your diagnoses?
  - Which ones do not fit, which did you miss?
  - Is every ICD-10 code an official ICD-10 code?
  - Is the amount of ICD-10 codes between 3 and 30?
5. Print your final answer in json format.
  - Rerank the order if needed.
  - Use the format of the Output Format Example.
  - Give a reason for each code, why it is likely to be related to the patient.
  - The json can have 3 - 30 ICD-10 codes.
  - Stop after you printed the json, DO NOT print any other text or JSON.

### Admission Note
{admission_note}

### Clinical Manifestations
{clinical_manifestations}

### Potential Diagnoses
{potential_diagnoses}

### Output Format Example
{icd_format_example}"""
ICD_EVAL_NO_LAB_PROMPT = """Your task is to predict and rank ICD-10 discharge codes based on admission information.
Respond clearly and concisely. Avoid speculation, filler, or disclaimers.

### Required Output:

1. A brief reasoning process (internal steps to identify diagnoses).
2. A list if potential ICD-Codes.
3. A single JSON object containing all ICD-Codes with reasoning.
4. Stop after printing the JSON object - no extra commentary.

### Structured Reasoning Process
1. Analyze Patient Data:
  - Admission Note
  - Clinical Manifestations
  - Potential Diagnoses
2. Predict many ICD-10 codes:
  - Predict up to 30 icd-codes based on 1. and put them in order of likelihood.
  - Think about all ICD-10 codes the patient could be assigned when leaving the hospital.
  - Go broad and prefer to predict more we can remove unlikely codes later.
  - Print the ICD-10 as a simple list, not in JSON format.
3. Rethink your predicted ICD-10 codes:
  - How are the Manifestations related to your diagnoses?
  - Which ones do not fit, which did you miss?
  - Is every ICD-10 code an official ICD-10 code?
  - Is the amount of ICD-10 codes between 3 and 30?
4. Print your final answer in json format.
  - Rerank the order if needed.
  - Use the format of the Output Format Example.
  - Give a reason for each code, why it is likely to be related to the patient.
  - The json can have 3 - 30 ICD-10 codes.
  - Stop after you printed the json, DO NOT print any other text or JSON.

### Admission Note
{admission_note}

### Clinical Manifestations
{clinical_manifestations}

### Potential Diagnoses
{potential_diagnoses}

### Output Format Example
{icd_format_example}"""
ICD_EVAL_MIMIC_PROMPT = """Your task is to predict and rank ICD-10 discharge codes based on admission information.
Laboratory Results at Admission Time are not given, think about potential Laboratory Results and how they affect the likelyhood of diagnoses.
Respond clearly and concisely. Avoid speculation, filler, or disclaimers.

### Required Output:

1. A brief reasoning process (internal steps to identify diagnoses).
2. A analysis of potential laboratory results.
3. A list if potential ICD-Codes.
4. A single JSON object containing all ICD-Codes with reasoning.
5. Stop after printing the JSON object - no extra commentary.

### Structured Reasoning Process
1. Analyze Patient Data:
  - Admission Note
2. Predict many ICD-10 codes:
  - Predict up to 30 icd-codes based on 1. and put them in order of likelihood.
  - Think about all ICD-10 codes the patient could be assigned when leaving the hospital.
  - Go broad and prefer to predict more we can remove unlikely codes later.
  - Print the ICD-10 as a simple list, not in JSON format.
3. Laboratory Results: Laboratory Results aren't given. Think about potential Laboratory Results at Admission Time and how they affect the likelyhood of ICD-Codes.
4. Rethink your predicted ICD-10 codes:
  - Which ones do not fit, which did you miss?
  - Is every ICD-10 code an official ICD-10 code?
  - Is the amount of ICD-10 codes between 3 and 30?
5. Print your final answer in json format.
  - Rerank the order if needed.
  - Use the format of the Output Format Example.
  - Give a reason for each code, why it is likely to be related to the patient.
  - The json can have 3 - 30 ICD-10 codes.
  - Stop after you printed the json, DO NOT print any other text or JSON.

### Admission Note
{admission_note}

### Output Format Example
{icd_format_example}"""
QUALITATIVE_EVAL_PROMPT = """You received the following instruction:
{v4_prompt}
Based on admission note and laboratory results at admission time you predicted:
{v4_json}
The correct answer is:
{ICD_CODES}
Evaluate the answer and explain why you answers differ from the correct answer.
"""

# AN Notes also include Admission Note
PROMPT_TEMPLATES_GEN = {
    'EXTRACT_PROMPT': EXTRACT_ABDOMINAL_PAIN_PROMPT,
    'DIAGNOSE_PROMPT': DIAGNOSE_MERLIN_PROMPT,
    'RERANK_PROMPT': RERANK_MERLIN_PROMPT,
    'ICD_PROMPT': ICD_MERLIN_PROMPT,
}
PROMPT_TEMPLATES_EVAL_MERLIN = {
    'EXTRACT_PROMPT': EXTRACT_ABDOMINAL_PAIN_PROMPT,
    'DIAGNOSE_PROMPT': DIAGNOSE_EVAL_MERLIN_PROMPT,
    'RERANK_PROMPT': DIAGNOSE_EVAL_MERLIN_PROMPT,
    'ICD_PROMPT': ICD_EVAL_MERLIN_PROMPT,
}
PROMPT_TEMPLATES_EVAL_MIMIC = {
    'EXTRACT_PROMPT': EXTRACT_ABDOMINAL_PAIN_PROMPT,
    'DIAGNOSE_PROMPT': DIAGNOSE_EVAL_MIMIC_PROMPT,
    'RERANK_PROMPT': DIAGNOSE_EVAL_MIMIC_PROMPT,
    'ICD_PROMPT': ICD_EVAL_MIMIC_PROMPT,
}
PROMPT_TEMPLATES_EVAL_NO_LAB = {
    'EXTRACT_PROMPT': EXTRACT_ABDOMINAL_PAIN_PROMPT,
    'DIAGNOSE_PROMPT': DIAGNOSE_EVAL_NO_LAB_PROMPT,
    'RERANK_PROMPT': DIAGNOSE_EVAL_NO_LAB_PROMPT,
    'ICD_PROMPT': ICD_EVAL_NO_LAB_PROMPT,
}
PROMPT_TEMPLATES_MIMIC = {
    'EXTRACT_PROMPT': EXTRACT_PROMPT,
    'DIAGNOSE_PROMPT': DIAGNOSE_MIMIC_PROMPT,
    'RERANK_PROMPT': RERANK_MIMIC_PROMPT,
    'ICD_PROMPT': ICD_MIMIC_PROMPT,
}
EXTRACT_PROMPT_DICT = {
    'abdominal_pain': EXTRACT_ABDOMINAL_PAIN_PROMPT,
    'generic': EXTRACT_PROMPT,
}