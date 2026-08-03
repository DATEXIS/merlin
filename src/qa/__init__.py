"""Two-stage qualitative error analysis (QA) for ICD-code predictions.

Stage 1 (taxonomy induction, `taxonomy.py`): an open-source judge reads each
mispredicted case (admission prompt + prediction + gold codes + discharge note)
and writes free-text root-cause notes. These are consolidated -- by the judge and
then by a human -- into a small, fixed error taxonomy. Runs on the dev split.

Stage 2 (quantification, `classify.py`): with the taxonomy frozen, the judge makes
one independent yes/no decision per error class per case, over the full target
split (test). Output is a reproducible per-case boolean matrix that we aggregate
into error-shift tables comparing base vs. fine-tuned models.

Everything is driven by `scripts/qa_config.yaml`; the CLI only picks the stage.
See `scripts/qa_analysis.py`.
"""
