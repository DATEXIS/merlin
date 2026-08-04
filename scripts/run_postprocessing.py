#!/usr/bin/env python3
"""
Full postprocessing pipeline — run this after the generation pipeline is done.

Steps
-----
1. Merge CC files Per model, merge all cc_*.pq into one <model>.pq and
                       deduplicate subject_ids across chief complaints within
                       that model (keep the sample from the smallest CC).
                       Also (re)writes data/preprocessed_mimic/subject_complaint_map.json
                       from that same cross-model assignment, so it's always
                       in sync with what actually got merged.

2. Combine models Concatenate all per-model files into combined.pq with
                       a 'gen_model' column. Each subject_id appears ~3× (once
                       per model). Only the relevant columns are kept.

3. Build instructions Build the flat instruction dataset (one row per
                       patient × verifier step) and write a single parquet
                       ready for fine-tuning.

4. MIMIC instructions Build the MIMIC instruction dataset variants: one row
                       per patient x verifier step (V1-V4), MIMIC-style direct
                       prompts (no reasoning-chain scaffolding), pure
                       ground-truth outputs, no Thinking column. Built from
                       combined.pq (same patients as MeRLIn).

5. Test dataset Build the downstream-eval test_dataset from combined.pq:
                       one row per patient, train split excluded. This is what
                       the dataset-upload step uploads as the
                       'test_dataset' artifact and what eval configs' client
                       loads (file_name: "test_dataset").

Prerequisites
-------------
The per-model, per-CC raw generation files must already be present under
GEN_DATA_DIR as:
    <GEN_DATA_DIR>/<model>/cc_<chief_complaint>.pq

Download them first using:
    python scripts/download_gen_data.py
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.postprocessing.merge_gen_data import merge_all, combine_models
from src.postprocessing.build_instructions import (
    build_instruction_dataset,
    GEN_THRESHOLDS,
    GEN_THRESHOLDS_HALF,
)
from src.postprocessing.build_mimic_instructions import build_mimic_instructions
from src.postprocessing.build_test_dataset import build_test_dataset

# ── CONFIG ────────────────────────────────────────────────────────────────────

# Directory with per-model subdirs containing cc_*.pq files
GEN_DATA_DIR = "data/results/gen_data"

# Output path for the combined model data
COMBINED_OUTPUT = "data/results/gen_data/combined.pq"

# Where all instruction datasets are written.
INSTR_DIR = "data/results/instructions"

# Rebuilt from the cross-model cc assignment every time Step 1 runs.
COMPLAINT_MAP_PATH = "data/preprocessed_mimic/subject_complaint_map.json"

# Teacher model used for the single-model MeRLIn variants (Qwen won the ablation).
# best-per-id uses all models and picks the highest-scoring one per (patient, step).
MERLIN_MODELS = ["Qwen3-32B", "Llama-3.3-70B-Instruct", "medgemma-27b-it"]

# The six instruction datasets to emit. Every MeRLIn variant sends nan/0-score
# rows (and, where a threshold is set, below-threshold rows) to the mimic-style
# no-reasoning fallback instead of dropping them.
# name -> (thresholds, models, best_per_id, duplicate_verifiers)
MERLIN_VARIANTS = {
    # 2. Only nan/0 rows lose their trace; everything else keeps reasoning.
    "merlin_replace":       dict(thresholds=None,                 models=MERLIN_MODELS, best_per_id=False),
    # 3. Generation thresholds [0.6, 0.5, 0.8, 0.55].
    "merlin_thr_full":      dict(thresholds=GEN_THRESHOLDS,       models=MERLIN_MODELS, best_per_id=False),
    # 4. Halved thresholds [0.3, 0.25, 0.4, 0.275].
    "merlin_thr_half":      dict(thresholds=GEN_THRESHOLDS_HALF,  models=MERLIN_MODELS, best_per_id=False),
    # 5. Best model per (hadm_id, verifier); still replace nan/0 with fallback.
    "merlin_best_per_id":   dict(thresholds=None,                 models=None,          best_per_id=True),
    # 6. Ablation: same as merlin_replace (no threshold filtering), but every
    # patient gets a second V4 (ICD-prediction) row -- oversamples that step.
    "merlin_icd_2x":        dict(thresholds=None,                 models=MERLIN_MODELS, best_per_id=False,
                                  duplicate_verifiers=(4,)),
    # 7. Combines the two variants that improved fine-tuning individually:
    # thr_full's generation thresholds [0.6, 0.5, 0.8, 0.55] AND icd_2x's V4
    # (ICD-prediction) row duplication, together.
    "merlin_thrfull_icd2x": dict(thresholds=GEN_THRESHOLDS,       models=MERLIN_MODELS, best_per_id=False,
                                  duplicate_verifiers=(4,)),
    # 8. Ablation: same thresholds as thr_full, but below-threshold/nan/0 rows
    # are DROPPED instead of replaced by the mimic-style fallback. Not
    # expected to beat thr_full (loses the below-threshold signal entirely and
    # skews verifier-step balance -- V4/ICD rows are hit hardest, ~61% dropped
    # vs ~24-29% for V1-V3 on the v1.3 data), but it isolates what the
    # fallback is actually contributing.
    "merlin_thrfull_drop": dict(thresholds=GEN_THRESHOLDS,        models=MERLIN_MODELS, best_per_id=False,
                                  drop_below_threshold=True),
}

# New MIMIC instruction dataset variants (full V1-V4, ground-truth, MIMIC-style
# direct prompts -- see src/postprocessing/build_mimic_instructions.py). Kept
# under separate names from the original mimic_instructions.pq (single-stage,
# V4-only) so that dataset is never overwritten -- it's already uploaded/used
# by ft-qwen3-8b-lora-r128-mimic-v1-3 (never overwrite a prior
# version's folder; old checkpoints' manifests may still reference it).
# name -> (duplicate_verifiers,)
MIMIC_VARIANTS = {
    # "Fixed" mimic: full V1-V4 (the old mimic_instructions was V4/ICD only).
    "mimic_fixed":          dict(duplicate_verifiers=()),
    # icd_2x-style ablation on top of the fixed mimic: every patient gets a
    # second V4 row.
    "mimic_fixed_icd_2x":   dict(duplicate_verifiers=(4,)),
}

# Output path for the downstream-eval test_dataset. Fixed filename (not
# per-round-versioned like the instruction datasets) — the dataset-upload step
# hardcodes this path and creates a new wandb artifact VERSION each time it's
# uploaded, so history is preserved there, not via local filenames.
TEST_DATASET_OUTPUT = "data/results/test_dataset.pq"

# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("Step 1 — Merge CC files per model")
    print("=" * 60)
    merge_all(gen_data_dir=GEN_DATA_DIR, complaint_map_path=COMPLAINT_MAP_PATH)

    print("=" * 60)
    print("Step 2 — Combine all models into one dataset")
    print("=" * 60)
    combine_models(gen_data_dir=GEN_DATA_DIR, output_path=COMBINED_OUTPUT)

    print("=" * 60)
    print("Step 3 — Build MeRLIn instruction datasets")
    print("=" * 60)
    for name, cfg in MERLIN_VARIANTS.items():
        print(f"\n── {name}")
        build_instruction_dataset(
            gen_data_dir=GEN_DATA_DIR,
            output_path=f"{INSTR_DIR}/{name}.pq",
            **cfg,
        )

    print("=" * 60)
    print("Step 4 — Build MIMIC instruction datasets (from combined.pq)")
    print("=" * 60)
    for name, cfg in MIMIC_VARIANTS.items():
        print(f"\n── {name}")
        build_mimic_instructions(
            combined_path=COMBINED_OUTPUT,
            output_path=f"{INSTR_DIR}/{name}.pq",
            **cfg,
        )

    print("=" * 60)
    print("Step 5 — Build downstream-eval test_dataset (from combined.pq)")
    print("=" * 60)
    build_test_dataset(combined_path=COMBINED_OUTPUT, output_path=TEST_DATASET_OUTPUT)

    print("=" * 60)
    print("Postprocessing complete.")
    print("=" * 60)
