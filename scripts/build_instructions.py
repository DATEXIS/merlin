#!/usr/bin/env python3
"""
Build the instruction dataset from merged per-model cc files.

Input:  one <model>.pq per model in GEN_DATA_DIR  (output of scripts/merge_gen_data.py)
Output: a single combined instructions parquet file at OUTPUT_PATH

Run:
    python scripts/build_instructions.py
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.postprocessing.build_instructions import build_all_instructions

# ── CONFIG ────────────────────────────────────────────────────────────────────

GEN_DATA_DIR = "data/results/gen_data"
OUTPUT_PATH = "data/results/instructions.pq"

# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    build_all_instructions(gen_data_dir=GEN_DATA_DIR, output_path=OUTPUT_PATH)
