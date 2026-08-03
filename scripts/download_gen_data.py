#!/usr/bin/env python3
"""
Download generated data artifacts from wandb project anon-entity/merlin-generation.

Edit the CONFIG block below, then run:
    python scripts/download_gen_data.py
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.postprocessing.download_gen_data import download, resolve_api_key

# ── CONFIG ────────────────────────────────────────────────────────────────────

# wandb API key — or leave as None to read from config.json / WANDB_API_KEY env var
API_KEY = None

# Path to config.json that contains {"WANDB_API_KEY": "..."}
CONFIG_JSON = "config.json"

# Where to save downloaded files
OUTPUT_DIR = "./data/results/gen_data"

# Set to True to list matching runs without downloading anything
DRY_RUN = False

# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    os.environ["WANDB_API_KEY"] = resolve_api_key(API_KEY, CONFIG_JSON)
    download(output_dir=OUTPUT_DIR, dry_run=DRY_RUN)
