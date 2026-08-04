#!/usr/bin/env python3
"""Dataset-level analyses -- figures about the underlying data itself (ICD
code long-tail distribution, etc.), as opposed to the checkpoint/eval-result
analyses in scripts/eval_analysis.py. No CLI arguments: parameters for each
analysis live in the ICD_LONGTAIL config dict below -- edit it directly
instead of passing flags.

Each analysis is its own module under src/eval/dataset_analyses/; this
script just calls them and saves figures to figures/dataset_analyses/
(repo-relative, alongside the existing bars_8b_full/lora dataset-ablation
figures).

Usage:
    python scripts/dataset_analyses.py
"""
import sys
from pathlib import Path

# Make `src...`/`paper...` importable no matter where the script is called from.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.eval.colors import BLUE, YELLOW, tint

# --- ICD 3-digit long-tail -- edit these to tweak the figure -----------------
# Single-model eval.pq (one row per hadm_id); non-"mimic" filename = Merlin.
ICD_LONGTAIL = {
    "input": str(REPO_ROOT / "data/results/gen_data/Qwen3-32B.pq"),
    "digits": 3,
    "split_fraction": 0.8,        # "80% of total samples" cutoff, matches the printed reference
    # The house palette are quite strong at full
    # intensity for a dense bar plot -- tinted toward white here. Swap in
    # RED/TEAL, or change the tint() amount, if these feel off.
    "majority_color": tint(BLUE, 0.5),
    "tail_color": tint(YELLOW, 0.1),
    "out_stem": "icd_3digit_longtail",
}


def main():
    from src.eval.dataset_analyses import icd_longtail
    print("\n--- ICD 3-digit long-tail ---")
    icd_longtail.run(ICD_LONGTAIL)


if __name__ == "__main__":
    main()
