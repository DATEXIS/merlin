"""Figures about the underlying dataset itself (ICD code distribution, note
length, etc.) -- as opposed to checkpoint/eval-result analyses, which live in
the sibling modules of src/eval/ (checkpoint_metrics,
plot_common). Each analysis is its own module here; scripts/dataset_analyses.py
is the CLI that dispatches to them by name, driven by the DatasetAnalyses
block of scripts/eval_config.yaml the same way scripts/eval_analysis.py is
driven by its Analysis block.
"""
