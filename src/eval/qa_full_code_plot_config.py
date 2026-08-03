"""Plot config for the ICD full-code specificity figure
(src/eval/qa_full_code_plot.py). Same knob-file convention as
main_figure_config.py / ablation_plot_config.py.

2x2 grid: columns are the two granularities computed in scripts/qa_full_code.py
('category' = 3-char, the level used everywhere else in the paper; 'full' =
exact code), rows are the two aggregates the paper already distinguishes
everywhere else -- micro-F1 (pools counts across codes, dominated by frequent
ones) vs. macro-F1 (per-code F1 averaged unweighted, so rare codes count as
much as common ones). A 'chapter' (2-char) column existed in an earlier
version of this figure; cut 2026-07-27 to keep the comparison to what
the paper reports (category) plus the new thing (full code).

Colors are imported from checkpoint_plots rather than redefined, so this
figure's "grey = untuned base, orange = + MERLIN full FT" reads identically to
every other bar chart in the paper.
"""
from src.eval.checkpoint_plots import BASE_COLOR, FULL_COLOR  # noqa: F401

VALUE = "f1"          # "recall" | "precision" | "f1" -- column in the long CSV to plot

METRIC_TYPES = ["micro", "macro"]           # grid rows
METRIC_TYPE_LABELS = {"micro": "Micro", "macro": "Macro"}

PANEL_ORDER = ["category", "full"]          # grid columns
PANEL_TITLES = {
    "category": "Category (3-char)",
    "full": "Full code",
}

# qa_full_code.py keys sizes "06b"/"8b"/"14b"/"32b" (matches qa_deterministic.py),
# which is NOT the "0.6b" convention checkpoint_plots._size_sort_key expects --
# so this figure uses its own fixed order/labels instead of importing that
# helper, rather than silently mis-sorting or mis-parsing "06b" as 6B.
SIZE_ORDER = ["06b", "8b", "14b", "32b"]
SIZE_LABELS = {"06b": "0.6B", "8b": "8B", "14b": "14B", "32b": "32B"}

BAR_W = 0.38
DPI = 300
OUT_STEM = "icd_specificity_by_size"
