"""Plot config for the head/body/tail long-tail figure
(src/eval/longtail_strata_plot.py). Same knob-file convention as
qa_full_code_plot_config.py / main_figure_config.py.

2 rows x 3 cols: rows are F1 / Recall (the two columns already computed per
stratum in src/eval/longtail_strata.py -> longtail_strata.csv), columns are
head / body / tail (src/eval/longtail_config.py::STRATA, ordered many-shot ->
few-shot). Each cell is the same base-vs-full-FT-per-size grouped bar layout
as qa_full_code_plot.py / main_figure_plots.py.

Colors are imported from checkpoint_plots rather than redefined, so this
figure's "grey = untuned base, orange = + MERLIN full FT" reads identically
to every other bar chart in the paper.
"""
from src.eval.checkpoint_plots import BASE_COLOR, FULL_COLOR  # noqa: F401

VALUE_TYPES = ["f1", "recall"]                    # grid rows
VALUE_TYPE_LABELS = {"f1": "F1", "recall": "Recall"}

# Matches src.eval.longtail_config.STRATA order (many-shot -> few-shot).
PANEL_ORDER = ["head", "body", "tail"]             # grid columns
PANEL_TITLES = {"head": "Head (>=100 train)", "body": "Body (10-99 train)",
                "tail": "Tail (<10 train)"}

# longtail_strata.csv keys sizes "0.6B"/"8B"/"14B"/"32B" (matches
# longtail_config.CHECKPOINTS display names) -- already display-ready, so no
# separate label map is needed, only an explicit order (string-sorting would
# put "14B"/"32B" before "8B").
SIZE_ORDER = ["0.6B", "8B", "14B", "32B"]

BAR_W = 0.38
DPI = 300
OUT_STEM = "longtail_head_body_tail"
