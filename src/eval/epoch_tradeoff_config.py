"""Plot config for the epoch trade-off figure (src/eval/epoch_tradeoff_plots.py).

Appendix figure backing the checkpoint-selection claim in the paper: selecting
the epoch that maximizes ICD macro-F1 costs upstream structured output and
diagnosis ranking, most sharply at 14B.

All knobs live here (rows, metrics, colors, labels) so the plotting script
stays free of hardcoded run names -- same convention as
src/eval/robustness_plot_config.py and src/eval/ablation_plot_config.py.

Rows are DEV-split epoch sweeps of the full fine-tuning runs on the headline
dataset variant (thrfull + ICD oversampling; the 0.6B run predates the
suffixed naming and is just `06b-full-eN`). Dev, not test, because this is
the sweep the paper actually selects the checkpoint on.
"""

# --- rows: {size label: (row-name template, epochs)} -----------------------
EPOCHS = [1, 2, 3, 4]

ROW_TEMPLATES = {
    "0.6B": "06b-full-e{e}",
    "8B": "8b-full-thrfull-icd2-e{e}",
    "14B": "14b-full-thrfull-icd2-e{e}",
    "32B": "32b-full-thrfull-icd2-e{e}",
}

# Epoch selected for the paper's main table, per size (dev-selected by
# ICD F1 Macro). Marked with a ring in the plot.
SELECTED_EPOCH = {"0.6B": 4, "8B": 3, "14B": 3, "32B": 3}

# --- metrics: (csv column, panel title, y-axis label) ----------------------
# Recall@3 rather than Recall@1: the pipeline's acceptance gate is Recall@1,
# but @3 is what the paper reports for ranking quality (see body.tex), and
# @1 at 32B is dominated by list-length effects.
# Diagnosis-stage JSON validity added: it is the mechanism
# behind the 14B Recall@3 collapse. A trace whose diagnosis stage emits
# unparseable JSON has no ranked list at all, so it scores 0 on Recall@k --
# the ranking drop is largely an output-format failure, not the model losing
# diagnostic knowledge. Showing V2 validity next to V2 Recall@3 makes that
# readable straight off the figure.
#
# "V1 JSON Valid Rate" (symptom stage) was in this list briefly the same day
# and removed again: the figure exists to explain the *ranking* drop,
# and the symptom stage is not on that causal path -- it only added a fourth
# panel whose 14B line goes to zero for unrelated reasons. The symptom-stage
# numbers are still quoted in the appendix text where they belong.
METRICS = [
    ("ICD F1 Macro", "ICD macro-F1 (selection metric)", "%"),
    ("V2 Recall@3", "Diagnosis ranking, Recall@3", "%"),
    ("V2 JSON Valid Rate", "Diagnosis-stage JSON validity", "%"),
]

# --- colors ---------------------------------------------------------------
# One color per model size, ordered small -> large. Sequential so the reader
# reads size off the line without checking the legend.
SIZE_COLORS = {
    "0.6B": "#B7C9D9",
    "8B": "#7FA8C9",
    "14B": "#DD8452",
    "32B": "#4D4D4D",
}
SIZE_MARKERS = {"0.6B": "o", "8B": "s", "14B": "^", "32B": "D"}

FIGSIZE = (9.0, 2.8)
