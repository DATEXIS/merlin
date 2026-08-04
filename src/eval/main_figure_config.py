"""Config for the paper's headline results figure
(src/eval/main_figure_plots_macro.py). Same knob-file convention as
robustness_plot_config.py / ablation_plot_config.py.

Three panels:
ICD macro-F1 is the paper's primary metric -- it is the rare/long-tail
diagnosis read, which is what a "don't overlook the rare diagnosis" tool is
actually evaluated on -- with ICD micro-F1 beside it for the frequent-code view,
and diagnoses Recall@3 as a third panel now that the 14B eval-repeat outlier
behind V2's earlier instability is understood (a single low-JSON-validity
eval run, not a reproducible per-size regression -- \S\ref{sec:ranking}
explains it), so the panel no longer reads as unexplained noise. Symptom
extraction stays in the main results TABLE only.

To go back to two panels, drop "V2 Recall@3" from METRICS_TO_PLOT.
"""
from src.eval.plot_common import ALL_METRICS, build_metrics  # noqa: F401

METRICS_TO_PLOT = ["ICD F1 Macro", "ICD F1 Micro", "V2 Recall@3"]
METRICS = build_metrics(METRICS_TO_PLOT)

PANEL_TITLES = {
    "ICD F1 Macro": "ICD codes: macro-F1",
    "ICD F1 Micro": "ICD codes: micro-F1",
    "V2 Recall@3": "Diagnoses: Recall@3",
}
PANEL_YLABELS = {
    "ICD F1 Macro": "macro-F1 (%)",
    "ICD F1 Micro": "micro-F1 (%)",
    "V2 Recall@3": "Recall@3 (%)",
}

# Reference line for the strongest untuned external competitor, matched as a
# substring against the `collection` column of checkpoint_metrics.csv. Set to
# None: in a figure whose one job is base-vs-fine-tuned
# across scale, a third horizontal line for a model that is on neither axis
# reads as clutter -- the competitor comparison lives in the main table and
# is made in the text.
EXTERNAL_REFERENCE = None
EXTERNAL_LABEL = "Baichuan-M2-32B (untuned)"

# --- style ----------------------------------------------------------------
BASE_COLOR = "#4D4D4D"
FULL_COLOR = "#DD8452"
REF_COLOR = "#2F6690"
BAR_W = 0.38
DPI = 300
OUT_STEM = "main_icd_by_size"
