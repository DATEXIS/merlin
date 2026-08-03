"""Plot config for the 8B data-construction ablation figure
(src/eval/ablation_plots.py).

Same convention as robustness_plot_config.py: this is the knob file --
metrics, grouping, labels and colors all live here, nothing plot-related is
hardcoded in the plotting script.

The figure re-cuts the 8B full-FT dataset ablation as a 2x factorial rather
than a flat bar-per-variant chart (which is what
paper_plots.plot_dataset_analyses / the old ad-hoc 8b_ablation.py drew, and
which made the actual finding invisible):

    trace quality filter  x  ICD-step oversampling

    family              | no oversampling | ICD rows x2
    --------------------|-----------------|-------------
    label-only          | mimic           | mimic-icd2
    all traces          | replace         | icd2
    half thresholds     | thrhalf         | --
    verifier thresholds | thrfull         | thrfull-icd2

Reading down a column is the paper's headline ablation claim: keeping only
traces that clear their verifier threshold monotonically improves
rare-diagnosis (macro) ICD F1, at BOTH oversampling settings. `thrfulldrop`
(same thresholds, but below-threshold cases dropped instead of replaced by
the label-only fallback) is drawn as a hatched control inside the
verifier-threshold family.

Metric catalogue is reused from src/eval/checkpoint_plots.py's ALL_METRICS
(single source of truth for metric name -> axis label).
"""
from src.eval.checkpoint_plots import ALL_METRICS, build_metrics, BASE_COLOR, LORA_COLOR, FULL_COLOR

# --- what to plot ---------------------------------------------------------
# ICD F1 Macro first: it is the paper's primary metric (rare/long-tail
# diagnoses). See ablation_plots.py for why no V1/CosSim panel is shown.
# Two panels, not three: macro vs. micro IS the argument of this figure
# (filtering buys the long tail; oversampling buys the frequent codes).
# Diagnosis Recall@3 is flat across variants and lives in the appendix table.
METRICS_TO_PLOT = ["ICD F1 Macro", "ICD F1 Micro"]
METRICS = build_metrics(METRICS_TO_PLOT)  # [(name, ylabel), ...]

# Panel titles override the raw metric names, so the figure speaks the
# paper's clinical vocabulary (symptoms / diagnoses / ICD codes) instead of
# the pipeline's internal V1..V4 stage names.
# (Jan, 2026-08-03) No "(long-tail)" / "(frequent diagnoses)" qualifiers:
# they pre-announce an interpretation the panels themselves don't show, and
# the macro-vs-micro contrast is made in the caption. Titles now match the
# headline figure (main_figure_plots_macro.py).
PANEL_TITLES = {
    "ICD F1 Macro": "ICD codes: macro-F1",
    "ICD F1 Micro": "ICD codes: micro-F1",
    "V2 Recall@3": "Diagnoses: Recall@3",
}

# Axis labels, likewise in the paper's vocabulary rather than the internal
# stage-prefixed metric names.
PANEL_YLABELS = {
    "ICD F1 Macro": "macro-F1 (%)",
    "ICD F1 Micro": "micro-F1 (%)",
    "V2 Recall@3": "Recall@3 (%)",
}

SIZE = "8b"      # only size with a real dataset ablation
MODE = "full"    # full fine-tuning only -- a LoRA bar here would conflate
                 # data construction with training mode (see revision notes)

# --- the factorial layout -------------------------------------------------
# (family label, dataset without ICD oversampling, dataset with ICD x2)
FAMILIES = [
    ("Label-only\n(no traces)", "mimic", "mimic-icd2"),
    ("All traces\n(unfiltered)", "replace", "icd2"),
    ("Half\nthresholds", "thrhalf", None),
    ("Full\nthresholds", "thrfull", "thrfull-icd2"),
]

# Extra control drawn inside a family's slot as a hatched bar:
#   dataset -> (family index, which column, short annotation)
CONTROLS = {
    "thrfulldrop": (3, "plain", "drop"),
}

# --- colors ---------------------------------------------------------------
# Reuse the paper's one base/lora/full palette (src/eval/checkpoint_plots.py)
# instead of this figure's own pastel blue/orange pair, so every bar chart in
# the paper reads off the same three colors. PLAIN_COLOR (no ICD
# oversampling) takes the LORA slot -- it's the secondary/non-headline
# condition here, same role LORA_COLOR plays in the size-comparison figure.
# ICD2_COLOR (the headline recipe once thresholds+oversampling both apply)
# already matched FULL_COLOR by coincidence; now it's imported so it can't
# drift from it. BASE_COLOR likewise imported rather than redefined locally.
PLAIN_COLOR = LORA_COLOR      # no ICD oversampling
ICD2_COLOR = FULL_COLOR       # ICD rows duplicated
CONTROL_COLOR = "#C9C9C9"     # hatched control bars
HIGHLIGHT_EDGE = "#1F1F1F"    # outline on the headline recipe

HEADLINE_DATASET = "thrfull-icd2"  # gets the highlight outline

# --- misc -----------------------------------------------------------------
SPLIT = "dev"        # ablation is reported on dev (test is reserved for the
                     # headline table); see paper body.
# Print bar values on both panels. The three-bar family (Full thresholds)
# is the tightest fit -- if labels collide there, shrink label fontsize or
# rotate before dropping labels again.
VALUE_LABEL_METRICS = ["ICD F1 Macro", "ICD F1 Micro"]
DPI = 300
OUT_STEM = "dataset_ablation_8b_factorial"
