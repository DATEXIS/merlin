"""Plot config for the seed-robustness figure (src/eval/robustness_plots.py).

This is the general, easy-to-edit knob file for that figure -- metrics AND
colors both live here, so nothing plot-related is hardcoded inside the
plotting script itself. Metric catalogue is reused from
src/eval/plot_common.py's ALL_METRICS (single source of truth -- see
that file to add a new metric column) so "same metrics as the other bar
plots" never drifts into a second, divergent copy.

To change which metrics the robustness bars show: copy any subset of the
names below into METRICS_TO_PLOT, in the order you want them plotted.

All metric names available (kept in sync with plot_common.ALL_METRICS):
    ICD F1 Macro, ICD F1 Micro, V2 F1 Macro@1, V2 F1 Micro@1, V2 MRR,
    V2 Accuracy@1, V2 Recall@1, V2 Recall@3, V2 Recall@5, V2 Recall@10,
    CosSim, DotProduct, V1 JSON Valid Rate,
    V2 JSON Valid Rate, V4 JSON Valid Rate
"""
from src.eval.plot_common import ALL_METRICS, build_metrics, FULL_COLOR, LORA_COLOR

# The active selection. Ranking is reported as Recall@3, matching the paper
# body: the V2 F1@1 / MRR variants were dropped from the
# paper because the pipeline's acceptance gate is effectively Recall@1, so a
# graded ranking score implies an objective the pipeline does not optimize.
# DotProduct dropped: not needed on this figure. Validated
# against ALL_METRICS by build_metrics() below.
METRICS_TO_PLOT = ["ICD F1 Macro", "ICD F1 Micro", "V2 Recall@3"]

METRICS = build_metrics(METRICS_TO_PLOT)  # [(name, ylabel), ...]

# --- colors ---------------------------------------------------------------
# Bar color: reuse the same "full fine-tune" color used everywhere else
# (plot_common.FULL_COLOR) so this figure matches the
# rest of the paper's palette instead of introducing a new one.
BAR_COLOR = FULL_COLOR

# Per-seed scatter overlay: seed 42 is the one already reported in the paper
# (marked with a star), 43/44 are the new robustness-check seeds. Used by the
# single-arm figures (plot_robustness), where there's only one bar color so
# color is free to encode seed.
SEED_COLORS = {42: "#4D4D4D", 43: "#4C72B0", 44: "#55A868"}
SEED_MARKERS = {42: "*", 43: "o", 44: "o"}
PAPER_SEED = 42

# --- combined (grouped-bar) figure -----------------------------------------
# One bar color per ARM instead of per seed (plot_robustness_grouped): reuse
# the existing full-FT/LoRA pair so this figure stays in the paper's
# established palette rather than inventing a third color. "LoRA" doesn't
# mean anything in this figure -- it's just repurposed as "arm 2's color",
# always labelled by arm name in the legend, never "LoRA".
ARM_COLORS = {"thrfull2icd": FULL_COLOR, "mimic-icd2": LORA_COLOR}

# Seed markers here must be SHAPE-only (no color) since color is taken by
# arm -- all three seeds render in the same neutral black-edge/white-fill
# style (matching the error-bar halo look elsewhere in this figure) so they
# stay legible over either bar color, distinguished purely by marker shape.
GROUPED_SEED_MARKERS = {42: "*", 43: "^", 44: "s"}
GROUPED_SEED_MARKER_COLOR = "white"
GROUPED_SEED_MARKER_EDGE = "#1A1A1A"
