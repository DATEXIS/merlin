"""Config for the appendix generation-dynamics figure
(src/eval/generation_plots.py).

Recreates paper/*/figures/generation_plots.pdf (\\label{fig:generation_plots}),
"Iterative improvement of verifier scores across generation-budget steps
(abdominal-pain cases)". The script that originally produced this figure was
never checked in, so this rebuilds it from the three generators' run logs
(Jan, 2026-08-01). Same knob-file convention as robustness_plot_config.py /
epoch_tradeoff_config.py: edit this file, not generation_plots.py.

Data provenance -- read this before trusting a number out of SCORES below:
most points are "V{n} end k/4 with Score: ..." lines lifted verbatim from the
run logs. A handful of points are missing because a log was truncated or a
round's final line never printed; those are eyeballed off the shipped PDF
instead and marked (E) in the comments. This whole figure is illustrative of
the generation-budget dynamic, not a source of truth -- don't cite SCORES
values as reported results.

MRR label (Jan, 2026-08-01): considered relabeling V2/V3 to "Recall@3" since
we never saved the underlying ranked lists for these exploratory runs, only
the scalar "Score" the pipeline logged per round -- so a real Recall@3 or MRR
recompute isn't possible from what survives. Decided against it: Recall@3
and MRR aren't the same quantity (MRR credits rank position, Recall@3 is a
flat in/out-of-top-3 threshold), so swapping the label to a different named
metric would carry the same "did we actually verify this" problem, just
under a more technical-sounding name. Kept "MRR" -- matches the original
figure, and the SCORES values were never a real MRR recompute either way.

Design pass (Jan, 2026-08-01): switched to match Figure 7's style
(src/eval/epoch_tradeoff_plots.py / epoch_tradeoff_config.py) rather than the
lost original -- same family of figure (line-per-model/size over an ordinal
training/budget axis), so it should read as the same visual language:
percent-scale y-axis grounded at 0 (ax.set_ylim(bottom=0), "%" everywhere,
metric identity moved into the panel title instead), a distinct marker shape
per line in addition to color (readable in grayscale / for colorblind
readers), solid low-alpha gridlines instead of dashed, and the same
title/label/legend font sizes. Line charts themselves are the right choice
here, same reasoning as Figure 7: x is an ordered discrete step (generation
round, like training epoch there), and the point is the trajectory across
it, which a bar chart per step would obscure.
"""

STEPS = [1, 2, 3, 4]

# stage -> panel title (metric name folded in, y-axis is a uniform "%" --
# see design-pass note above)
STAGE_TITLES = {
    "V1": "V1: Symptoms Extraction (NormDot)",
    "V2": "V2: Diagnoses Prediction (MRR)",
    "V3": "V3: Lab-based Reranking (MRR)",
    "V4": "V4: ICD Codes Prediction (F1 Micro)",
}
STAGE_ORDER = ["V1", "V2", "V3", "V4"]
XLABEL = "generation-budget step"
YLABEL = "%"

# Colors reused from checkpoint_plots.py's seaborn-deep hexes (COLORS
# "icd2"/"thrfull-icd2"/"mimic") rather than inventing a new palette -- also
# happens to match the original figure's blue/orange/green. Markers follow
# Figure 7's convention of one distinct shape per line, on top of color.
MODEL_COLORS = {
    "Qwen3-32B": "#4C72B0",
    "medgemma-27b-it": "#DD8452",
    "Llama-3.3-70B-Instruct": "#55A868",
}
MODEL_MARKERS = {
    "Qwen3-32B": "o",
    "medgemma-27b-it": "^",
    "Llama-3.3-70B-Instruct": "s",
}
MODEL_ORDER = ["Qwen3-32B", "medgemma-27b-it", "Llama-3.3-70B-Instruct"]

# model -> stage -> [score at step 1, 2, 3, 4]
SCORES = {
    "Qwen3-32B": {
        # all 4 stages fully present in the qwen log
        "V1": [0.690, 0.693, 0.695, 0.696],
        "V2": [0.719, 0.740, 0.748, 0.756],
        "V3": [0.749, 0.761, 0.769, 0.776],
        "V4": [0.499, 0.520, 0.530, 0.538],
    },
    "medgemma-27b-it": {
        "V1": [0.186, 0.306, 0.415, 0.511],
        "V2": [0.511, 0.695, 0.763, 0.795],
        # log only has step 1 (0.524); steps 2-4 (E) read off the PDF
        "V3": [0.524, 0.680, 0.755, 0.793],
        # log only has steps 3-4 (0.409, 0.436); steps 1-2 (E) read off the PDF
        "V4": [0.260, 0.365, 0.409, 0.436],
    },
    "Llama-3.3-70B-Instruct": {
        "V1": [0.708, 0.712, 0.714, 0.716],
        "V2": [0.743, 0.760, 0.768, 0.775],
        "V3": [0.769, 0.776, 0.785, 0.795],
        # log is cut off before the final "end 4/4" line; step 4 (E) read off the PDF
        "V4": [0.411, 0.445, 0.466, 0.480],
    },
}

FIGSIZE = (6.4, 5.8)
DPI = 300
OUT_STEM = "generation_plots"
