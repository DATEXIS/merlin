#!/usr/bin/env python3
"""Generate the paper's LaTeX result tables straight from the result CSVs.

    python -m src.eval.paper_tables

Writes into paper/EACL_2026_v2/:
    table_main.tex      held-out test split, mean +- std over eval seeds
    table_ablation.tex  8B data-construction ablation (dev split)
    table_qa.tex        deterministic ICD error decomposition (test split)

Why generate rather than hand-write: the v1 tables were typed by hand and
drifted from the CSVs twice (the thrfulldrop relabel and the seed-43/44
sweep both changed numbers that stayed stale in the .tex). Everything here
reads the same files the figures read, so a table can no longer disagree
with a plot.

Sources
-------
data/eval_metrics_merlin-eval-1.4.csv
    the union of every eval collection, including ones that never made it
    into checkpoint_metrics.csv (the seed-43/44 reruns whose doubled
    `-seed{N}-seed{N}` suffix misses FAMILY_RE, and the one-off label-only
    and external-model test evals). Row key is the collection name.
data/results/evaluation/results_dev.csv + checkpoint_best_epochs.csv
    the 8B dataset ablation at each variant's dev-selected best epoch.
data/qa/deterministic_rates.csv
    scripts/qa_deterministic.py output; no LLM judge is used any more.

Seeds: rows whose collection differs only by a `-seed43`/`-seed44` suffix
are averaged and reported as mean +- sample std. These are EVAL repeats of
one trained checkpoint, so they capture decoding/verifier stochasticity but
NOT training-seed variance -- the separate 3-training-seed run is reported
in robustness_summary.csv and must be labelled differently in the text.
"""
import re
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
EVAL_CSV = REPO_ROOT / "data" / "eval_metrics_merlin-eval-1.4.csv"
DEV_CSV = REPO_ROOT / "data" / "results" / "evaluation" / "results_dev.csv"
BEST_EPOCHS_CSV = REPO_ROOT / "data" / "results" / "evaluation" / "checkpoint_best_epochs.csv"
QA_CSV = REPO_ROOT / "data" / "qa" / "deterministic_rates.csv"
OUT_DIR = REPO_ROOT / "paper" / "EACL_2026_v3"

SEED_SUFFIX = re.compile(r"-seed4[34]")

# JSON validity in the main table is the mean over the three JSON-emitting
# stages, same definition as the v1 table.
JSON_COLS = ["V1 JSON Valid Rate", "V2 JSON Valid Rate", "V4 JSON Valid Rate"]


def _load_eval() -> pd.DataFrame:
    df = pd.read_csv(EVAL_CSV)
    df = df.rename(columns={df.columns[0]: "collection"})
    df["JSON Valid"] = df[JSON_COLS].mean(axis=1)
    df["group"] = df["collection"].str.replace(SEED_SUFFIX, "", regex=True)
    return df


# Placeholder printed in the +- slot of a row that still has only one eval
# seed, so a pending rerun is visible in the typeset table rather than
# silently absent (per Jan, 2026-07-24: "placeholders for pending externals").
PENDING_STD = "{\\scriptsize\\textcolor{gray}{$\\pm$--}}"


def _cell(sub: pd.DataFrame, metric: str, digits: int = 1, with_std: bool = True,
          bold: bool = False) -> str:
    v = sub[metric].dropna()
    if v.empty:
        return "--"
    num = f"{v.mean():.{digits}f}" if len(v) > 1 else f"{v.iloc[0]:.{digits}f}"
    num_tex = f"\\mathbf{{{num}}}" if bold else num
    if len(v) > 1 and with_std:
        return f"${num_tex}${{\\scriptsize$\\pm${v.std(ddof=1):.{digits}f}}}"
    return f"${num_tex}$" + (PENDING_STD if with_std else "")


def _cell_mean(sub: pd.DataFrame, metric: str) -> float:
    """Row's mean for this metric, used only to find the best value per
    column for bolding -- matches what _cell prints (mean over eval-seed
    repeats when there is more than one row, else the single value)."""
    v = sub[metric].dropna()
    return float(v.mean()) if not v.empty else float("nan")


def _fmt(v: float, digits: int = 1, bold: bool = False) -> str:
    s = f"{v:.{digits}f}"
    return f"$\\mathbf{{{s}}}$" if bold else f"${s}$"


# --------------------------------------------------------------------------
# Table 1 -- main results
# --------------------------------------------------------------------------
# Diagnoses Recall@1 RESTORED to the main table (Jan, 2026-07-28), reversing
# the 2026-07-24 decision to drop it. It is the pipeline's curation gate, so
# hiding it looks like hiding the 32B rank-1 regression; the paper now reports
# it and discusses the regression directly in "Diagnosis Ranking".
#
# JSON Valid dropped, ICD Recall/Precision Macro added (Jan, 2026-07-27):
# decomposes the existing F1 Macro column into its components, mirroring how
# F1 Micro already sits next to F1 Macro -- deliberately macro, not micro,
# Recall/Precision, since macro is the paper's stated primary metric
# throughout (see analyses-table-redesign memory). JSON validity moved out of
# the headline table; it is still discussed in prose where it matters (the
# 14B epoch trade-off).
#
# NormDot (DotProduct) REMOVED from the main table (2026-07-30, reviewer
# feedback for v3): the paper itself argues that NormDot scores agreement
# with a disease-level symptom prototype rather than per-note extraction
# accuracy, so presenting it as one of three evaluated tasks and then
# explaining flat results as "a result rather than a null" is close to
# unfalsifiable. Symptom extraction is now framed as a construction-time
# acceptance gate; its NormDot numbers move to the appendix diagnostic
# table (table_symptom_body.tex) where they document the gate rather than
# claim an evaluation result.
MAIN_METRICS = [
    ("V2 Recall@1", 1),
    ("V2 Recall@3", 1),
    ("ICD F1 Micro", 1),
    ("ICD F1 Macro", 1),
    ("ICD Recall Macro", 1),
    ("ICD Precision Macro", 1),
]

# (latex row label, collection group, kind)
#
# Encoder row promoted into the main table (Jan, 2026-07-29). BioClinical
# ModernBERT's macro-F1 moved 4.3 -> 8.2 on the reseeded run, which makes it a
# real baseline rather than the collapsed one the appendix used to describe --
# and encoder classifiers are the incumbent method for this task (we align the
# task with CliniBench), so keeping the incumbent out of the headline table
# once it is competitive reads as evasive. Only the best of the three (by
# macro-F1, the paper's primary metric) goes in the body; all three plus
# AUROC stay in Appendix~\ref{app:encoders}.
#
# The symptom/diagnosis columns come out as "--" on their own, because
# encoder_metrics.NOT_APPLICABLE_COLUMNS leaves DotProduct/V2 Recall@k NaN for
# these rows. That is deliberate and load-bearing: the intro argues encoders
# map notes into a fixed label space without exposing intermediate reasoning,
# and three empty cells show it rather than assert it.
#
# RESOLVED (2026-07-29): the encoder predictions used to sit on a DIFFERENT
# test split -- 2,350 admissions, only 441 shared with the 2,184-case MERLIN
# test split, and 1,639 of them were MERLIN *train* admissions. Jan supplied a
# corrected export (data/results/encoder_results_29_07, hadm_ids verified 1:1
# against data/results/test_dataset.pq's test split) and
# data/results/encoder_results/ + results_test.csv were regenerated from it
# the same day. BioClinical ModernBERT is still the best-macro-F1 encoder
# (8.2, was 9.1 on the wrong split) so the row choice above is unchanged, but
# S-Proto's micro-F1 moved from worst (16.2) to best (29.8) of the three --
# see table_encoders.tex's caption/backing numbers for the full picture.
MAIN_ROWS = [
    ("\\emph{Untuned baselines}", None, "section"),
    ("MedGemma-27B", "test-medgemma-27b-it", "plain"),
    ("Llama-3.3-70B-Instruct", "test-llama-3-70b-instruct", "plain"),
    ("Baichuan-M2-32B", "test-baichuan-m2-32b", "plain"),
    ("\\emph{Qwen3 base vs.\\ $+$\\,\\merlinshort{} (separately trained checkpoints)}", None, "section"),
    ("Qwen3-0.6B", "test-06b-base", "plain"),
    ("\\quad$+$\\,\\merlinshort{}", "test-06b-full-e4", "ft"),
    ("Qwen3-8B", "test-8b-base", "plain"),
    ("\\quad$+$\\,label-only (no traces)$^{\\dagger}$", "test-8b-mimic-icd2-e4", "plain"),
    ("\\quad$+$\\,\\merlinshort{}", "test-8b-full-e3", "ft"),
    ("Qwen3-14B", "test-14b-base", "plain"),
    ("\\quad$+$\\,\\merlinshort{}", "test-14b-full-e3", "ft"),
    ("Qwen3-32B", "test-32b-base", "plain"),
    ("\\quad$+$\\,\\merlinshort{}", "test-32b-full-e3", "ft"),
    ("\\emph{Encoder classifier}", None, "section"),
    ("BioClin.\\ ModernBERT$^{\\ddagger}$",
     "test-encoder-bioclinical-modernbert-base-balanced-bce", "plain"),
]

ENCODER_SEEDS_CSV = REPO_ROOT / "data" / "results" / "evaluation" / "encoder_seeds.csv"


def _load_encoder_seeds() -> pd.DataFrame:
    """Per-seed encoder rows, written by src/eval/encoder_metrics.py.

    Read as a separate file rather than from EVAL_CSV because encoders never
    enter the vLLM/wandb evaluation pipeline that populates it. Columns are
    already named to match (ICD F1 Micro/Macro, ICD Recall/Precision Macro),
    so the frame concatenates straight onto the LLM rows and _cell computes
    mean +- std over the three training seeds exactly as it does elsewhere.

    NOTE the +- means something different here: encoder seeds are *training*
    seeds, while the LLM rows' +- are evaluation repeats of one checkpoint
    (see the Table 2 caption). The encoder spread is therefore the stricter
    quantity, which is worth stating rather than hiding."""
    if not ENCODER_SEEDS_CSV.exists():
        return pd.DataFrame()
    df = pd.read_csv(ENCODER_SEEDS_CSV)
    df["group"] = df["collection"]
    return df


def table_main(df: pd.DataFrame) -> str:
    # Encoder rows live in their own per-seed CSV (see _load_encoder_seeds);
    # fold them in here only, so the LoRA table is untouched.
    enc = _load_encoder_seeds()
    if not enc.empty:
        df = pd.concat([df, enc], ignore_index=True)

    # Best value per column, for bolding -- across every model row (untuned
    # baselines included), since the table's whole point is that a
    # fine-tuned small model can beat the untuned baselines outright; the
    # column winner happens to always be a Qwen3+MERLIN row in practice.
    subs = {group: df[df["group"] == group] for _, group, kind in MAIN_ROWS if kind != "section"}
    best_val = {m: max((_cell_mean(sub, m) for sub in subs.values()), default=float("nan"))
                for m, _ in MAIN_METRICS}

    lines = [
        "% GENERATED by src/eval/paper_tables.py -- do not edit by hand.",
        "\\begin{table*}[t]",
        "\\centering",
        "\\small",
        "\\setlength{\\tabcolsep}{5pt}",
        "\\begin{tabular}{lcccccc}",
        "\\toprule",
        " & \\multicolumn{2}{c}{\\textbf{Diagnosis ranking}} & "
        "\\multicolumn{4}{c}{\\textbf{ICD codes}} \\\\",
        "\\cmidrule(lr){2-3} \\cmidrule(lr){4-7}",
        "\\textbf{Model} & \\textbf{R@1} & \\textbf{R@3} & "
        "$\\mathbf{F1}_{\\mathrm{mic}}$ & $\\mathbf{F1}_{\\mathrm{mac}}$ & "
        "$\\mathbf{R}_{\\mathrm{mac}}$ & $\\mathbf{P}_{\\mathrm{mac}}$ \\\\",
        "\\midrule",
    ]
    n_col = len(MAIN_METRICS) + 1
    for label, group, kind in MAIN_ROWS:
        if kind == "section":
            lines.append(f"\\multicolumn{{{n_col}}}{{l}}{{{label}}}\\\\")
            continue
        sub = df[df["group"] == group]
        cells = [_cell(sub, m, d, bold=np.isclose(_cell_mean(sub, m), best_val[m]))
                 for m, d in MAIN_METRICS]
        lines.append(f"{label} & " + " & ".join(cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(lines)


# --------------------------------------------------------------------------
# Table 2 -- 8B data-construction ablation (dev)
# --------------------------------------------------------------------------
ABLATION_ROWS = [
    ("8B base (untuned)", None, None),
    ("\\emph{No reasoning traces (label-only)}", "section", None),
    ("\\quad label-only", "mimic", ""),
    ("\\quad label-only $+$ ICD$\\times$2", "mimic-icd2", ""),
    ("\\emph{Reasoning traces, unfiltered}", "section", None),
    ("\\quad all traces", "replace", "$^{\\ast}$"),
    ("\\quad all traces $+$ ICD$\\times$2", "icd2", ""),
    ("\\emph{Reasoning traces, verifier-filtered}", "section", None),
    ("\\quad $\\geq$ ½ thresholds", "thrhalf", ""),
    ("\\quad $\\geq$ thresholds", "thrfull", ""),
    ("\\quad $\\geq$ thresholds, drop not fallback", "thrfulldrop", ""),
    ("\\quad $\\geq$ thresholds $+$ ICD$\\times$2 (\\merlinshort{})", "thrfull-icd2", ""),
]
ABLATION_METRICS = ["DotProduct", "V2 Recall@1", "V2 Recall@3", "ICD F1 Micro", "ICD F1 Macro"]


def table_ablation() -> str:
    dev = pd.read_csv(DEV_CSV)
    best = pd.read_csv(BEST_EPOCHS_CSV)
    best = best[best["model_size"] == "8b"]
    runs = dev[(dev["model_size"] == "8b") & (dev["mode"] == "full")]
    picked = {}
    for _, r in best.iterrows():
        hit = runs[(runs["dataset"] == r["dataset"]) & (runs["epoch"] == r["chosen_epoch"])]
        if not hit.empty:
            picked[r["dataset"]] = hit.iloc[0]
    base = dev[(dev["model_size"] == "8b") & (dev["is_base"] == True)].iloc[0]  # noqa: E712

    # best value per column, for bolding (base row excluded)
    best_val = {m: max(row[m] for row in picked.values() if not np.isnan(row[m]))
                for m in ABLATION_METRICS}

    lines = [
        "% GENERATED by src/eval/paper_tables.py -- do not edit by hand.",
        "\\begin{table*}[t]",
        "\\centering",
        "\\small",
        "\\setlength{\\tabcolsep}{3.5pt}",
        "\\begin{tabular}{lccccc}",
        "\\toprule",
        " & \\textbf{Sympt.} & \\multicolumn{2}{c}{\\textbf{Diagn.}} & "
        "\\multicolumn{2}{c}{\\textbf{ICD}} \\\\",
        "\\cmidrule(lr){2-2} \\cmidrule(lr){3-4} \\cmidrule(lr){5-6}",
        "\\textbf{Training data} & \\textbf{NDot} & \\textbf{R@1} & \\textbf{R@3} & "
        "\\textbf{mic} & \\textbf{mac} \\\\",
        "\\midrule",
    ]
    for label, key, mark in ABLATION_ROWS:
        if key == "section":
            lines.append(f"\\multicolumn{{6}}{{l}}{{{label}}}\\\\")
            continue
        row = base if key is None else picked.get(key)
        if row is None:
            continue
        cells = []
        for m in ABLATION_METRICS:
            v = row[m]
            if np.isnan(v):
                cells.append("--")
            elif key is not None and np.isclose(v, best_val[m]):
                cells.append(f"$\\mathbf{{{v:.1f}}}$")
            else:
                cells.append(f"${v:.1f}$")
        lines.append(f"{label}{mark or ''} & " + " & ".join(cells) + " \\\\")
        if key is None:
            lines.append("\\midrule")
    lines += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(lines)


# --------------------------------------------------------------------------
# Table 3 -- deterministic ICD error decomposition, merged with the
# head/body/tail frequency-stratum F1 (Jan, 2026-07-24: put the long-tail
# numbers in the same table as the error decomposition rather than a
# separate one -- both are "where the ICD gain comes from" cuts of the same
# base-vs-full comparison, at the same four sizes, so they share a row
# structure and can share a table). QA_ROWS's key doubles as the join key
# into longtail_strata.csv via LONGTAIL_SIZE.
# --------------------------------------------------------------------------
QA_ROWS = [("0.6B", "06b"), ("8B", "8b"), ("14B", "14b"), ("32B", "32b")]
# Redesigned 2026-07-27 (Jan, analyses-table-redesign): JSON is out everywhere
# and the old missed_X% columns (lower-is-better, sitting next to everything
# else higher-is-better) are replaced by real class-conditional F1 for
# history/medication/chronic (scripts/qa_deterministic.py::score_model) and a
# recall-only Primary column (missed_primary has no meaningful F1 -- see the
# docstring there). Head/Body/Tail are unchanged. Full F1$_{mac}$ (full,
# untruncated ICD code, macro over the whole label space -- scripts/
# qa_full_code.py) is new: unlike history/medication/chronic, which are
# narrow hand-picked subsets where micro vs. macro barely diverges, full-code
# spans the entire real label space, so this is deliberately macro -- the
# sharpest "the tail isn't solved" evidence in the paper (full-code macro-F1
# stays under 4% even for 32B+FT, vs. 26.6 micro).
QA_METRICS = [("recall_primary", "Primary"), ("f1_history", "Hist.\\ F1"),
              ("f1_medication", "Med.\\ F1"), ("f1_chronic", "Chron.\\ F1")]
LONGTAIL_CSV = REPO_ROOT / "data" / "results" / "evaluation" / "longtail_strata.csv"
# QA_ROWS labels ("0.6B") already match longtail_strata.csv's model_size
# column directly -- no remapping needed, unlike the qa_key ("06b").
LONGTAIL_STRATA = ["head", "body", "tail"]
FULL_CODE_CSV = REPO_ROOT / "data" / "qa" / "full_code_by_granularity.csv"


def table_qa() -> str:
    qa = pd.read_csv(QA_CSV, index_col=0)
    lt = pd.read_csv(LONGTAIL_CSV).set_index(["model_size", "mode", "stratum"])
    fc = pd.read_csv(FULL_CODE_CSV).set_index(["size", "variant", "granularity", "metric_type"])

    # Gather every (size, mode) row's values first so the best value per
    # column can be found across the whole table -- in this table the 32B
    # +MERLIN row happens to win every column (Analysis \S\ref{sec:longtail}
    # reports the gain growing with scale), so this ends up bolding that one
    # row throughout rather than scattering winners across sizes.
    col_keys = [m for m, _ in QA_METRICS] + LONGTAIL_STRATA + ["full_f1"]
    row_values = {}
    for label, key in QA_ROWS:
        for mode in ("base", "full"):
            row = qa.loc[f"{key}_{mode}"]
            vals = {m: row[m] for m, _ in QA_METRICS}
            for s in LONGTAIL_STRATA:
                vals[s] = lt.loc[(label, mode, s), "f1"]
            vals["full_f1"] = fc.loc[(key, mode, "full", "macro"), "f1"] * 100
            row_values[(label, mode)] = vals
    best_val = {c: max(v[c] for v in row_values.values()) for c in col_keys}

    lines = [
        "% GENERATED by src/eval/paper_tables.py -- do not edit by hand.",
        "\\begin{table*}[t]",
        "\\centering",
        "\\small",
        "\\setlength{\\tabcolsep}{3.2pt}",
        "\\begin{tabular}{ll" + "c" * len(QA_METRICS) + "c" * len(LONGTAIL_STRATA) + "c}",
        "\\toprule",
        " & & \\multicolumn{4}{c}{\\textbf{Recall / F1 by code class (\\%)}} & "
        "\\multicolumn{3}{c}{\\textbf{F1 by train.\\ frequency}} & \\textbf{Full code} \\\\",
        "\\cmidrule(lr){3-6} \\cmidrule(lr){7-9} \\cmidrule(lr){10-10}",
        "\\textbf{Size} & & " + " & ".join(f"\\textbf{{{h}}}" for _, h in QA_METRICS)
        + " & " + " & ".join(f"\\textbf{{{s.capitalize()}}}" for s in LONGTAIL_STRATA)
        + " & $\\mathbf{F1}_{\\mathrm{mac}}$ \\\\",
        "\\midrule",
    ]
    for label, key in QA_ROWS:
        for i, (mode, mode_label) in enumerate((("base", "base"), ("full", "$+$\\,\\merlinshort{}"))):
            vals = row_values[(label, mode)]
            cells = [_fmt(vals[m], 1, np.isclose(vals[m], best_val[m])) for m, _ in QA_METRICS]
            cells += [_fmt(vals[s], 1, np.isclose(vals[s], best_val[s])) for s in LONGTAIL_STRATA]
            cells.append(_fmt(vals["full_f1"], 1, np.isclose(vals["full_f1"], best_val["full_f1"])))
            first = f"\\multirow{{2}}{{*}}{{{label}}}" if i == 0 else ""
            lines.append(f"{first} & {mode_label} & " + " & ".join(cells) + " \\\\")
        if key != "32b":
            lines.append("\\addlinespace[2pt]")
    lines += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(lines)


# --------------------------------------------------------------------------
# Table 4 (appendix) -- LoRA vs. full fine-tuning
# --------------------------------------------------------------------------
LORA_ROWS = [
    ("Qwen3-0.6B", "test-06b-base", ""),
    ("\\quad$+$\\,\\merlinshort{} (full)", "test-06b-full-e4", ""),
    ("\\quad$+$\\,\\merlinshort{} (LoRA $r{=}128$)", "test-06b-r128-e4", ""),
    ("Qwen3-8B", "test-8b-base", ""),
    ("\\quad$+$\\,\\merlinshort{} (full)", "test-8b-full-e3", ""),
    ("\\quad$+$\\,\\merlinshort{} (LoRA $r{=}128$)", "test-8b-r128-e3", "$^{\\ast}$"),
    ("Qwen3-14B", "test-14b-base", ""),
    ("\\quad$+$\\,\\merlinshort{} (full)", "test-14b-full-e3", ""),
    ("\\quad$+$\\,\\merlinshort{} (LoRA $r{=}128$)", "test-14b-r128-e3", ""),
    ("Qwen3-32B", "test-32b-base", ""),
    ("\\quad$+$\\,\\merlinshort{} (full)", "test-32b-full-e3", ""),
    ("\\quad$+$\\,\\merlinshort{} (LoRA $r{=}256$)", "test-32b-r256-e4", ""),
]


# ICD Recall/Precision Macro added 2026-07-28 (Jan) so this appendix table
# carries the same ICD block as the main table. Seed-42 values for these two
# columns came from src/eval/backfill_eval_metrics_pr.py.
LORA_METRICS = ("DotProduct", "V2 Recall@1", "V2 Recall@3",
                "ICD F1 Micro", "ICD F1 Macro",
                "ICD Recall Macro", "ICD Precision Macro")


def table_lora(df: pd.DataFrame) -> str:
    # canonical seed-42 row only: the LoRA runs have no seed replicates, so
    # this table is single-seed throughout and must not silently mix a
    # seed-43/44 row in for the base/full comparison rows.
    subs = {group: df[df["collection"] == group] for _, group, _ in LORA_ROWS}
    # Best value per column, across every row (base, full FT, LoRA, all
    # sizes) -- e.g. the 32B LoRA row turns out to win Recall@3 outright,
    # which matches the appendix claim that LoRA "matches or exceeds" full
    # fine-tuning on that column.
    best_val = {m: max((_cell_mean(sub, m) for sub in subs.values()), default=float("nan"))
                for m in LORA_METRICS}

    lines = [
        "% GENERATED by src/eval/paper_tables.py -- do not edit by hand.",
        "\\begin{table*}[t]",
        "\\centering",
        "\\small",
        "\\setlength{\\tabcolsep}{3pt}",
        "\\begin{tabular}{lccccccc}",
        "\\toprule",
        " & \\textbf{Sympt.} & \\multicolumn{2}{c}{\\textbf{Diagn.}} & "
        "\\multicolumn{4}{c}{\\textbf{ICD codes}} \\\\",
        "\\cmidrule(lr){2-2} \\cmidrule(lr){3-4} \\cmidrule(lr){5-8}",
        "\\textbf{Model} & \\textbf{NormDot} & \\textbf{R@1} & \\textbf{R@3} & "
        "$\\mathbf{F1}_{\\mathrm{mic}}$ & $\\mathbf{F1}_{\\mathrm{mac}}$ & "
        "$\\mathbf{R}_{\\mathrm{mac}}$ & $\\mathbf{P}_{\\mathrm{mac}}$ \\\\",
        "\\midrule",
    ]
    for label, group, mark in LORA_ROWS:
        sub = subs[group]
        cells = [_cell(sub, m, 1, with_std=False,
                        bold=np.isclose(_cell_mean(sub, m), best_val[m]))
                 for m in LORA_METRICS]
        lines.append(f"{label}{mark} & " + " & ".join(cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(lines)


# --------------------------------------------------------------------------
# Appendix -- symptom-extraction diagnostic (NormDot)
# --------------------------------------------------------------------------
# Added 2026-07-30 when NormDot left the main table. This is deliberately a
# one-metric table in the appendix rather than a column in Table 2: NormDot
# scores agreement with a disease-level symptom prototype, so it documents
# the pipeline's first acceptance gate and nothing is claimed about
# extraction accuracy from it. No bolding -- there is no "best" prototype
# agreement, which is precisely the point.
def table_symptom(df: pd.DataFrame) -> str:
    lines = [
        "% GENERATED by src/eval/paper_tables.py -- do not edit by hand.",
        "\\begin{table}[htbp]",
        "\\centering",
        "\\small",
        "\\setlength{\\tabcolsep}{5pt}",
        "\\begin{tabular}{lc}",
        "\\toprule",
        "\\textbf{Model} & \\textbf{NormDot} \\\\",
        "\\midrule",
    ]
    # Buffer a section header and only emit it once a row under it survives;
    # the encoder section would otherwise print an empty heading, since a
    # discriminative classifier emits no symptom profile to score.
    pending_section = None
    for label, group, kind in MAIN_ROWS:
        if kind == "section":
            pending_section = label
            continue
        sub = df[df["group"] == group]
        cell = _cell(sub, "DotProduct", 1)
        if cell == "--":
            continue
        if pending_section is not None:
            lines.append(f"\\multicolumn{{2}}{{l}}{{{pending_section}}}\\\\")
            pending_section = None
        # Footnote markers belong to Table 2's caption, not this one.
        label = label.replace("$^{\\dagger}$", "").replace("$^{\\ddagger}$", "")
        lines.append(f"{label} & {cell} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(lines)


def run(out_dir: Path = OUT_DIR):
    out_dir.mkdir(parents=True, exist_ok=True)
    df = _load_eval()
    for name, body in (("table_main_body", table_main(df)),
                       ("table_ablation_body", table_ablation()),
                       ("table_qa_body", table_qa()),
                       ("table_symptom_body", table_symptom(df)),
                       ("table_lora_body", table_lora(df))):
        (out_dir / f"{name}.tex").write_text(body + "\n")
        print(f"wrote {out_dir / (name + '.tex')}")


if __name__ == "__main__":
    run()
