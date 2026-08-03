#!/usr/bin/env python3
"""Head / body / tail decomposition of ICD performance.

    python -m src.eval.longtail_strata

The paper claims a long-tail problem and reports macro-F1 because of it, but
a single macro number does not show *where* on the tail the gain lands. This
script splits the three-character ICD label space by how often each code
occurs in the MERLIN training split and reports per-stratum F1 and recall on
the held-out test split, base vs. full fine-tuning, at every model size.

Strata (standard many/medium/few-shot convention, thresholds in
longtail_config.py): head >= 100 training admissions, body 10-99, tail < 10
(including codes never seen in training).

Denominator note: per-stratum macro-F1 averages per-label F1 over labels that
occur in the *test gold* set, whereas the headline macro-F1 in
classification_metrics.calculate_icd_metrics_cpu averages over gold OR
predicted labels. Stratum values therefore do not average back to the
headline number -- predicted-only labels (false positives for a code that
never appears in test gold) contribute a zero to the headline macro and are
excluded here, because they have no training frequency to bucket them by.

Outputs
-------
data/results/evaluation/longtail_strata.csv   per size/mode/stratum
paper/EACL_2026_v2/table_longtail_body.tex    LaTeX table body
"""
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from src.eval import longtail_config as cfg

REPO_ROOT = Path(__file__).resolve().parents[2]
EVAL_DIR = REPO_ROOT / "data" / "results" / "evaluation" / "1_4"
NOTES_PQ = REPO_ROOT / "data" / "preprocessed_mimic" / "annotated_mimic_notes.pq"
INSTRUCTIONS_PQ = REPO_ROOT / "data" / "results" / "instructions" / cfg.TRAIN_INSTRUCTIONS
OUT_CSV = REPO_ROOT / "data" / "results" / "evaluation" / "longtail_strata.csv"
OUT_TEX = REPO_ROOT / "paper" / "EACL_2026_v2" / "table_longtail_body.tex"


def _short(codes) -> set:
    """Gold/predicted code list -> set of three-character categories, matching
    utils.convert_codes_to_short_codes and the eval scorer."""
    if codes is None:
        return set()
    if isinstance(codes, str):  # some checkpoints store v4_preds as a string
        codes = [t.strip().strip("'\"") for t in codes.strip("[]").split(",") if t.strip()]
    if codes is None:
        return set()
    return {str(c)[:3] for c in codes if c is not None and str(c).strip()}


def training_frequency() -> pd.Series:
    """Per three-character code: number of TRAIN admissions carrying it.

    Train admissions come from the instruction dataset actually used for the
    headline recipe (read column-wise -- the full file does not fit in
    memory), gold codes from the annotated MIMIC notes."""
    ids = pq.read_table(INSTRUCTIONS_PQ, columns=["hadm_id", "split"]).to_pandas()
    train_ids = set(ids.loc[ids["split"] == "train", "hadm_id"])
    notes = pd.read_parquet(NOTES_PQ, columns=["hadm_id", "ICD_CODES"])
    notes = notes[notes["hadm_id"].isin(train_ids)].drop_duplicates("hadm_id")
    freq = defaultdict(int)
    for codes in notes["ICD_CODES"]:
        for c in _short(codes):
            freq[c] += 1
    return pd.Series(freq, dtype=int).sort_values(ascending=False)


def stratum_of(code: str, freq: pd.Series) -> str:
    n = int(freq.get(code, 0))
    for name, lo in cfg.STRATA:          # ordered high -> low
        if n >= lo:
            return name
    return cfg.STRATA[-1][0]


def _load(collection: str):
    p = EVAL_DIR / f"eval_results_{collection}" / f"eval_results_{collection}.pq"
    d = pd.read_parquet(p, columns=["ICD_CODES", "v4_preds"])
    return [_short(c) for c in d["ICD_CODES"]], [_short(v) for v in d["v4_preds"]]


def score(collection: str, freq: pd.Series) -> dict:
    gold, pred = _load(collection)
    tp, fp, fn = defaultdict(int), defaultdict(int), defaultdict(int)
    for g, p in zip(gold, pred):
        for c in p & g:
            tp[c] += 1
        for c in p - g:
            fp[c] += 1
        for c in g - p:
            fn[c] += 1

    buckets = {name: {"f1": [], "tp": 0, "gold": 0} for name, _ in cfg.STRATA}
    for c in set().union(*gold):
        pr = tp[c] / (tp[c] + fp[c]) if tp[c] + fp[c] else 0.0
        rc = tp[c] / (tp[c] + fn[c]) if tp[c] + fn[c] else 0.0
        b = buckets[stratum_of(c, freq)]
        b["f1"].append(2 * pr * rc / (pr + rc) if pr + rc else 0.0)
        b["tp"] += tp[c]
        b["gold"] += tp[c] + fn[c]
    return {name: dict(f1=100 * float(np.mean(b["f1"])) if b["f1"] else 0.0,
                       recall=100 * b["tp"] / b["gold"] if b["gold"] else 0.0,
                       n_labels=len(b["f1"]))
            for name, b in buckets.items()}


def run():
    freq = training_frequency()
    print(f"train label space: {len(freq)} codes, {int(freq.sum())} code-admission pairs")
    for i, (name, lo) in enumerate(cfg.STRATA):
        # STRATA is ordered high -> low, so the upper bound of this bucket is
        # the previous entry's threshold (exclusive); the first has none.
        sel = freq[freq >= lo] if i == 0 else \
            freq[(freq >= lo) & (freq < cfg.STRATA[i - 1][1])]
        print(f"  {name:5s} >= {lo:4d}: {len(sel):4d} codes, "
              f"{100 * sel.sum() / freq.sum():5.1f}% of training occurrences")

    rows = []
    for size, base, full in cfg.CHECKPOINTS:
        for mode, coll in (("base", base), ("full", full)):
            res = score(coll, freq)
            for stratum, v in res.items():
                rows.append(dict(model_size=size, mode=mode, stratum=stratum, **v))
    df = pd.DataFrame(rows)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_CSV, index=False)
    print(f"\nwrote {OUT_CSV}")
    print(df.pivot_table(index=["model_size", "mode"], columns="stratum",
                         values="f1").round(1).to_string())

    OUT_TEX.parent.mkdir(parents=True, exist_ok=True)
    OUT_TEX.write_text(_latex(df, freq) + "\n")
    print(f"wrote {OUT_TEX}")


def _latex(df: pd.DataFrame, freq: pd.Series) -> str:
    """F1-only (Recall dropped 2026-07-27: it tracked F1 in every cell here
    and added no separate signal -- see revision notes -- and this table is
    now combined with the long-tail histogram in the appendix, so column
    space is at a premium). #labels row kept: it's the one piece of context
    (how many distinct codes are actually in each stratum) the main-text
    prose doesn't repeat."""
    names = [n for n, _ in cfg.STRATA]
    piv = df.set_index(["model_size", "mode", "stratum"])
    n_lab = {s: int(df[df.stratum == s]["n_labels"].iloc[0]) for s in names}
    lines = [
        "% GENERATED by src/eval/longtail_strata.py -- do not edit by hand.",
        "\\begin{minipage}[t]{0.47\\textwidth}",
        "\\vspace{0pt}",
        "\\centering", "\\small",
        "\\setlength{\\tabcolsep}{6pt}",
        "\\begin{tabular}{l" + "c" * len(names) + "}",
        "\\toprule",
        "\\textbf{Size} & " + " & ".join(f"\\textbf{{{n.capitalize()}}}" for n in names) + " \\\\",
        "\\emph{\\#labels} & " + " & ".join(f"\\emph{{{n_lab[n]}}}" for n in names) + " \\\\",
        "\\midrule",
    ]
    for size, _, _ in cfg.CHECKPOINTS:
        for mode, label in (("base", f"{size} base"), ("full", f"{size} $+$\\,\\merlinshort{{}}")):
            r = None
            cells = []
            for n in names:
                r = piv.loc[(size, mode, n)]
                cells.append(f"${r['f1']:.1f}$")
            lines.append(f"{label} & " + " & ".join(cells) + " \\\\")
        if size != cfg.CHECKPOINTS[-1][0]:
            lines.append("\\addlinespace[1pt]")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{minipage}"]
    return "\n".join(lines)


if __name__ == "__main__":
    run()
