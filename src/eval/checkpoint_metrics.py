#!/usr/bin/env python3
"""Reshape scripts/eval_analysis.py new_evals's flat eval_metrics CSV into the
long format src/eval/checkpoint_plots.py needs: one row per
(model_size, dataset, epoch) for every fine-tuned checkpoint family in
scripts/eval_config.yaml, plus one row per untuned base
model (no epoch -- the `{size}b-base` entries from
scripts/eval_config.yaml, evaluated in the same
merlin-eval-1.4 wandb project so they land in the same input CSV).

Distinct from scripts/compute_checkpoint_metrics_local.py (the v1.3 sibling):
that script reads raw .pq files from wandb_downloads/eval_results/ and
hand-computes ICD F1/MRR/etc. itself. This script instead just parses names
out of the already-computed metrics CSV (src/eval/eval_results.py) -- no metric
math here, evaluate_experiment() already did it.

Lives in src/eval/ (moved from scripts/ on 2026-07-15) alongside the rest of
the post-hoc eval/analysis code, since it's a data-reshaping step in that
pipeline rather than a one-off script. `scripts/eval_analysis.py new_evals`
runs this reshape itself -- run that, or
`python -m src.eval.checkpoint_metrics` from repo root.

scripts/eval_analysis.py new_evals calls this module's main() itself right
after it writes eval_metrics_merlin-eval-1.4.csv (2026-07-15 -- the two used
to be separate manual steps), so in the normal case you don't need to run
this on its own at all:
    python scripts/eval_analysis.py new_evals     # writes both CSVs
    python scripts/eval_analysis.py create_plots --plots quick
    python scripts/eval_analysis.py create_plots --plots paper

Only run this file directly if you've hand-edited FAMILY_INFO below and want
to re-reshape the existing eval_metrics CSV without re-downloading/
re-evaluating anything from WandB:
    python -m src.eval.checkpoint_metrics
"""
import re
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[2]
IN_CSV = REPO / "data" / "eval_metrics_merlin-eval-1.4.csv"
OUT_DIR = REPO / "data" / "results" / "evaluation"
OUT_CSV = OUT_DIR / "checkpoint_metrics.csv"

# short_name (from eval_config.yaml manifests) -> (model_size, dataset, mode).
# mode is "full" (full fine-tune) or "lora" (LoRA adapter, any rank) -- used
# downstream to compare full vs. lora at sizes that trained both.
# Keep in sync with that file's `manifests:` list if families are added/renamed.
FAMILY_INFO = {
    "32b-r256-icd2":         ("32b", "icd2",         "lora"),
    "32b-r256-thrfull-icd2": ("32b", "thrfull-icd2", "lora"),
    "8b-full-icd2":          ("8b",  "icd2",         "full"),
    "8b-full-thrfull-icd2":  ("8b",  "thrfull-icd2", "full"),
    "8b-full-mimic":         ("8b",  "mimic",        "full"),
    "8b-full-mimic-icd2":    ("8b",  "mimic-icd2",   "full"),
    # Added 2026-07-12: 8B ablation of thrfull-icd2 that DROPS below-threshold/
    # nan/0 rows instead of mimic-fallback'ing them (drop_below_threshold=True
    # in build_instructions.py) -- a different dataset/treatment from
    # "thrfull-icd2" above, not a rename of it, so it gets its own dataset
    # label rather than sharing that key.
    "8b-full-thrfulldrop":   ("8b",  "thrfulldrop",  "full"),
    # NOTE (2026-07-15): there is no 8B LoRA v1.4 data yet. A "8b-r128-
    # thrfulldrop" entry briefly lived here, but its numbers turned out to be
    # stale v1.3 data mislabeled under a v1.4-looking name (identical to
    # 8b-full-thrfulldrop at every epoch) -- removed. We are starting real 8B
    # LoRA (r128) SFT runs now; once those are evaluated, add their
    # short_name(s) here as ("8b", <dataset>, "lora") the same way the other
    # families above are wired in.
    # Added 2026-07-11: 14B-lora-r128, 0.6B-full, 0.6B-lora-r128 -- all on the
    # single canonical merlin_thrfull_icd2x dataset (same one the 32b/8b
    # families above call "thrfull-icd2"), varying model size/mode instead of
    # dataset. Note "0.6b-full" and "0.6b-r128" share the same (model_size,
    # dataset) key on purpose -- there's only one dataset per size here, so
    # the two training modes just compete as alternate checkpoints for
    # "best epoch on thrfull-icd2 at 0.6b" rather than being split into
    # separate dataset columns.
    "14b-r128":  ("14b",  "thrfull-icd2", "lora"),
    "06b-r128":  ("0.6b", "thrfull-icd2", "lora"),
    "06b-full":  ("0.6b", "thrfull-icd2", "full"),
    # Added 2026-07-13: full fine-tune runs for 14b and 32b on thrfull-icd2,
    # giving those two sizes a real full-vs-lora pair too (previously
    # lora-only).
    "14b-full-thrfull-icd2": ("14b", "thrfull-icd2", "full"),
    "32b-full-thrfull-icd2": ("32b", "thrfull-icd2", "full"),
    # Added 2026-07-16: run_plan.yaml runs 8-11 (see
    # eval_config.yaml) have now been evaluated -- 8B LoRA
    # (r128) finally exists (resolves the "no 8B LoRA v1.4 data yet" note
    # that used to sit here), plus three more 8B full-FT dataset ablations
    # (replace/thr_half/thr_full, distinct from thrfull-icd2 and thrfulldrop
    # above -- see data/results/instructions/merlin-v1.4/ for the underlying
    # dataset variants).
    #
    # CORRECTION (2026-07-21): this same 2026-07-16 edit also re-added a
    # "8b-r128-thrfulldrop" entry (tagged "lora") -- this is the exact bug the
    # 2026-07-15 note above already caught and removed once: there is no real
    # 8B LoRA thrfulldrop run. Its eval_metrics rows are byte-for-byte
    # identical to "8b-full-thrfulldrop" at every epoch (confirmed in
    # data/eval_metrics_merlin-eval-1.4.csv), i.e. it's the full-FT run's
    # numbers duplicated under a misleading r128-looking name, not a second
    # checkpoint. We confirmed thrfulldrop was only ever trained full-param
    # (`ft-8b-full-thrfulldrop-v1-4`, `use_lora: False` in run_plan.yaml).
    # Removed again -- do not re-add a "8b-r128-thrfulldrop" key without
    # first checking it isn't just this duplicate resurfacing. This bug fed
    # a phantom "thrfulldrop" bar/point into every LoRA-side 8B figure
    # (bars_8b_lora, epoch_bars_8b_lora, heatmap_dataset_x_epoch_8b_lora) --
    # regenerate those after this fix.
    "8b-r128-thrfull-icd2": ("8b", "thrfull-icd2", "lora"),
    "8b-full-replace":      ("8b", "replace",      "full"),
    "8b-full-thrhalf":      ("8b", "thrhalf",      "full"),
    "8b-full-thrfull":      ("8b", "thrfull",      "full"),
}

# The held-out TEST-split sweep (originally driven by
# the test-split config -- since cleaned up/superseded, same as
# the one-off configs noted on FAMILY_INFO above): the size x mode grid
# (lora/full at 0.6B/8B/14B/32B), all trained on the single canonical
# merlin_thrfull_icd2x dataset. Unlike FAMILY_INFO above, each of these
# evaluates EXACTLY ONE checkpoint per size/mode -- the epoch already
# selected on dev by ICD F1 Macro -- so there's no per-family epoch sweep to
# disambiguate, just a single short_name -> (model_size, mode) lookup. Job
# names come out as "test-{short_name suffix}-e{epoch}" e.g. "test-8b-full-e3".
# "test-8b-r128" (8B LoRA on test) added 2026-07-17 once that SFT run
# finished dev-eval and was run on test -- completes the size x mode grid.
TEST_FAMILY_INFO = {
    "test-06b-full": ("0.6b", "full"),
    "test-06b-r128": ("0.6b", "lora"),
    "test-8b-full":  ("8b",   "full"),
    "test-8b-r128":  ("8b",   "lora"),
    "test-14b-full": ("14b",  "full"),
    "test-14b-r128": ("14b",  "lora"),
    "test-32b-full": ("32b",  "full"),
    "test-32b-r256": ("32b",  "lora"),
}
# All TEST_FAMILY_INFO runs trained on the same canonical dataset -- test
# rows don't get a per-family dataset like FAMILY_INFO does.
TEST_DATASET = "thrfull-icd2"

# eval_job_name() in eval_checkpoints.py emits "{short_name}-e{epoch}" for
# every v1.4 config entry (all well under the 51-char k8s budget, so none of
# them fall back to the sha1-hash naming used in v1.3 -- see _fit_k8s_name()).
FAMILY_RE = re.compile(r"^(?P<family>.+)-e(?P<epoch>\d+)$")

# job_name convention from eval_config.yaml: "{digits}b-base"
# (e.g. "32b-base", "8b-base", "06b-base" for the 0.6B model).
BASE_RE = re.compile(r"^(?P<size>\d+)b-base$")

# job_name convention from eval_base_models_config_test.yaml (added
# 2026-07-21): "test-{digits}b-base" -- a REAL test-split eval of the raw
# base models (test-32b-base / test-14b-base / test-8b-base / test-06b-base),
# not the same number reused from BASE_RE's dev-split run. Until this run
# existed, every test-split figure's "base" bar/column silently reused the
# DEV-split base score (see plot_main_figure_test's docstring in
# paper_plots.py) -- that was a documented stand-in, not a bug, but now that
# real test-split base numbers exist they should be preferred wherever a
# test-split figure needs a base reference.
TEST_BASE_RE = re.compile(r"^test-(?P<size>\d+)b-base$")

# job_name convention from the external-models config:
# "test-{model-slug}" -- one-off TEST-split spot-checks of external/competitor
# models, not part of the Qwen3 size/mode grid (no fine-tuning, no epoch
# sweep, `model_size` doesn't apply so figures that group by size skip them
# via the usual `.dropna()` on that column). Added 2026-07-21: Baichuan-M2-32B
# and MEDITRON3-8B; MedGemma-27B-it and Llama-3.3-70B-Instruct queued next
# (see the revision notes, "Competitor / base-model baseline
# testing"). Explicit allowlist rather than a generic "test-*" catchall, so
# an unrecognized test- collection still lands in the "Skipped" list instead
# of silently misparsing.
EXTERNAL_TEST_MODELS = {
    "test-baichuan-m2-32b": "Baichuan-M2-32B",
    "test-meditron3-8b": "MEDITRON3-8B",
    "test-medgemma-27b-it": "MedGemma-27B-it",
    "test-llama-3-70b-instruct": "Llama-3.3-70B-Instruct",
}

# Eval-time-only seed sweep (scripts/eval_error_bars_config_seed43[.yaml|
# _8b.yaml] / _seed44[...], added 2026-07-23): re-evals the SAME seed-42-
# trained checkpoints/base models at a different Client_Job.seed (43 or 44)
# -- NO retraining, just resampling the generator-verifier pipeline -- so the
# main test-split figure (plot_main_figure_2x1 / plot_main_figure_test in
# paper_plots.py) can show error bars. Distinct from the EARLIER
# training-seed sweep (run_plan.yaml runs 14-15, robustness_plots.py), which
# retrains SFT and only covers 8b-full-thrfull2icd -- collection names from
# that sweep (e.g. "8b-full-thrfull-icd2-seed43-e3-seed43") don't start with
# "test-", so they don't match the regexes below and keep falling through to
# FAMILY_RE same as before (still unrecognized there -> skipped, same as
# today -- that sweep's own numbers live in eval_metrics_merlin-eval-1.4.csv
# directly, read by robustness_plots.py, not through this reshape).
#
# Naming: the manifest/`models:` short_name or job_name in the eval_error_bars
# configs already ends in "-seed{N}" (short_name "test-06b-full-seed43",
# job_name "test-06b-base-seed43"), and src/wandb/run.py separately appends
# its OWN automatic "-seed{N}" suffix whenever Client_Job.seed != 42 -- so the
# actual wandb collection name has a doubled suffix:
#   full/lora checkpoint: "test-06b-full-seed43-e4-seed43"
#   base model:            "test-06b-base-seed43-seed43"
# (documented in the eval_error_bars_config_seed{43,44}[.yaml|_8b.yaml]
# headers). The two seed numbers are always identical -- both trace back to
# the same Client_Job.seed -- so either one can be used as `seed`. Greedy
# `.+`/`\d+` backtracking makes the family/epoch capture land exactly where
# TEST_FAMILY_RE/TEST_BASE_RE's plain (non-seed) names would -- e.g. family
# "test-06b-full" out of "test-06b-full-seed43-e4-seed43", not
# "test-06b-full-seed43" -- so the family/size lookups below reuse
# TEST_FAMILY_INFO / the same size-label logic as TEST_BASE_RE unchanged.
EVAL_SEED_TEST_FAMILY_RE = re.compile(
    r"^(?P<family>test-.+)-seed(?P<seed1>\d+)-e(?P<epoch>\d+)-seed(?P<seed2>\d+)$"
)
EVAL_SEED_TEST_BASE_RE = re.compile(
    r"^test-(?P<size>\d+)b-base-seed(?P<seed1>\d+)-seed(?P<seed2>\d+)$"
)


def parse_name(name: str):
    """Return (model_size, dataset, mode, epoch, is_base, is_test, is_external,
    seed) for a recognized v1.4 collection name, or None if `name` doesn't
    match any known v1.4 checkpoint family, base model, external-model
    spot-check, or test-split run (e.g. leftover v1.3 rows sitting in the
    same input CSV from an earlier run -- those are skipped, not errors).

    `seed` is 42 (the canonical/only-ever-trained seed) for every ordinary
    collection, and 43/44 for the eval-time seed-robustness reruns matched by
    EVAL_SEED_TEST_FAMILY_RE / EVAL_SEED_TEST_BASE_RE above -- see those
    regexes' comment for why the collection name has a doubled "-seed{N}"
    suffix. Checked before the plain TEST_BASE_RE/FAMILY_RE patterns since
    those wouldn't match the doubled-suffix names anyway ($ anchors don't
    line up), but ordering them first keeps the seed-aware path the obvious
    one to read."""
    if name in EXTERNAL_TEST_MODELS:
        return None, None, "external", None, False, True, True, 42

    m = EVAL_SEED_TEST_BASE_RE.match(name)
    if m:
        size = m.group("size")
        size_label = f"{int(size) / 10}b" if size.startswith("0") else f"{size}b"
        return size_label, None, None, None, True, True, False, int(m.group("seed1"))

    m = TEST_BASE_RE.match(name)
    if m:
        size = m.group("size")
        size_label = f"{int(size) / 10}b" if size.startswith("0") else f"{size}b"
        return size_label, None, None, None, True, True, False, 42

    m = BASE_RE.match(name)
    if m:
        size = m.group("size")
        size_label = f"{int(size) / 10}b" if size.startswith("0") else f"{size}b"
        return size_label, None, None, None, True, False, False, 42

    m = EVAL_SEED_TEST_FAMILY_RE.match(name)
    if m:
        family, epoch, seed = m.group("family"), int(m.group("epoch")), int(m.group("seed1"))
        if family in TEST_FAMILY_INFO:
            model_size, mode = TEST_FAMILY_INFO[family]
            return model_size, TEST_DATASET, mode, epoch, False, True, False, seed
        return None  # unrecognized family -- skipped, same as any other unmatched name

    m = FAMILY_RE.match(name)
    if m:
        family, epoch = m.group("family"), int(m.group("epoch"))
        if family in TEST_FAMILY_INFO:
            model_size, mode = TEST_FAMILY_INFO[family]
            return model_size, TEST_DATASET, mode, epoch, False, True, False, 42
        if family in FAMILY_INFO:
            model_size, dataset, mode = FAMILY_INFO[family]
            return model_size, dataset, mode, epoch, False, False, False, 42

    return None


def main(in_csv=None, out_dir=None):
    in_csv = Path(in_csv) if in_csv else IN_CSV
    out_dir = Path(out_dir) if out_dir else OUT_DIR
    out_csv = out_dir / OUT_CSV.name
    if not in_csv.exists():
        print(f"{in_csv} not found -- run `scripts/eval_analysis.py new_evals` first.")
        return
    df = pd.read_csv(in_csv, index_col=0)

    rows, skipped = [], []
    for name, metrics in df.iterrows():
        parsed = parse_name(name)
        if parsed is None:
            skipped.append(name)
            continue
        model_size, dataset, mode, epoch, is_base, is_test, is_external, seed = parsed
        rows.append({
            "collection": name,
            "model_size": model_size,
            "dataset": dataset,
            "mode": mode,
            "epoch": epoch,
            "is_base": is_base,
            "is_test": is_test,
            "is_external": is_external,
            # 42 for every ordinary run; 43/44 for the eval-time seed-
            # robustness reruns (EVAL_SEED_TEST_FAMILY_RE/EVAL_SEED_TEST_BASE_RE
            # above) -- lets paper_plots.py average error bars across seed
            # replicates of the same (model_size, mode) without disturbing
            # every other figure, which pins seed==42 explicitly.
            "seed": seed,
            "model_name": EXTERNAL_TEST_MODELS.get(name),
            **metrics.to_dict(),
        })

    if skipped:
        preview = ", ".join(skipped[:5]) + (" ..." if len(skipped) > 5 else "")
        print(f"Skipped {len(skipped)} row(s) not matching a v1.4 checkpoint/base-model "
              f"name: {preview}")

    out = pd.DataFrame(rows)
    if out.empty:
        print("No v1.4 rows found in the input CSV -- nothing to write.")
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_csv, index=False)
    n_base = int(out["is_base"].sum())
    n_external = int(out["is_external"].sum())
    n_test = int(out["is_test"].sum())
    n_dev_ckpt = len(out) - n_base - n_test
    print(f"Wrote {len(out)} row(s) ({n_dev_ckpt} dev checkpoint, {n_test} test checkpoint "
          f"[{n_external} external], {n_base} base model) -> {out_csv}")

    n_seed_replicates = int((out["seed"] != 42).sum())
    if n_seed_replicates:
        replicate_rows = out.loc[out["seed"] != 42, ["model_size", "mode", "seed", "collection"]]
        print(f"  {n_seed_replicates} eval-time seed-robustness replicate row(s) "
              f"(seed != 42, feeds plot_main_figure_2x1/plot_main_figure_test's error bars):")
        for _, r in replicate_rows.sort_values(["model_size", "mode", "seed"]).iterrows():
            print(f"    {r['model_size']} {r['mode'] or 'base'} seed{r['seed']}: {r['collection']}")

    # model_size is NaN for external-model rows (no Qwen3 size grid slot) --
    # dropna() before sorting, or a NaN mixed with size-label strings blows up
    # sorted()'s comparisons.
    for size in sorted(out["model_size"].dropna().unique()):
        sub = out[out["model_size"] == size]
        families = sorted(sub.loc[~sub["is_base"], "dataset"].dropna().unique())
        base_flag = " (+ base model reference)" if sub["is_base"].any() else " (no base model yet)"
        print(f"  {size}: {families}{base_flag}")
    if n_external:
        ext_names = sorted(out.loc[out["is_external"], "model_name"].dropna().unique())
        print(f"  external (test-split spot-checks): {ext_names}")


if __name__ == "__main__":
    main()
