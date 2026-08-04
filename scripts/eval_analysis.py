#!/usr/bin/env python3
"""One entry point for the whole post-hoc eval analysis pipeline
(replaces evaluate_eval_results.py, checkpoint_metrics.py,
and the figure modules).

Configured via the `Analysis:` block of scripts/eval_config.yaml (project,
output_dir, primary_metric, metrics, plots, filter, ...) -- the command line
only picks WHICH stages run, matching how the rest of this repo's scripts are
driven by their config files rather than flags.

Stages (positional, run in this fixed order regardless of how you list them):

  wandb_check List which eval_results collections on WandB are new vs. the
                local metrics CSV -- no downloads, no evaluation. Use this to
                see whether anything finished since the last run.
  new_evals Download + locally re-evaluate every NEW collection, append
                the rows to the metrics CSV, then reshape it into the
                checkpoint-metrics CSV (src/eval/checkpoint_metrics).
                Resumable: already-evaluated collections are skipped
                (--force redoes everything).
  create_plots  Render every publication figure from the checkpoint-metrics
                CSV (src/eval/{main_figure_plots_macro, ablation_plots,
                epoch_tradeoff_plots, generation_plots, robustness_plots}).
  encoder_results
                Fold the encoder-classifier baselines (precomputed prediction
                parquets under the Encoder.results_dir config -- bypasses the
                vLLM/wandb pipeline entirely) into Encoder.results_test_csv.
                See src/eval/encoder_metrics.py. Independent of the other
                stages; runs wherever it's listed.
  clinibench Recompute CliniBench's own encoder baselines from their
                RELEASED per-admission predictions (config block `CliniBench`),
                on our test admissions, under one stated protocol -- so the
                paper's encoder comparison is ours end-to-end rather than a
                quote of their published numbers. Also re-verifies that we
                reproduce their Table 2 (recall/precision/MD-accuracy match to
                0.00 for three of four rows; the exceptions are documented in
                src/eval/clinibench_headtohead.py). Independent of the others.

Typical uses:
    python scripts/eval_analysis.py wandb_check
    python scripts/eval_analysis.py new_evals create_plots
    python scripts/eval_analysis.py new_evals --force
    python scripts/eval_analysis.py create_plots --config scripts/eval_config.yaml
    python scripts/eval_analysis.py encoder_results

Outputs land under Analysis.output_dir (default data/):
    eval_metrics_{project}.csv flat per-collection metrics
    results/evaluation/ reshaped CSV + best-epoch summary table
    (all figures -- quick-look and publication -- go to
    Analysis.paper_figures_dir instead, default figures/, not data/)

All the logic lives in src/eval/ (eval_results, checkpoint_metrics,
and the figure modules); this is just the CLI.
"""
import os

# ── OpenMP / threading guards — MUST be set before numpy/torch/sklearn import ──
# Prevents the macOS "OMP: Error #15... multiple copies of the OpenMP runtime"
# abort/segfault that occurs when torch and scikit-learn each load libomp.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import argparse
import logging
import sys
from pathlib import Path

import yaml

# Make `src...` importable no matter where the script is called from.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

STAGES = ["wandb_check", "new_evals", "create_plots", "encoder_results",
          "merlin"]
DEFAULT_CONFIG = REPO_ROOT / "scripts" / "eval_config.yaml"

# Fallbacks when the config has no Analysis block / omits a key.
DEFAULTS = {
    "entity": "anon-entity",
    "project": "merlin-eval-1.4",
    "output_dir": "data",
    "primary_metric": "ICD F1 Macro",
    "metrics": None,          # None -> DEFAULT_METRIC_SELECTION
    "plots": "both",
    "paper_figures_dir": "figures",
    "filter": None,
    "limit": None,
    "isolate": True,
    "as_percent": True,
}

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
log = logging.getLogger("eval_analysis")


def load_analysis_cfg(config_path: Path) -> dict:
    with open(config_path) as f:
        full = yaml.safe_load(f) or {}
    cfg = dict(DEFAULTS)
    cfg.update({k: v for k, v in (full.get("Analysis") or {}).items() if v is not None
                or k in ("filter", "limit", "metrics")})
    return cfg


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("stages", nargs="+", choices=STAGES, metavar="STAGE",
                   help=f"One or more of: {', '.join(STAGES)}. Always executed "
                        f"in that order, whatever order they're given in.")
    p.add_argument("--config", default=str(DEFAULT_CONFIG),
                   help="Config file with the Analysis block (default: scripts/eval_config.yaml).")
    p.add_argument("--force", action="store_true",
                   help="new_evals: ignore cached downloads and already-evaluated "
                        "rows; redo everything from scratch.")
    return p.parse_args()


def main():
    args = parse_args()
    cfg = load_analysis_cfg(Path(args.config))
    from src.eval import eval_results

    out_root = Path(cfg["output_dir"])
    if not out_root.is_absolute():
        out_root = REPO_ROOT / out_root
    metrics_csv = eval_results.default_metrics_csv(cfg["project"], out_root)
    # Moved from out_root/"checkpoint_analysis"
    # relocated the reshaped CSV + best-epoch table (and encoder_results/)
    # under data/results/ alongside the raw eval-artifact cache; that old
    # directory no longer exists.
    analysis_dir = out_root / "results" / "evaluation"
    checkpoint_csv = analysis_dir / "checkpoint_metrics.csv"
    paper_dir = Path(cfg["paper_figures_dir"])
    if not paper_dir.is_absolute():
        paper_dir = REPO_ROOT / paper_dir

    if "wandb_check" in args.stages:
        new, done = eval_results.find_new_collections(
            cfg["entity"], cfg["project"], metrics_csv, cfg["filter"], args.force)
        print(f"\n=== wandb_check — {cfg['entity']}/{cfg['project']} ===")
        print(f"{len(done)} collection(s) already evaluated in {metrics_csv.name}.")
        if new:
            print(f"{len(new)} NEW collection(s):")
            for c in new:
                print(f"  - {eval_results.experiment_name(c)}")
            print("\nRun `scripts/eval_analysis.py new_evals` to fetch + evaluate them.")
        else:
            print("Nothing new on WandB.")

    if "new_evals" in args.stages:
        df = eval_results.run_new_evals(
            cfg["entity"], cfg["project"], metrics_csv,
            download_dir=None,  # default: data/results/evaluation/{version}
            name_filter=cfg["filter"], limit=cfg["limit"],
            isolate=cfg["isolate"], as_percent=cfg["as_percent"], force=args.force)
        if df is None:
            sys.exit(1)
        # Reshape straight into the checkpoint-metrics CSV (one row per
        # (model_size, dataset, mode, epoch) + base/test rows).
        from src.eval.checkpoint_metrics import main as reshape
        print("\n--- reshaping into checkpoint metrics ---")
        reshape(in_csv=metrics_csv, out_dir=analysis_dir)

    if "create_plots" in args.stages:
        if not checkpoint_csv.exists():
            # Plots asked for but the reshaped CSV is missing -- try to build
            # it from an existing metrics CSV before giving up.
            from src.eval.checkpoint_metrics import main as reshape
            reshape(in_csv=metrics_csv, out_dir=analysis_dir)
        print("\n--- publication figures ---")
        from src.eval import (ablation_plots, epoch_tradeoff_plots, generation_plots,
                              main_figure_plots_macro, robustness_plots)
        for label, mod in [
            ("main_icd_by_size_macro", main_figure_plots_macro),
            ("dataset_ablation_8b_factorial", ablation_plots),
            ("epoch_tradeoff", epoch_tradeoff_plots),
            ("generation_plots", generation_plots),
        ]:
            print(f"  {label}")
            mod.main()
        print("  robustness_seed_variance")
        robustness_plots.run_all()

    if "encoder_results" in args.stages:
        with open(args.config) as f:
            encoder_cfg = (yaml.safe_load(f) or {}).get("Encoder", {})
        from src.eval.encoder_metrics import main as encoder_main
        print("\n--- encoder baselines -> results_test.csv ---")
        encoder_main(encoder_cfg)

    if "merlin" in args.stages:
        with open(args.config) as f:
            cb_cfg = (yaml.safe_load(f) or {}).get("CliniBench", {})
        from src.eval.clinibench_headtohead import run as clinibench_run
        print("\n--- CliniBench head-to-head (recomputed from their raw preds) ---")
        repro, table = clinibench_run(cb_cfg)
        print(repro.to_string(index=False))
        print()
        print(table.to_string(index=False))


if __name__ == "__main__":
    main()
