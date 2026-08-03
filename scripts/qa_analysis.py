"""Driver for the two-stage QA error analysis.

stage1/stage2 are self-contained: they deploy the Qwen3-32B judge via the shared
eval server template, port-forward to it, join discharge notes, run all configured
result sets, write data/qa/* and mirror tables/artifacts to wandb (project
merlin_qa), then tear the server down (QA.server in qa_config.yaml controls this;
keep_alive: true leaves the server up between stages).

Stages (CLI selects the stage; everything else lives in scripts/qa_config.yaml,
consistent with scripts/eval_analysis.py):

    python scripts/qa_analysis.py stage1
        Induce a candidate error taxonomy from dev-split mispredictions.
        -> data/qa/stage1_rca.json, data/qa/taxonomy_candidate.json

    # human: review taxonomy_candidate.json, edit into taxonomy.json,
    #        set "approved": true.

    python scripts/qa_analysis.py stage2
        Score every test-split misprediction yes/no against the frozen taxonomy.
        -> data/qa/qa_classifications_*.parquet, data/qa/qa_error_shift.csv

    python scripts/qa_analysis.py validate --export
        Write a human-annotation template sampled from the stage-2 output.
    python scripts/qa_analysis.py validate --score
        Report judge-vs-human Cohen's kappa per class.

Set --config to point elsewhere; defaults to scripts/qa_config.yaml.
"""
import argparse
import asyncio
import os
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
# Make `src.*` importable no matter how the script is launched (laptop with an
# editable install, or bare `python3 scripts/qa_analysis.py` in the client pod).
sys.path.insert(0, str(REPO_ROOT))


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def main() -> None:
    parser = argparse.ArgumentParser(description="Two-stage QA error analysis")
    parser.add_argument("stage", choices=["stage1", "stage2", "validate", "server"])
    parser.add_argument("--config", default=str(REPO_ROOT / "scripts" / "qa_config.yaml"))
    parser.add_argument("--export", action="store_true", help="validate: write annotation template")
    parser.add_argument("--score", action="store_true", help="validate: score human vs judge")
    parser.add_argument("--up", action="store_true", help="server: deploy the judge and leave it running")
    parser.add_argument("--down", action="store_true", help="server: tear the judge down")
    args = parser.parse_args()

    cfg = load_config(args.config)
    repo_root = str(REPO_ROOT)

    if args.stage == "stage1":
        from src.qa.server import judge_endpoint
        from src.qa.taxonomy import run_stage1
        with judge_endpoint(cfg["QA"]) as base:
            asyncio.run(run_stage1(cfg, repo_root, base))
    elif args.stage == "stage2":
        from src.qa.server import judge_endpoint
        from src.qa.classify import run_stage2
        # Fail fast on an unapproved taxonomy BEFORE spinning up GPUs.
        from src.qa.classify import load_approved_taxonomy
        load_approved_taxonomy(repo_root, cfg["QA"])
        with judge_endpoint(cfg["QA"]) as base:
            asyncio.run(run_stage2(cfg, repo_root, base))
    elif args.stage == "validate":
        from src.qa.validate import export_template, score_agreement
        if args.export:
            export_template(cfg, repo_root)
        elif args.score:
            score_agreement(cfg, repo_root)
        else:
            parser.error("validate needs --export or --score")
    elif args.stage == "server":
        # Standalone judge lifecycle, for the in-cluster workflow: bring the
        # server up from the laptop, run the client as a k8s Job (which cannot
        # manage deployments itself), tear down when done.
        from src.qa.server import deploy_judge, teardown_judge
        if args.up:
            deploy_judge(cfg["QA"])
            print("Judge server is up. Launch the client job "
                  "(kubectl apply -f k8s/qa_client_job.yaml), then "
                  "`qa_analysis.py server --down` when finished.")
        elif args.down:
            teardown_judge(cfg["QA"])
        else:
            parser.error("server needs --up or --down")


if __name__ == "__main__":
    main()
