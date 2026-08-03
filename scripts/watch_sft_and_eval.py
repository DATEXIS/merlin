#!/usr/bin/env python3
"""Watch one or more SFT training Jobs (scripts/run_plan.yaml) and, as each
finishes successfully, immediately eval its epoch-3 checkpoint on the TEST
split -- no manual step, no per-epoch watcher needed.

Built for the seed-robustness check (2026-07-16): the two new seed runs
(ft-8b-full-thrfull2icd-seed43-v1-4, ft-8b-full-thrfull2icd-seed44-v1-4) both
train `epochs: 3` only (see run_plan.yaml) since epoch 3 is already the
dev-selected best epoch for this model/dataset (8b-full-thrfull-icd2, see
data/results/evaluation/checkpoint_best_epochs.csv) -- so each run produces
exactly one checkpoint and there's nothing to sweep across epochs for.

Reuses eval_checkpoints.py's read_manifest/select_checkpoints/eval_one
directly (same deploy-serve-eval-teardown path used by the dev sweep)
instead of shelling out to a hand-written throwaway eval config per seed.
Eval settings (Hardware/Model/Client_Job/pvc) come from --eval_config
(default scripts/eval_config.yaml) with two overrides applied per run:
  - Client_Job.split -> "test"   (paper test numbers; eval_config.yaml's
    default is "dev", used for epoch selection, not final reporting)
  - Client_Job.seed  -> the run's training seed (43 / 44), so each seed's
    eval sampling isn't accidentally identical across runs

short_name follows the existing convention (FAMILY_INFO in
src/eval/checkpoint_metrics.py): "8b-full-thrfull-icd2", with "-seed{N}"
appended ONLY when the training seed isn't the default 42. So results land
as test-8b-full-e3 for the original seed-42 run (unchanged, already in the
paper) and 8b-full-thrfull-icd2-seed43-e3 / -seed44-e3 for the two new ones.

Run (blocks until all listed jobs finish; evaluates each as it completes):
    python scripts/watch_sft_and_eval.py
    # or override the default seed-43/44 pair:
    python scripts/watch_sft_and_eval.py \\
        --runs ft-8b-full-thrfull2icd-seed43-v1-4:43 ft-8b-full-thrfull2icd-seed44-v1-4:44

Safe to leave running in a terminal/tmux session for as long as training
takes; polls every 10s (wait_for_job_complete's default) and does nothing
else in between.
"""

import argparse
import concurrent.futures as cf
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.kubernetes.yaml_spawner import load_config, wait_for_job_complete
from scripts.eval_checkpoints import read_manifest, select_checkpoints, eval_one

# The 2026-07-16 seed-robustness pair (see run_plan.yaml runs 14-15). Override
# with --runs for any other job_name:seed set.
DEFAULT_RUNS = [
    ("ft-8b-full-thrfull2icd-seed43-v1-4", 43),
    ("ft-8b-full-thrfull2icd-seed44-v1-4", 44),
]
BASE_SHORT_NAME = "8b-full-thrfull-icd2"
EPOCH = 3  # dev-selected best epoch for this model/dataset -- see header note


def short_name_for(seed: int) -> str:
    """Seed suffix only when it deviates from the default (42) -- keeps the
    original run's naming (test-8b-full-e3) unchanged."""
    return BASE_SHORT_NAME if seed == 42 else f"{BASE_SHORT_NAME}-seed{seed}"


def parse_runs(specs):
    out = []
    for s in specs:
        job_name, seed = s.rsplit(":", 1)
        out.append((job_name, int(seed)))
    return out


def watch_and_eval(job_name: str, seed: int, base: dict, namespace: str, models_root: str):
    print(f"[watch] waiting for training job {job_name} (seed {seed}) ...")
    ok = wait_for_job_complete(job_name, namespace)
    if not ok:
        print(f"[watch] {job_name} failed -- skipping eval.")
        return job_name, False

    run_base = dict(base)
    run_base["Client_Job"] = dict(base["Client_Job"], split="test", seed=seed)

    manifest_path = f"{models_root}/{job_name}/checkpoints.json"
    manifest = read_manifest(manifest_path, run_base)
    checkpoints = select_checkpoints(manifest, [EPOCH])
    if not checkpoints:
        print(f"[watch] {job_name}: no epoch-{EPOCH} checkpoint in manifest -- skipping eval.")
        return job_name, False

    short_name = short_name_for(seed)
    all_ok = True
    for ckpt in checkpoints:
        eval_job, label, eval_ok = eval_one(run_base, manifest, ckpt, short_name=short_name)
        print(f"[watch] eval {label} -> {'ok' if eval_ok else 'FAILED'}")
        all_ok = all_ok and eval_ok
    return job_name, all_ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval_config", default="scripts/eval_config.yaml",
                    help="Base Hardware/Model/Client_Job/pvc settings "
                         "(split and seed are overridden per run).")
    ap.add_argument("--runs", nargs="*", default=None,
                    help="job_name:seed pairs, e.g. "
                         "ft-8b-full-thrfull2icd-seed43-v1-4:43. "
                         "Default: the two 2026-07-16 seed-robustness runs.")
    ap.add_argument("--namespace", default=None,
                    help="Override namespace (default: eval_config's).")
    args = ap.parse_args()

    base = load_config(args.eval_config)
    namespace = args.namespace or base["namespace"]
    models_root = base["models_root"]
    runs = parse_runs(args.runs) if args.runs else DEFAULT_RUNS

    print(f"Watching {len(runs)} training job(s), evaluating epoch {EPOCH} on "
          f"the TEST split as each completes:")
    for job_name, seed in runs:
        print(f"  {job_name} (seed {seed}) -> short_name {short_name_for(seed)}")
    print()

    with cf.ThreadPoolExecutor(max_workers=len(runs)) as ex:
        futures = [ex.submit(watch_and_eval, job_name, seed, base, namespace, models_root)
                   for job_name, seed in runs]
        for fut in cf.as_completed(futures):
            job_name, ok = fut.result()
            print(f"[done] {job_name}: {'eval ok' if ok else 'skipped/failed'}")


if __name__ == "__main__":
    main()
