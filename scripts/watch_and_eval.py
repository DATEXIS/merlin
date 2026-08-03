#!/usr/bin/env python3
"""Watch one SFT training job and, the moment it finishes, eval whatever
checkpoints its manifest has -- no manual step once training completes.

Generic counterpart to watch_sft_and_eval.py (which watches a hardcoded pair
of seed-robustness runs at one fixed epoch). This one takes job_name /
short_name / epochs / split / Hardware overrides from a YAML --config instead,
so a new one-off watch is a new config file, not a new script or a pile of
CLI flags (see config_files_over_cli_flags feedback memory). Reuses
eval_checkpoints.py's read_manifest/select_checkpoints/eval_one directly --
same deploy-serve-eval-teardown path as the dev sweep in eval_config.yaml.

Config shape (see scripts/watch_and_eval_r256.yaml for a live example):
    job_name: ft-...-v1-4        # training Job name (scripts/run_plan.yaml)
    short_name: ...              # optional; eval jobs come out {short_name}-e{N}
    epochs: []                   # [] = whatever checkpoints exist once the job
                                  #   finishes; or a list like [3, 4]
    eval_config: scripts/eval_config.yaml   # base Hardware/Model/Client_Job/pvc
    Client_Job: {split: dev}     # optional overrides merged onto eval_config's
    Hardware: {server_gpu: b200, client_gpu: [p100, v100]}   # optional overrides

Run (blocks until the job completes, then evaluates each checkpoint as
listed above; safe to leave running in a terminal/tmux session for as long as
training takes -- polls every 10s, does nothing else in between):
    python scripts/watch_and_eval.py --config scripts/watch_and_eval_r256.yaml
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.kubernetes.yaml_spawner import load_config, wait_for_job_complete
from scripts.eval_checkpoints import read_manifest, select_checkpoints, eval_one


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True,
                    help="Watch config, e.g. scripts/watch_and_eval_r256.yaml.")
    ap.add_argument("--force", action="store_true",
                    help="Eval even if the training job reports failure "
                         "(e.g. to score whatever checkpoints exist after a "
                         "manual kill).")
    args = ap.parse_args()

    watch = load_config(args.config)
    job_name = watch["job_name"]
    short_name = watch.get("short_name")
    epochs = watch.get("epochs", [])

    base = load_config(watch.get("eval_config", "scripts/eval_config.yaml"))
    namespace = watch.get("namespace", base["namespace"])
    models_root = watch.get("models_root", base["models_root"])
    if "Client_Job" in watch:
        base["Client_Job"] = dict(base["Client_Job"], **watch["Client_Job"])
    if "Hardware" in watch:
        base["Hardware"] = dict(base["Hardware"], **watch["Hardware"])

    print(f"[watch] waiting for training job {job_name} ...")
    ok = wait_for_job_complete(job_name, namespace)
    if not ok and not args.force:
        print(f"[watch] {job_name} failed -- skipping eval (use --force to eval anyway).")
        sys.exit(1)
    if not ok:
        print(f"[watch] {job_name} failed -- evaluating anyway (--force).")

    manifest_path = f"{models_root}/{job_name}/checkpoints.json"
    manifest = read_manifest(manifest_path, base)
    checkpoints = select_checkpoints(manifest, epochs)
    if not checkpoints:
        print(f"[watch] {job_name}: no matching checkpoints in manifest -- nothing to eval.")
        sys.exit(1)

    print(f"[watch] {job_name} finished -- evaluating {len(checkpoints)} checkpoint(s) "
          f"on split={base['Client_Job']['split']}, "
          f"server_gpu={base['Hardware']['server_gpu']}, "
          f"client_gpu={base['Hardware']['client_gpu']}:")

    all_ok = True
    for ckpt in checkpoints:
        eval_job, label, eval_ok = eval_one(base, manifest, ckpt, short_name=short_name)
        print(f"[watch] eval {label} -> {'ok' if eval_ok else 'FAILED'}")
        all_ok = all_ok and eval_ok

    print(f"[done] {job_name}: {'all evals ok' if all_ok else 'some evals FAILED'}")
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
