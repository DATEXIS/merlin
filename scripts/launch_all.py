#!/usr/bin/env python3
"""Launch every run in scripts/run_plan.yaml as a parallel k8s fine-tuning job.

One command fires the whole sweep; each run lands on its own GPUs and trains
concurrently. Per-run fields override the shared `defaults`.

Run:
    python scripts/launch_all.py                       # launch all
    python scripts/launch_all.py --only 32b 14b        # launch a subset (job_name substrings)
    python scripts/launch_all.py --dry_run             # print YAML, don't apply
"""

import argparse
import os
import subprocess

from src.kubernetes.yaml_spawner import (
    load_config, render_template, apply_yaml, shutdown_yaml, get_kubectl_path,
)

HARDWARE_KEYS = {"gpu", "gpu_count"}
TOP_KEYS = {"job_name"}


def build_cfg(plan: dict, run: dict) -> dict:
    """Merge shared defaults + per-run overrides into a fine_tuning template cfg."""
    training = dict(plan["defaults"])
    training.update({k: v for k, v in run.items()
                     if k not in HARDWARE_KEYS | TOP_KEYS})
    training["dataset_models"] = ",".join(training["dataset_models"])

    return {
        "job_name": run["job_name"],
        "project": plan["project"],
        "models_root": plan["models_root"],
        "Hardware": {"gpu": run["gpu"], "gpu_count": run["gpu_count"]},
        "training": training,
    }


def eff_batch(cfg: dict) -> int:
    t = cfg["training"]
    return t["batch_size"] * t["grad_accum"] * cfg["Hardware"]["gpu_count"]


def job_exists(name: str, namespace: str) -> bool:
    """True if a k8s Job with this name already exists (running, done, or failed)."""
    r = subprocess.run(
        [get_kubectl_path(), "get", "job", name, "-n", namespace,
         "-o", "jsonpath={.metadata.name}"],
        capture_output=True, text=True, env=os.environ.copy(),
    )
    return r.stdout.strip() == name


def clean_output_dir(plan: dict, job_name: str):
    """Wipe a run's checkpoint dir on the PVC (via the tool pod) so a restart
    doesn't accumulate stale checkpoints."""
    c = plan.get("cleanup")
    if not c:
        print(f"  (no 'cleanup' config; skipping dir wipe for {job_name})")
        return
    root = c["pvc_models_root"].rstrip("/")
    path = f"{root}/{job_name}"
    assert job_name and path.startswith(root + "/"), f"refusing to rm unsafe path: {path}"
    print(f"  wipe dir:       {c['tool_pod']}:{path}")
    subprocess.run(
        [get_kubectl_path(), "exec", "-n", plan["project"]["namespace"],
         c["tool_pod"], "--", "rm", "-rf", path],
        env=os.environ.copy(),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="scripts/run_plan.yaml")
    ap.add_argument("--only", nargs="*", default=None,
                    help="Only act on runs whose job_name contains any of these substrings.")
    ap.add_argument("--dry_run", action="store_true")
    ap.add_argument("--stop", action="store_true", help="Delete the run_plan jobs instead of launching.")
    ap.add_argument("--force", action="store_true",
                    help="Apply even if the job already exists (default: skip existing).")
    ap.add_argument("--clean", action="store_true",
                    help="Clean restart: delete the job + wipe its output dir, then (re)launch. "
                         "With --stop, also wipe dirs.")
    args = ap.parse_args()

    plan = load_config(args.config)
    runs = plan["runs"]
    if args.only:
        runs = [r for r in runs if any(s in r["job_name"] for s in args.only)]

    ns = plan["project"]["namespace"]
    if args.stop:
        print(f"Stopping {len(runs)} job(s) in {ns}:")
        for run in runs:
            shutdown_yaml("job", run["job_name"], ns)
            if args.clean:
                clean_output_dir(plan, run["job_name"])
        return

    print(f"Considering {len(runs)} run(s):")
    launched = skipped = 0
    for run in runs:
        cfg = build_cfg(plan, run)
        hw = cfg["Hardware"]
        name = run["job_name"]
        exists = (not args.dry_run) and job_exists(name, ns)
        if args.clean and not args.dry_run:
            # Clean restart: remove the existing job + stale checkpoints first.
            if exists:
                shutdown_yaml("job", name, ns)
            clean_output_dir(plan, name)
            exists = False
        if not args.force and exists:
            print(f"  skip (exists):  {name}")
            skipped += 1
            continue
        print(f"  launch:         {cfg['job_name']:<34} {hw['gpu_count']}x{hw['gpu']:<5} "
              f"{'LoRA r'+str(cfg['training']['lora_r']) if cfg['training']['use_lora'] else 'full':<10} "
              f"eff_batch={eff_batch(cfg)}")
        yaml_str = render_template(cfg, "fine_tuning")
        if args.dry_run:
            print(yaml_str)
        else:
            apply_yaml(yaml_str)
            launched += 1

    if not args.dry_run:
        print(f"\nLaunched {launched}, skipped {skipped} existing. "
              f"Monitor: kubectl get jobs -n {ns} | grep ft-qwen3")


if __name__ == "__main__":
    main()
