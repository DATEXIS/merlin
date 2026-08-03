#!/usr/bin/env python3
"""Poll a checkpoints.json manifest and eval each epoch as it lands, instead
of waiting for the whole training job to finish.

Why this is a separate script from watch_and_eval.py / watch_sft_and_eval.py:
both of those block on wait_for_job_complete() and only look at checkpoints
once the k8s Job has fully succeeded -- fine when there's exactly one epoch
worth evaluating (watch_sft_and_eval.py's seed-43/44 thrfull2icd pair,
epochs:3 only) or when dev/test selection can wait for the run to finish.
Neither fits a run where every epoch needs a dev eval as soon as it's
written (e.g. the mimic-2icd seed-43/44 retrains, run_plan.yaml runs 16-17 --
the best epoch there is e4 at seed 42, i.e. the LAST epoch, so there's no
known early-stop point to skip, and it could shift at a new seed).

This is a config-driven revival of the older eval_watch_v1.4.py (dropped in
the 2026-07-16 config consolidation, commit 4fb6aa4) -- same polling logic,
generalized to take any manifest config via --config instead of a hardcoded
path, so a new one-off watch is a new config file (see
config_files_over_cli_flags feedback memory), not a new script.

eval_checkpoints.py reads every manifest ONCE at start and submits everything
it finds in one batch, then exits -- it has no way to notice a checkpoint SFT
writes an hour later. This wrapper just re-invokes it periodically. That's
safe because of eval_checkpoints.py's own state file
(data/results/eval_checkpoints_state.json): checkpoints already marked "ok"
are skipped, so re-running only ever picks up genuinely new work.

No hardware check needed: eval_checkpoints.py always submits as soon as a
checkpoint is ready -- the server Deployment's nodeSelector/nodeAffinity puts
it straight onto a free GPU if one's open, or leaves it Pending until one is.
k8s scheduling itself is the hardware gate.

Each poll only launches a NEW eval_checkpoints.py run if the previous one
(tracked via the Popen handle) has actually finished -- overlapping
invocations against the same manifests/state file would race in
preclean_checkpoint().

eval_checkpoints.py fail-loudly crashes its whole run if ANY configured
manifest's checkpoints.json doesn't exist yet (deliberate there, for a human
curating a static manifest list by hand). This watcher instead probes each
manifest itself first and writes a filtered "active" config
(data/results/_watch_epochs_active_<config stem>.yaml, gitignored along with
the rest of data/results/) containing only the manifests that exist right
now -- eval_checkpoints.py's fail-loud behavior stays intact for real
misconfigs, it just never sees a manifest we already know isn't there yet
(e.g. training hasn't started / hasn't checkpointed).

Run (from repo root):
    python scripts/watch_and_eval_epochs.py --config scripts/watch_epochs_mimic2icd_seed43.yaml
    python scripts/watch_and_eval_epochs.py --config <cfg> --interval 300  # poll every 5 min
    python scripts/watch_and_eval_epochs.py --config <cfg> --once          # single check, no loop
"""

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))

import eval_checkpoints as ec  # noqa: E402
from src.kubernetes.yaml_spawner import load_config  # noqa: E402


def active_config_path(config: str) -> str:
    """Per-config active-config path, so running this script twice at once
    against two different --config files (e.g. the seed-43 and seed-44 mimic
    watchers) never has both instances writing the same temp file out from
    under each other."""
    return f"data/results/_watch_epochs_active_{Path(config).stem}.yaml"


def scan_manifests(base: dict, state: dict):
    """Probe every manifest entry once. Returns (existing_entries, pending):
    existing_entries is the subset of base["manifests"] whose checkpoints.json
    currently exists on the PVC (SFT hasn't written it yet -- expected, not
    an error, so read_manifest's FileNotFoundError is swallowed here rather
    than propagating). pending is [(label, epoch), ...] for checkpoints in
    the existing manifests not yet marked "ok" in the state file."""
    existing_entries = []
    pending = []
    for entry in base["manifests"]:
        path = entry["path"] if isinstance(entry, dict) else entry
        short_name = entry.get("short_name") if isinstance(entry, dict) else None
        epochs = (entry["epochs"] if isinstance(entry, dict) and "epochs" in entry
                  else base.get("epochs", []))
        if not isinstance(epochs, list):   # allow scalar `epochs: 3` in the config
            epochs = [epochs]
        try:
            manifest = ec.read_manifest(path, base)
        except FileNotFoundError:
            continue
        existing_entries.append(entry)
        for ckpt in ec.select_checkpoints(manifest, epochs):
            job_name = ec.eval_job_name(manifest, ckpt, short_name)
            if state.get(job_name) != "ok":
                pending.append((short_name or manifest["job_name"], round(ckpt["epoch"])))
    return existing_entries, pending


def check_and_maybe_launch(config: str, state_file: str):
    """Returns a Popen handle if it launched eval_checkpoints.py, else None."""
    base = load_config(config)
    state = ec.load_state(state_file)
    existing_entries, pending = scan_manifests(base, state)

    if not pending:
        print("[watch] no new checkpoints waiting.", flush=True)
        return None

    active_base = dict(base)
    active_base["manifests"] = existing_entries
    active_config = active_config_path(config)
    Path(active_config).parent.mkdir(parents=True, exist_ok=True)
    with open(active_config, "w") as f:
        yaml.safe_dump(active_base, f, sort_keys=False)

    print(f"[watch] {len(pending)} checkpoint(s) ready across "
          f"{len(existing_entries)}/{len(base['manifests'])} existing manifest(s) -- "
          f"launching eval_checkpoints.py: {', '.join(f'{n} ep{e}' for n, e in pending)}",
          flush=True)
    # PYTHONUNBUFFERED so the child's own prints hit the shared log file
    # immediately too, not just this script's -- under `nohup ... > file &`,
    # stdout isn't a tty, so Python fully-buffers by default and output can
    # sit invisible in the buffer for a long time.
    child_env = dict(os.environ, PYTHONUNBUFFERED="1")
    return subprocess.Popen(
        [sys.executable, "scripts/eval_checkpoints.py",
         "--config", active_config, "--state_file", state_file],
        env=child_env,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True,
                    help="Same shape as eval_config.yaml -- manifests/epochs/"
                         "Hardware/Model/Client_Job/pvc.")
    ap.add_argument("--state_file", default=ec.STATE_FILE,
                    help="Local JSON file tracking which checkpoint evals already "
                         "succeeded (by job_name) -- shared with eval_checkpoints.py.")
    ap.add_argument("--interval", type=int, default=600,
                    help="Seconds between checks (default 600 = 10 min).")
    ap.add_argument("--once", action="store_true",
                    help="Check once and exit (e.g. for a cron job) instead of looping.")
    args = ap.parse_args()

    current = None  # subprocess.Popen of the currently-running eval_checkpoints.py, or None

    while True:
        if current is not None and current.poll() is None:
            print("[watch] previous eval_checkpoints.py run still in progress, skipping this cycle.",
                  flush=True)
        else:
            try:
                launched = check_and_maybe_launch(args.config, args.state_file)
                if launched is not None:
                    current = launched
            except Exception as e:
                print(f"[watch] error this cycle (will retry next poll): {e}", flush=True)

        if args.once:
            return
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
