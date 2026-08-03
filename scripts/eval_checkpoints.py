#!/usr/bin/env python3
"""Post-hoc downstream-task evaluation of SFT checkpoints.

For every checkpoint in the manifest(s), serve it with vLLM and run the full
generator-verifier pipeline once in eval_mode (each note once, num_choices=4)
over the whole eval set (all chief complaints, --split filters to dev+val).
Downstream metrics land in the 'merlin-eval' wandb project (init_wandb routes
eval_mode there).

Resumable: a local state file (--state_file, default STATE_FILE below) records
every checkpoint that finished successfully (keyed by its eval job_name, which
already encodes run+epoch+step). Re-running the script — e.g. after a laptop/VPN
drop killed a mid-sweep run — skips anything already marked done instead of
redeploying it. Use --force to ignore the state file and redo everything.

Config's `manifests` list entries are normally plain checkpoints.json paths
(default eval_job_name: eval-{run}-ep{N}-s{step}). An entry can instead be a
{path, short_name} dict for a shorter `{short_name}-e{N}` job name when every
eval in the config shares fixed params — see eval_config.yaml.

Concurrency: config's `max_parallel` bounds total concurrent checkpoints
(deploy+eval+teardown each held to one worker slot for their whole lifetime) —
the original, simple throttle. Config's `max_pending` (optional; takes
priority over max_parallel when set) is fair-share-aware instead: it only caps
how many servers may be simultaneously *unscheduled* (created but not yet
Ready) at once, via a semaphore acquired around deploy_server()'s apply+wait
and released the moment the pod goes Ready. Every ready checkpoint gets its
own worker thread immediately, so as many can run concurrently as the cluster
actually has room for — max_pending only throttles how big a backlog of
Pending pods this script leaves sitting in the scheduler queue, so someone
else's job isn't stuck behind a wall of your reservations. See
eval_config.yaml.

Run:
    python scripts/eval_checkpoints.py [--config scripts/eval_config.yaml]
"""

import argparse
import concurrent.futures as cf
import contextlib
import hashlib
import json
import os
import re
import subprocess
import threading
from pathlib import Path

# Default local state file: which checkpoint evals already succeeded (by
# job_name). data/results/ is already gitignored wholesale, so this never
# needs its own .gitignore entry.
STATE_FILE = "data/results/eval_checkpoints_state.json"

from src.kubernetes.yaml_spawner import (
    load_config, render_template, apply_yaml, shutdown_yaml,
    calculate_hardware_requirements, convert_lists_to_str,
    wait_for_job_complete, wait_for_deployment_ready, get_kubectl_path,
)


def sanitize(name: str) -> str:
    """RFC1123-safe k8s name fragment."""
    return re.sub(r"[^a-z0-9-]", "-", str(name).lower()).strip("-")


# k8s Service names AND label values (the "app: vllm-server-<name>" label used
# by server_template.py's Deployment/Service selector) are both capped at 63
# chars. "vllm-server-"/"vllm-client-" prefixes are 12 chars each, so the
# eval_job_name() output must stay <= 51 chars. Longer job_names (e.g. the
# *icd2x/*fixed* variants) blew past this: kubectl apply silently failed
# validation for that resource, so the later teardown then reported it as
# "NotFound" -- not a real cleanup problem, but a sign the checkpoint never
# actually got served/evaluated at all. 51 is also the exact budget already
# proven safe by every checkpoint in data/results/eval_checkpoints_state.json
# (longest existing entry is 50 chars) -- do not tighten this without checking
# that file, or already-succeeded checkpoints get silently re-hashed under a
# new key and re-evaluated for nothing.
_K8S_PREFIX_LEN = len("vllm-server-")  # == len("vllm-client-")
_K8S_NAME_BUDGET = 63 - _K8S_PREFIX_LEN

# Second-order issue, cosmetic only: when a Job name is near that 63-char
# ceiling, Kubernetes' own Pod-name generation (job_name + "-" + 5 random
# chars, then re-truncated to fit 63) chops characters off the END of the job
# name. If our disambiguating hash were also at the end, it gets chopped right
# back off -- every Pod from a long-named Job then looks identical in
# `k9s`/`kubectl get pods`, distinguished only by k8s's own random tail (this
# is what happened: job names ...-d7875e / ...-742f6e / ...-08f053 all render
# as indistinguishable pods). Putting the hash right after "eval-" instead of
# at the tail means it's very unlikely to be the part k8s truncates away.
_EVAL_PREFIX = "eval-"


def _fit_k8s_name(name: str, budget: int = _K8S_NAME_BUDGET) -> str:
    """Shorten `name` to fit `budget` chars. Only rewrites names that are
    already too long for a k8s object name -- anything under budget (i.e.
    every checkpoint evaluated so far) is returned byte-for-byte unchanged, so
    this never invalidates existing eval_checkpoints_state.json entries."""
    if len(name) <= budget:
        return name
    digest = hashlib.sha1(name.encode()).hexdigest()[:6]
    assert name.startswith(_EVAL_PREFIX), name  # see eval_job_name()
    rest = name[len(_EVAL_PREFIX):]
    rest_budget = budget - len(_EVAL_PREFIX) - len(digest) - 1
    return f"{_EVAL_PREFIX}{digest}-{rest[:rest_budget]}"


def read_manifest(path: str, base: dict) -> dict:
    """Read a checkpoints.json manifest from the PVC.

    Manifests live on the cluster PVC (pods see it at `pvc.ft_root`, e.g.
    /ft_models). This orchestrator runs locally, so it can't open that path
    directly — it reads through the tool pod, which mounts the SAME PVC at
    `pvc.tool_root` (e.g. /merlin_ddx). Falls back to a local open() if no
    `pvc:` block is configured.
    """
    pvc = base.get("pvc")
    if pvc:
        remote = path.replace(pvc["ft_root"], pvc["tool_root"], 1)
        out = subprocess.run(
            [get_kubectl_path(), "exec", "-n", base["namespace"], pvc["tool_pod"],
             "--", "cat", remote],
            capture_output=True, text=True, env=os.environ.copy(),
        )
        if out.returncode != 0:
            raise FileNotFoundError(
                f"Could not read {remote} from pod {pvc['tool_pod']}: {out.stderr.strip()}"
            )
        return json.loads(out.stdout)

    with open(path) as f:
        return json.load(f)


def select_checkpoints(manifest: dict, epochs: list) -> list:
    cps = manifest["checkpoints"]
    if epochs:
        wanted = {float(e) for e in epochs}
        cps = [c for c in cps if round(c["epoch"]) in wanted or c["epoch"] in wanted]
    return cps


def eval_job_name(manifest: dict, ckpt: dict, short_name: str = None) -> str:
    """The unique, deterministic id for one (checkpoint) eval. Used as both the
    k8s job_name and the state-file key. Kept <= 51 chars (see _fit_k8s_name) so
    "vllm-server-"/"vllm-client-" + this never breaks the k8s 63-char
    name/label-value limit.

    Default form: eval-{run}-ep{epoch}-s{step} (run + epoch + step -- unambiguous
    across configs with mixed params).

    If a manifest entry in the config sets `short_name` (e.g. "8b-full-mimic-icd2"
    -- see eval_config.yaml), use `{short_name}-e{epoch}` instead:
    no "eval-" prefix, no step. Safe when every eval in the config shares the same
    fixed params (so epoch alone disambiguates within a run) -- that's the case
    for a whole config file's worth of v1.4 evals, per the design notes convention."""
    if short_name:
        return _fit_k8s_name(sanitize(f"{short_name}-e{round(ckpt['epoch'])}"))
    run = sanitize(manifest["job_name"])
    full = sanitize(f"eval-{run}-ep{round(ckpt['epoch'])}-s{ckpt['global_step']}")
    return _fit_k8s_name(full)


def load_state(path: str) -> dict:
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        return json.load(f)


def save_state(path: str, state: dict):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(state, f, indent=2, sort_keys=True)


def build_cfg(base: dict, manifest: dict, ckpt: dict, short_name: str = None) -> dict:
    """Assemble a pipeline_config-shaped dict for one checkpoint (server-level)."""
    mode = manifest["finetuning_mode"]
    rel = os.path.relpath(ckpt["path"], base["models_root"])  # e.g. run-dir/checkpoint-N
    job_name = eval_job_name(manifest, ckpt, short_name)

    cfg = {
        "job_name": job_name,
        "namespace": base["namespace"],
        "server_image": base["server_image"],
        "client_image": base["client_image"],
        "models_root": base["models_root"],
        "load_from_checkpoint": False,
        "Hardware": dict(base["Hardware"]),
        "Model": {
            **base["Model"],
            "base_model": manifest["base_model"],
            "lora": mode == "lora",
            "full_finetuning": mode == "full",
            "lora_modules": rel,
            "max_lora_rank": manifest.get("lora_r", 256),
        },
        "Client_Job": dict(base["Client_Job"]),
    }
    return cfg


def preclean_checkpoint(cfg: dict):
    """Delete any leftover server/service/client from a previous run of THIS
    checkpoint, so a restart starts clean. Idempotent: --ignore-not-found means
    it's a no-op when nothing exists, and --wait blocks until the old resources
    (and their GPUs) are actually gone before we redeploy. Re-applying a Job with
    an existing name would otherwise fail (Job specs are immutable)."""
    ns = cfg["namespace"]
    name = cfg["job_name"]
    kubectl = get_kubectl_path()
    for kind, res in (("job", f"vllm-client-{name}"),
                      ("deployment", f"vllm-server-{name}"),
                      ("service", f"vllm-server-{name}")):
        subprocess.run(
            [kubectl, "delete", kind, res, "-n", ns, "--ignore-not-found", "--wait=true"],
            env=os.environ.copy(), capture_output=True, text=True,
        )


def deploy_server(cfg: dict, pending_semaphore: threading.Semaphore = None):
    """Apply the server Deployment and block until it's Ready.

    If `pending_semaphore` is given, it's held from just before the
    Deployment is created until the pod is actually Ready (Semaphore supports
    the context-manager protocol) -- i.e. exactly the "unscheduled" window.
    Once Ready, the semaphore is released immediately, before this returns, so
    the (potentially long) eval that follows never occupies a pending slot.
    """
    cfg = calculate_hardware_requirements(cfg)
    with (pending_semaphore or contextlib.nullcontext()):
        apply_yaml(render_template(cfg, template_type="server"))
        wait_for_deployment_ready(
            f"vllm-server-{cfg['job_name']}", cfg["namespace"], cfg["Hardware"]["replicas"]
        )


def run_client(cfg: dict) -> bool:
    cfg = convert_lists_to_str(cfg)
    apply_yaml(render_template(cfg, template_type="client"))
    job = f"vllm-client-{cfg['job_name']}"
    ok = wait_for_job_complete(job, cfg["namespace"])
    if not ok:
        # shutdown_yaml deletes the Job (and its pod) right below -- capture
        # its logs first or the failure reason is gone for good. This is why
        # the thrfull-icd2-e1 / mimic-icd2-e4 TypeError (get_hardware_tag on a
        # list server_gpu) had to be root-caused by reading code instead of
        # logs: both failed pods were already deleted by the time anyone looked.
        logs = subprocess.run(
            [get_kubectl_path(), "logs", f"job/{job}", "-n", cfg["namespace"], "--tail=200"],
            capture_output=True, text=True, env=os.environ.copy(),
        )
        print(f"[client-logs] {job} failed -- last 200 lines before teardown:\n"
              f"{logs.stdout}{logs.stderr}")
    shutdown_yaml("job", job, cfg["namespace"])
    return ok


def teardown_server(cfg: dict):
    ns = cfg["namespace"]
    shutdown_yaml("deployment", f"vllm-server-{cfg['job_name']}", ns)
    shutdown_yaml("service", f"vllm-server-{cfg['job_name']}", ns)


def eval_one(base: dict, manifest: dict, ckpt: dict, server_gpu: str = None,
             short_name: str = None, pending_semaphore: threading.Semaphore = None) -> tuple:
    """Serve + evaluate a single checkpoint end-to-end, then tear its server down.

    Each checkpoint gets a uniquely-named server deployment + client job
    (eval-<run>-ep<N>-s<step>, or `{short_name}-e<N>` -- see eval_job_name),
    so many can run concurrently on separate GPUs. `server_gpu`, if given,
    force-pins this checkpoint to one specific type instead of whatever the
    config's Hardware.server_gpu says (a single type stays a nodeSelector; a
    list becomes a nodeAffinity `In` block so Kubernetes itself picks whichever
    listed type is actually free -- see server_template.py). `pending_semaphore`,
    if given, throttles how many checkpoints may be simultaneously unscheduled
    (see deploy_server) without limiting how many may run concurrently.
    """
    cfg = build_cfg(base, manifest, ckpt, short_name)
    if server_gpu:
        cfg["Hardware"]["server_gpu"] = server_gpu
    gpu_label = cfg["Hardware"]["server_gpu"]
    gpu_label = gpu_label if isinstance(gpu_label, str) else f"any of {gpu_label}"
    label = (f"{manifest['job_name']} ep{round(ckpt['epoch'])} "
             f"s{ckpt['global_step']} [{manifest['finetuning_mode']}] "
             f"@{gpu_label}")
    ok = False
    try:
        print(f"[start] {label} -> {cfg['Model']['lora_modules']}")
        preclean_checkpoint(cfg)   # remove any orphaned server/job from a prior run
        deploy_server(cfg, pending_semaphore)
        try:
            ok = run_client(cfg)
        finally:
            teardown_server(cfg)
    except Exception as e:  # don't let one bad checkpoint abort the siblings
        print(f"[error] {label}: {e}")
        try:
            teardown_server(cfg)
        except Exception:
            pass
    print(f"[done]  {label} ok={ok}")
    return cfg["job_name"], label, ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="scripts/eval_config.yaml")
    ap.add_argument("--max_parallel", type=int, default=None,
                    help="Checkpoints to evaluate concurrently "
                         "(default: config 'max_parallel', else 4). Ignored if "
                         "--max_pending / config 'max_pending' is set.")
    ap.add_argument("--max_pending", type=int, default=None,
                    help="Cap on simultaneously-unscheduled (not-yet-Ready) "
                         "servers, for fair-share on a busy cluster -- every "
                         "ready checkpoint still gets its own worker and runs "
                         "as soon as it's scheduled, unlimited (default: "
                         "config 'max_pending', else unset = use --max_parallel "
                         "instead).")
    ap.add_argument("--cleanup", action="store_true",
                    help="Delete this config's server/service/client resources and exit "
                         "(no eval). Use to wipe a messy/interrupted run before restarting.")
    ap.add_argument("--state_file", default=STATE_FILE,
                    help="Local JSON file tracking which checkpoint evals already "
                         "succeeded (by job_name). Restarting the script skips them.")
    ap.add_argument("--force", action="store_true",
                    help="Ignore the state file and re-run every checkpoint, "
                         "including ones already marked done.")
    args = ap.parse_args()
    base = load_config(args.config)
    state = load_state(args.state_file)

    tasks = []
    for entry in base["manifests"]:
        # A manifest entry is either a plain path string (default eval_job_name,
        # e.g. eval-ft-8b-full-mimic-2icd-v1-4-ep1-s...), or a dict {path,
        # short_name} to use the short `{short_name}-e{epoch}` naming instead
        # (see eval_config.yaml). A dict entry may additionally
        # set `server_gpu` (overrides Hardware.server_gpu for just this
        # manifest's checkpoints; string or list, same semantics as the global)
        # and `epochs` (overrides the global `epochs` filter for just this
        # manifest -- used by the test-set config to pin each run to its single
        # dev-selected epoch; see the test-split config).
        manifest_path = entry["path"] if isinstance(entry, dict) else entry
        short_name = entry.get("short_name") if isinstance(entry, dict) else None
        server_gpu = entry.get("server_gpu") if isinstance(entry, dict) else None
        epochs = (entry["epochs"] if isinstance(entry, dict) and "epochs" in entry
                  else base.get("epochs", []))
        if not isinstance(epochs, list):   # allow scalar `epochs: 3` in the config
            epochs = [epochs]
        manifest = read_manifest(manifest_path, base)
        checkpoints = select_checkpoints(manifest, epochs)
        print(f"=== {manifest['job_name']}: {len(checkpoints)} checkpoint(s) ===")
        tasks.extend((manifest, ckpt, short_name, server_gpu) for ckpt in checkpoints)

    if not args.force:
        before = len(tasks)
        tasks = [(m, c, s, g) for m, c, s, g in tasks
                 if state.get(eval_job_name(m, c, s)) != "ok"]
        skipped = before - len(tasks)
        if skipped:
            print(f"\nSkipping {skipped} checkpoint(s) already marked done in "
                  f"{args.state_file} (use --force to redo).")

    if args.cleanup:
        print(f"\nCleaning up {len(tasks)} checkpoint(s)' resources ...")
        for manifest, ckpt, short_name, _ in tasks:
            cfg = build_cfg(base, manifest, ckpt, short_name)
            print(f"  clean {cfg['job_name']}")
            preclean_checkpoint(cfg)
        print("Cleanup done.")
        return

    max_pending = args.max_pending or base.get("max_pending")

    # Server-GPU placement: build_cfg() already copies base["Hardware"]
    # (including server_gpu) into every checkpoint's cfg untouched, so nothing
    # extra is needed here. A single string ("b200") renders as a nodeSelector
    # (unchanged, exact match). A list (["b200","h200","h100"]) renders as a
    # nodeAffinity `operator: In` block instead (see server_template.py) --
    # true OR semantics, so Kubernetes' own scheduler places each pod on
    # whichever listed type is actually free when it schedules.
    #
    # This used to be resolved client-side instead: cycle the list by task
    # index (gpus[i % len(gpus)]) to spread checkpoints across types. That
    # broke in practice under the watcher -- it usually only finds 1 new
    # checkpoint per poll, so the index restarts at 0 every invocation and
    # everything lands on the first listed type (b200), even while h200 sits
    # idle. Letting the scheduler pick via nodeAffinity fixes that properly
    # instead of us trying to out-guess cluster occupancy from here.

    if max_pending:
        # Fair-share mode: every ready checkpoint gets a worker immediately
        # (worker count = task count, effectively unbounded), but deploy_server()
        # gates the create-and-wait-for-Ready phase on a semaphore of size
        # max_pending -- so at most max_pending are ever simultaneously
        # unscheduled, while as many as the cluster has room for run concurrently.
        pending_semaphore = threading.Semaphore(max_pending)
        worker_count = max(len(tasks), 1)
        print(f"\nEvaluating {len(tasks)} checkpoint(s): unlimited concurrent once "
              f"scheduled, at most {max_pending} pending (unscheduled) at a time.")
    else:
        pending_semaphore = None
        worker_count = args.max_parallel or base.get("max_parallel", 4)
        print(f"\nEvaluating {len(tasks)} checkpoint(s), {worker_count} at a time.")

    default_gpus = base["Hardware"]["server_gpu"]
    for m, c, s, g in tasks:
        gpus = g if g is not None else default_gpus
        gpu_summary = gpus if isinstance(gpus, str) else f"any of {gpus}"
        print(f"  {eval_job_name(m, c, s)} -> {gpu_summary}")
    print()

    results = []
    with cf.ThreadPoolExecutor(max_workers=worker_count) as ex:
        futures = [ex.submit(eval_one, base, m, c, g, s, pending_semaphore)
                   for m, c, s, g in tasks]
        for fut in cf.as_completed(futures):
            job_name, label, ok = fut.result()
            results.append((label, ok))
            # Persist immediately (not just at the end) so a crash mid-sweep
            # doesn't lose credit for checkpoints that already finished.
            if ok:
                state[job_name] = "ok"
                save_state(args.state_file, state)

    print("\nSummary:")
    for label, ok in sorted(results):
        print(f"  {'OK  ' if ok else 'FAIL'} {label}")


if __name__ == "__main__":
    main()
