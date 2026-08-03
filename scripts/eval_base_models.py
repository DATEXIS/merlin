#!/usr/bin/env python3
"""One-off downstream eval of raw (non-fine-tuned) Qwen3 base models.

Same deploy -> run -> teardown cycle as eval_checkpoints.py's eval_one(), but
driven by scripts/eval_config.yaml's flat `models:` list
instead of checkpoints.json manifests: no LoRA adapter, no epoch/step, no
resumability state file -- just N base models, each served directly
(Model.full_finetuning=False, Model.lora=False -> server_template.py serves
Model.base_model from the HF hub) and torn down again once its client job
finishes. Every model gets its own GPU type/count per the config, so all
entries run concurrently.

Run:
    python scripts/eval_base_models.py [--config scripts/eval_config.yaml]
"""

import argparse
import concurrent.futures as cf
import os
import subprocess

from src.kubernetes.yaml_spawner import (
    load_config, render_template, apply_yaml, shutdown_yaml,
    calculate_hardware_requirements, convert_lists_to_str,
    wait_for_job_complete, wait_for_deployment_ready, get_kubectl_path,
)


def build_cfg(base: dict, model: dict) -> dict:
    return {
        "job_name": model["job_name"],
        "namespace": base["namespace"],
        "server_image": base["server_image"],
        "client_image": base["client_image"],
        "load_from_checkpoint": False,
        "Hardware": {
            **base["Hardware"],
            "gpu_count": model["gpu_count"],
            "server_gpu": model["server_gpu"],
        },
        "Model": {**base["Model"], "base_model": model["base_model"]},
        "Client_Job": dict(base["Client_Job"]),
    }


def preclean(cfg: dict):
    """Remove any leftover server/service/client from a previous attempt at
    this job_name so a re-run starts clean (Job specs are immutable, so a
    stale client job would otherwise fail to re-apply)."""
    ns, name = cfg["namespace"], cfg["job_name"]
    kubectl = get_kubectl_path()
    for kind, res in (("job", f"vllm-client-{name}"),
                      ("deployment", f"vllm-server-{name}"),
                      ("service", f"vllm-server-{name}")):
        subprocess.run(
            [kubectl, "delete", kind, res, "-n", ns, "--ignore-not-found", "--wait=true"],
            env=os.environ.copy(), capture_output=True, text=True,
        )


def deploy_server(cfg: dict):
    cfg = calculate_hardware_requirements(cfg)
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


def eval_one(base: dict, model: dict) -> tuple:
    cfg = build_cfg(base, model)
    label = f"{model['job_name']} [{model['base_model']}] @{model['gpu_count']}x{model['server_gpu']}"
    ok = False
    try:
        print(f"[start] {label}")
        preclean(cfg)
        deploy_server(cfg)
        try:
            ok = run_client(cfg)
        finally:
            teardown_server(cfg)
    except Exception as e:  # don't let one bad model abort the siblings
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
    args = ap.parse_args()

    base = load_config(args.config)
    models = base.pop("models")

    with cf.ThreadPoolExecutor(max_workers=len(models)) as ex:
        futures = {ex.submit(eval_one, base, m): m for m in models}
        results = [fut.result() for fut in cf.as_completed(futures)]

    print("\n[summary]")
    for job_name, label, ok in results:
        print(f"  {'OK ' if ok else 'FAIL'} {label}")


if __name__ == "__main__":
    main()
