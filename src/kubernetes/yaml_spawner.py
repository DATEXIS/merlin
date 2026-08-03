import logging
import os
import shutil

import yaml
import subprocess
import time
from jinja2 import Template

from src.kubernetes.client_template import client_template
from src.kubernetes.server_template import server_template
from src.kubernetes.fine_tuning_template import finetune_template


def load_config(path: str):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def get_kubectl_path():
    return shutil.which("kubectl") or "/opt/homebrew/bin/kubectl"


def calculate_hardware_requirements(cfg) -> dict:
    gpu_count = cfg['Hardware'].get('gpu_count', 1)
    cfg['memory_limit_server'] = gpu_count * 256
    cfg['memory_request_server'] = gpu_count * 16
    cfg['cpu_limit_server'] = gpu_count * 32
    cfg['cpu_request_server'] = gpu_count * 2
    return cfg


def convert_lists_to_str(cfg):
    # Preprocess lists so Jinja only sees ready-to-use strings
    temps = ", ".join(str(x) for x in cfg["Client_Job"]["temperatures"])
    budget = ", ".join(str(x) for x in cfg["Client_Job"]["budget"])
    max_tokens = ", ".join(str(x) for x in cfg["Client_Job"]["max_tokens"])
    thresholds = ", ".join(str(x) for x in cfg["Client_Job"]["thresholds"])

    cfg = cfg.copy()  # shallow copy to avoid mutating original
    cfg["Client_Job"]["budget_str"] = budget
    cfg["Client_Job"]["temperatures_str"] = temps
    cfg["Client_Job"]["max_tokens_str"] = max_tokens
    cfg["Client_Job"]["thresholds_str"] = thresholds
    return cfg


def render_template(cfg, template_type='server'):
    if template_type == 'server':
        tmpl = Template(server_template)
    elif template_type == 'client':
        tmpl = Template(client_template)
    elif template_type == 'fine_tuning':
        tmpl = Template(finetune_template)
    else:
        raise NotImplementedError(f"Unknown template type: {template_type}")
    return tmpl.render(cfg=cfg)


def apply_yaml(yaml_str):
    try:
        proc = subprocess.run([get_kubectl_path(), "apply", "-f", '-'],
                              input=yaml_str.encode(),
                              check=True,
                              env=os.environ.copy())
    except subprocess.CalledProcessError as e:
        logging.error(f"Error applying YAML: {e}")
        raise
    return proc


def shutdown_yaml(kind: str, name: str, namespace: str):
    print(f"Deleting {kind}/{name} ...")
    result = subprocess.run(
        [get_kubectl_path(), "delete", kind, name, "-n", namespace],
        env=os.environ.copy(),
        capture_output=True,
        text=True
    )
    if result.returncode == 0:
        print(result.stdout.strip())
    else:
        print(result.stderr.strip())


def wait_for_job(job_name, namespace="default"):
    while True:
        res = subprocess.run(
            [get_kubectl_path(), "get", "job", job_name, "-o", "jsonpath={.status.succeeded}", "-n", namespace],
            capture_output=True, text=True, env=os.environ.copy(),
        )
        if res.stdout.strip() == "1":
            print(f"✅ Job {job_name} finished successfully.")
            return True
        time.sleep(10)


def _jsonpath(kind, name, namespace, path):
    res = subprocess.run(
        [get_kubectl_path(), "get", kind, name, "-n", namespace, "-o", f"jsonpath={path}"],
        capture_output=True, text=True, env=os.environ.copy(),
    )
    return res.stdout.strip()


def wait_for_job_complete(job_name, namespace="default", poll=10):
    """Block until a Job succeeds (True) or fails (False)."""
    while True:
        if _jsonpath("job", job_name, namespace, "{.status.succeeded}") == "1":
            print(f"✅ Job {job_name} succeeded.")
            return True
        if _jsonpath("job", job_name, namespace, "{.status.failed}") not in ("", "0"):
            print(f"❌ Job {job_name} failed.")
            return False
        time.sleep(poll)


def wait_for_deployment_ready(name, namespace="default", replicas=1, poll=10):
    """Block until a Deployment reports the expected number of ready replicas."""
    print(f"⏳ Waiting for deployment {name} ({replicas} replicas) ...")
    while True:
        ready = _jsonpath("deployment", name, namespace, "{.status.readyReplicas}")
        if ready and int(ready) >= int(replicas):
            print(f"✅ Deployment {name} ready ({ready}/{replicas}).")
            return True
        time.sleep(poll)


def deploy_server(server_yaml: str, namespace: str):
    ns_flag = ["-n", namespace] if namespace else []
    logging.info("🚀 Deploying server...")
    subprocess.run([get_kubectl_path, "apply", "-f", server_yaml] + ns_flag, check=True)
    # Wait until the pod is Ready
    while True:
        result = subprocess.run(
            [get_kubectl_path(), "get", "deploy", "-o", "jsonpath={.items[0].status.readyReplicas}"] + ns_flag,
            capture_output=True,
            env=os.environ.copy(),
            text=True,
            )
        if result.stdout.strip() == "1":
            logging.info("✅ Server is ready.")
            break
        logging.info("⏳ Waiting for server to become ready...")
        time.sleep(10)


def shutdown_server(server_yaml: str, namespace: str, server_name: str):
    ns_flag = ["-n", namespace] if namespace else []
    logging.info("🛑 Shutting down server...")
    subprocess.run([get_kubectl_path(), "delete", "-f", server_yaml] + ns_flag, check=False,
        env=os.environ.copy(),)
    subprocess.run([get_kubectl_path(), "delete", "service", server_name] + ns_flag, check=False,
        env=os.environ.copy(),)


