from pathlib import Path

import yaml

from src.kubernetes.yaml_spawner import shutdown_yaml


def shutdown_from_config(cfg_path: str):
    # Load config
    cfg = yaml.safe_load(Path(cfg_path).read_text())
    name = cfg["job_name"]

    # Resource names
    client_job = f"vllm-client-{name}"
    server_deployment = f"vllm-server-{name}"
    server_service = f"vllm-server-{name}"

    resources = [
        ("job", client_job),
        ("deployment", server_deployment),
        ("service", server_service),
    ]

    for kind, res in resources:
        shutdown_yaml(kind, res, cfg.get("namespace", "merlin"))


if __name__ == "__main__":
    shutdown_from_config('scripts/pipeline_config.yaml')
