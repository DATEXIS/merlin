"""Judge-server lifecycle for the QA stages.

Reuses the shared eval infra (src/kubernetes/server_template.py via yaml_spawner)
so the judge is deployed exactly like an eval server: deployment + service named
`vllm-server-{job_name}`. Because qa_analysis.py runs on the workstation (not as
an in-cluster Job like the old run_analyses), we reach the service through an
automatic `kubectl port-forward` by default.

Connection modes (QA.server.connection):
  port-forward -- deploy if manage=true, port-forward svc -> localhost:{port}. Default.
  in-cluster   -- use the cluster-internal service DNS (script runs in a pod).
  localhost    -- assume something already listens on localhost:{port} (manual
                  port-forward or a locally launched vLLM); no deploy, no teardown.
"""
from __future__ import annotations

import contextlib
import os
import subprocess
import threading
import time
from typing import Iterator, Optional

# NOTE: src.kubernetes.yaml_spawner (jinja2, kubectl) is imported lazily inside
# the functions that deploy/teardown. The in-cluster client pod only *talks* to
# the judge and must not require deployment tooling in its image.


def _server_cfg(qa_cfg: dict) -> dict:
    """Assemble the cfg dict the shared server template expects."""
    from src.kubernetes.yaml_spawner import calculate_hardware_requirements
    srv = qa_cfg["server"]
    cfg = {
        "job_name": srv["job_name"],
        "namespace": srv["namespace"],
        "server_image": srv["server_image"],
        "Hardware": dict(srv["Hardware"]),
        "Model": dict(srv["Model"]),
    }
    return calculate_hardware_requirements(cfg)


def deploy_judge(qa_cfg: dict) -> None:
    from src.kubernetes.yaml_spawner import (
        apply_yaml, render_template, wait_for_deployment_ready,
    )
    cfg = _server_cfg(qa_cfg)
    apply_yaml(render_template(cfg, template_type="server"))
    wait_for_deployment_ready(
        f"vllm-server-{cfg['job_name']}", cfg["namespace"],
        cfg["Hardware"].get("replicas", 1),
    )


def teardown_judge(qa_cfg: dict) -> None:
    from src.kubernetes.yaml_spawner import shutdown_yaml
    srv = qa_cfg["server"]
    name = f"vllm-server-{srv['job_name']}"
    shutdown_yaml("deployment", name, srv["namespace"])
    shutdown_yaml("service", name, srv["namespace"])


def _running_in_cluster() -> bool:
    return bool(os.environ.get("KUBERNETES_SERVICE_HOST"))


class _PortForward:
    """Self-healing `kubectl port-forward`.

    A port-forward attaches to ONE pod; if that pod goes away (e.g. the rolling
    update after a spec change replaces it), the tunnel dies and every request
    gets connection-refused even though the deployment is Ready. A watchdog
    thread restarts the tunnel whenever the kubectl process exits.
    """

    def __init__(self, service: str, namespace: str, port: int):
        self.service, self.namespace, self.port = service, namespace, port
        self.proc: Optional[subprocess.Popen] = None
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def _spawn(self) -> None:
        from src.kubernetes.yaml_spawner import get_kubectl_path
        self.proc = subprocess.Popen(
            [get_kubectl_path(), "port-forward",
             f"svc/{self.service}", f"{self.port}:80", "-n", self.namespace],
            env=os.environ.copy(),
            stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True,
        )

    def start(self) -> None:
        self._spawn()
        time.sleep(3)
        if self.proc.poll() is not None:  # died immediately -> config problem, say why
            err = self.proc.stderr.read() if self.proc.stderr else ""
            raise RuntimeError(
                f"kubectl port-forward exited (code {self.proc.returncode}): {err.strip()}"
            )
        print(f"[server] port-forward svc/{self.service} -> localhost:{self.port} "
              f"(pid {self.proc.pid}, watchdog on)")
        self._thread = threading.Thread(target=self._watchdog, daemon=True)
        self._thread.start()

    def _watchdog(self) -> None:
        while not self._stop.wait(5):
            if self.proc.poll() is not None:
                err = self.proc.stderr.read() if self.proc.stderr else ""
                print(f"[server] port-forward died ({err.strip() or 'no stderr'}) "
                      "-- restarting tunnel")
                try:
                    self._spawn()
                except Exception as e:
                    print(f"[server] port-forward restart failed: {e}")

    def stop(self) -> None:
        self._stop.set()
        if self.proc is not None and self.proc.poll() is None:
            self.proc.terminate()


@contextlib.contextmanager
def judge_endpoint(qa_cfg: dict) -> Iterator[str]:
    """Yield the OpenAI-compatible base URL, managing server + tunnel as configured."""
    srv = qa_cfg["server"]
    mode = srv.get("connection", "port-forward")
    manage = srv.get("manage", True) and mode != "localhost"
    if _running_in_cluster():
        # Inside a pod: the service DNS is directly reachable (no tunnel), and
        # there is no kubectl/RBAC to manage the deployment -- the judge must be
        # deployed beforehand (scripts/qa_analysis.py server --up).
        if mode == "port-forward":
            print("[server] detected in-cluster execution -> using service DNS "
                  "instead of port-forward")
            mode = "in-cluster"
        if manage:
            print("[server] in-cluster: skipping deploy/teardown -- expecting "
                  "vllm-server-%s to be running (qa_analysis.py server --up)"
                  % srv["job_name"])
            manage = False
    port = srv.get("local_port", 8000)
    pf = None

    if manage:
        print(f"[server] deploying vllm-server-{srv['job_name']} "
              f"({srv['Model']['base_model']}) ...")
        deploy_judge(qa_cfg)

    try:
        if mode == "in-cluster":
            base = f"http://vllm-server-{srv['job_name']}.{srv['namespace']}.svc.cluster.local/v1"
        else:
            if mode == "port-forward":
                pf = _PortForward(f"vllm-server-{srv['job_name']}", srv["namespace"], port)
                pf.start()
            base = f"http://localhost:{port}/v1"
        print(f"[server] judge endpoint: {base}")
        yield base
    finally:
        if pf is not None:
            pf.stop()
        if manage and not srv.get("keep_alive", False):
            print("[server] tearing down judge server")
            teardown_judge(qa_cfg)
        elif manage:
            print("[server] keep_alive=true -- judge server left running")
