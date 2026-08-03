"""Minimal async vLLM chat client for the QA stages.

Kept separate from src/vllm_client.py (which is wired into the eval pipeline's
ExpArgs/VerifierArgs) so the QA analyses stay decoupled and easy to run offline.
Supports optional guided decoding via response_format=json_schema for stage 2.
"""
from __future__ import annotations

import asyncio
from typing import List, Optional, Type

import aiohttp
from pydantic import BaseModel
from tqdm.asyncio import tqdm_asyncio

SYSTEM_PROMPT = "You are a meticulous medical coding auditor. Answer only with the requested JSON."


def api_base(service_name: str, namespace: str, local: bool = False) -> str:
    if local:
        return "http://localhost:8000/v1"
    return f"http://{service_name}.{namespace}.svc.cluster.local/v1"


def _payload(model: str, prompt: str, temperature: float, max_tokens: int,
             seed: int, schema: Optional[Type[BaseModel]]) -> dict:
    payload = {
        "model": model,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "seed": seed,
        "n": 1,
        "stream": False,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        # Qwen3's chat template defaults to thinking mode; the <think> block eats
        # most of the token budget and truncates the JSON. vLLM forwards this to
        # the chat template (no-op for models that don't know the kwarg).
        "chat_template_kwargs": {"enable_thinking": False},
    }
    if schema is not None:
        payload["response_format"] = {
            "type": "json_schema",
            "json_schema": {
                "name": schema.__name__,
                "schema": schema.model_json_schema(),
                "strict": True,
            },
        }
    return payload


async def _model_id(base: str) -> str:
    timeout = aiohttp.ClientTimeout(total=15)  # fail fast; retry loop handles waiting
    async with aiohttp.ClientSession(timeout=timeout) as s:
        async with s.get(f"{base}/models") as r:
            data = await r.json()
            return data["data"][0]["id"]


async def resolve_model_id(base: str, timeout_min: float = 45) -> str:
    """Poll {base}/models until the server answers or the deadline passes.

    Deadline-based rather than attempt-based: model load for a 32B judge takes
    many minutes, so the budget is wall time (QA.server.ready_timeout_min).
    Loud about every failure so a dead port-forward or wrong URL is visible
    immediately, not after a silent retry loop.
    """
    import time
    deadline = time.monotonic() + timeout_min * 60
    backoff = 2
    attempt = 0
    last_err = None
    while time.monotonic() < deadline:
        attempt += 1
        try:
            model = await _model_id(base)
            print(f"[client] judge model at {base}: {model}")
            return model
        except Exception as e:
            last_err = e
            remaining = int((deadline - time.monotonic()) / 60)
            print(f"[client] {base}/models not reachable "
                  f"(attempt {attempt}, ~{remaining}min left of {timeout_min}min budget): "
                  f"{type(e).__name__}: {e} -- retrying in {backoff}s")
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, 30)
    raise RuntimeError(
        f"vLLM /models never became ready at {base} within {timeout_min}min "
        f"(last error: {last_err!r}). If the server pod is Ready, the connection "
        "path is wrong: check QA.server.connection (port-forward needs kubectl + "
        "kubeconfig on THIS machine; use 'in-cluster' when running inside a pod)."
    )


async def batch_complete(
    prompts: List[str],
    base: str,
    model: str,
    temperature: float = 0.0,
    max_tokens: int = 1500,
    seed: int = 42,
    concurrency: int = 32,
    schema: Optional[Type[BaseModel]] = None,
    desc: str = "qa",
    request_retries: int = 3,
) -> List[Optional[str]]:
    """Send all prompts, return raw text per prompt (None on failure).

    Failed requests are retried in full extra passes (with a pause and a server
    readiness check in between), so transient outages -- a port-forward dropping
    mid-run, a pod restart -- degrade to a delay instead of thousands of silent
    Nones. Errors are counted and summarised per pass, never swallowed silently.
    """
    url = f"{base}/chat/completions"
    sem = asyncio.Semaphore(concurrency)
    timeout = aiohttp.ClientTimeout(total=3600)
    results: List[Optional[str]] = [None] * len(prompts)

    async def one(session, idx, prompt, errors):
        async with sem:
            try:
                async with session.post(
                    url, json=_payload(model, prompt, temperature, max_tokens, seed, schema)
                ) as resp:
                    if resp.status != 200:
                        body = (await resp.text())[:300]
                        errors.append(f"HTTP {resp.status}: {body}")
                        return
                    data = await resp.json()
                    results[idx] = data["choices"][0]["message"]["content"]
            except Exception as e:
                errors.append(f"{type(e).__name__}: {e}")

    pending = list(range(len(prompts)))
    for attempt in range(1, request_retries + 2):  # first pass + retries
        if not pending:
            break
        if attempt > 1:
            print(f"[client] {desc}: retrying {len(pending)} failed requests "
                  f"(pass {attempt}/{request_retries + 1})")
            await asyncio.sleep(10)
            await resolve_model_id(base, timeout_min=15)  # wait out server/tunnel blips
        errors: List[str] = []
        async with aiohttp.ClientSession(timeout=timeout) as session:
            tasks = [one(session, i, prompts[i], errors) for i in pending]
            await tqdm_asyncio.gather(*tasks, desc=f"{desc}(p{attempt})")
        pending = [i for i in pending if results[i] is None]
        if errors:
            print(f"[client] {desc} pass {attempt}: {len(errors)} failures, "
                  f"e.g. {errors[0]}")
    ok = sum(r is not None for r in results)
    print(f"[client] {desc}: {ok}/{len(prompts)} succeeded")
    return results
