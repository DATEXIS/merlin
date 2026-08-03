import asyncio
import logging
import os
import time
import traceback
from urllib.error import HTTPError

import httpx
import requests
import wandb
import aiohttp
from aiohttp import ClientSession, ClientTimeout
from openai import AsyncOpenAI
from tqdm.asyncio import tqdm_asyncio

from src.pipeline.prompt_builder import extract_admission_note_from_prompt
from src.exp_args import ExpArgs
from src.pipeline.verifier_args import VerifierArgs
from src.utils import is_model_chat_based

logger = logging.getLogger()


def get_api_config(service_name='vllm-server', namespace='merlin', local=False) -> dict:
    if local:
        api_base = "http://localhost:8000/v1"
    else:
        api_base = f"http://{service_name}.{namespace}.svc.cluster.local/v1"
    return {
        "service_name": service_name,
        "openai_api_key": os.getenv("OPENAI_API_KEY", 'openai-abc-key'),
        'openai_api_base': api_base,
        'openai_api_health_url': api_base[:-2] + "health",
    }


def check_connection(api_config: dict) -> bool:
    backoff_time = 1  # Start with 1 second
    num_tries = 0
    max_tries = 10000
    while num_tries <= max_tries:
        try:
            response = requests.get(api_config['openai_api_health_url'])
            if response.status_code == 200:
                return True
        except (requests.exceptions.RequestException, HTTPError):
            logging.info(f"Connect {num_tries} to {api_config['openai_api_base']}, "
                         f"retrying in {backoff_time}s")
            time.sleep(backoff_time)
            backoff_time = min(backoff_time * 2, 60)  # Exponential backoff (capped at 60s)
            num_tries += 1
    raise RuntimeError(f"Could not connect to vLLM server after {max_tries} attempts")


def get_endpoint(exp_args: ExpArgs) -> str:
    base = exp_args.api_config["openai_api_base"]
    if is_model_chat_based(exp_args.llm_name):
        return f"{base}/chat/completions"
    return f"{base}/completions"


async def get_model(api_config: dict) -> str:
    check_connection(api_config)
    client = AsyncOpenAI(
        api_key=api_config['openai_api_key'],
        base_url=api_config['openai_api_base'],
        timeout=httpx.Timeout(1000000),
    )
    backoff = 1
    while True:
        try:
            models = await client.models.list()
            for model in models.data:
                logging.info(f"Available model: {model.id}")
            return models.data[0].id
        except Exception as e:
            logging.info(f"Could not fetch /models ({e}), retrying in {backoff}s")
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, 60)


def build_payload(exp_args: ExpArgs, v_args: VerifierArgs, prompt: str,
                  chief_complaint: str = None) -> dict:
    payload = {
        "model": "lora_module" if exp_args.lora else exp_args.llm_name,
        "temperature": v_args.temperature,
        "n": v_args.num_choices,
        "max_tokens": v_args.max_tokens,
        "seed": exp_args.seed,
        "stream": False,
        "echo": False,
    }

    system_prompt = "You are a helpful medical assistant."

    if is_model_chat_based(exp_args.llm_name):
        payload["messages"] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ]
    else:
        payload["prompt"] = f"{system_prompt}\n{prompt}"

    if exp_args.guided_decoding:
        # Fetch the schema based on the specific chief complaint
        schema_class = v_args.get_pydantic_scheme(chief_complaint)
        payload["response_format"] = {
            "type": "json_schema",
            "json_schema": {
                "name": schema_class.__name__,
                "schema": schema_class.model_json_schema(),
                "strict": True
            }
        }
    return payload


def build_coroutine(session, exp_args: ExpArgs, v_args: VerifierArgs, prompt: str, chief_complaint: str = None):
    url = get_endpoint(exp_args)
    payload = build_payload(exp_args, v_args, prompt, chief_complaint)
    headers = {"Content-Type": "application/json"}

    async def request_coro():
        # 8 attempts with capped exponential backoff (2,4,8,16,32,60,60 ≈ 3 min
        # total) so a brief server restart / transient overload doesn't drop the
        # request. Connection errors are retried the same way as 5xx.
        max_retries = 8
        backoff = 2
        last_exc = None
        for attempt in range(max_retries):
            try:
                async with session.post(url, json=payload, headers=headers) as response:
                    if response.status in (500, 502, 503, 504):
                        text = await response.text()
                        last_exc = f"HTTP {response.status}: {text[:200]}"
                        if attempt < max_retries - 1:
                            logger.debug(f"Server error (attempt {attempt + 1}/{max_retries}), retrying in {backoff}s")
                            await asyncio.sleep(backoff)
                            backoff = min(backoff * 2, 60)
                            continue
                        logger.error(f"Request failed after {max_retries} attempts: {last_exc} | prompt: {prompt[:80]}...")
                        return None
                    response.raise_for_status()
                    return await response.json()
            except (aiohttp.ClientConnectorError, aiohttp.ServerDisconnectedError,
                    aiohttp.ClientOSError, asyncio.TimeoutError) as e:
                last_exc = repr(e)
                if attempt < max_retries - 1:
                    logger.debug(f"Connection error (attempt {attempt + 1}/{max_retries}), retrying in {backoff}s")
                    await asyncio.sleep(backoff)
                    backoff = min(backoff * 2, 60)
                else:
                    logger.error(f"Request failed after {max_retries} attempts: {last_exc} | prompt: {prompt[:80]}...")
                    return None
            except Exception:
                logger.error(f"Request failed for prompt: {prompt[:80]}...\n{traceback.format_exc()}")
                return None

    return request_coro, prompt


async def gather_with_concurrency(n, *coros_with_prompts):
    semaphore = asyncio.Semaphore(n)

    async def sem_wrapper(coro, prompt):

        async with semaphore:
            try:
                return await asyncio.wait_for(coro(), timeout=1800)
            except Exception:
                error_note = extract_admission_note_from_prompt(prompt)
                logger.error(f"Task failed for note: {error_note}")
                logger.error(traceback.format_exc())
                return None

    wrapped = [sem_wrapper(coro, prompt) for coro, prompt in coros_with_prompts]
    return await tqdm_asyncio.gather(*wrapped)


async def warm_up(exp_args, v_args: VerifierArgs, session: ClientSession) -> None:
    """Warm up the vLLM server with a simple prompt"""
    warmup_request = build_coroutine(session, exp_args, v_args, "Warmup test prompt")
    coro, _ = warmup_request
    await coro()


async def send_prompts(exp_args, v_args: VerifierArgs, prompts: list, chief_complaints: list, session) -> list:
    # No batching: Create one coroutine per prompt/complaint pair
    coroutines = [
        build_coroutine(session, exp_args, v_args, p, c)
        for p, c in zip(prompts, chief_complaints)
    ]
    return await gather_with_concurrency(v_args.concurrency, *coroutines)


def extract_text_from_responses(responses: list[dict], model: str) -> list:
    is_chat = is_model_chat_based(model)
    final_output = []

    for resp in responses:
        if not resp or "choices" not in resp:
            final_output.append([])
            continue

        # Extract content for all choices in this single response
        choices_text = []
        for choice in resp["choices"]:
            if is_chat:
                text = choice.get("message", {}).get("content", "")
            else:
                text = choice.get("text", "")
            choices_text.append(text)

        final_output.append(choices_text)

    return final_output


async def query_prompts(exp_args: ExpArgs, v_args: VerifierArgs, prompts: list,
                        chief_complaints: list) -> list:
    check_connection(exp_args.api_config)

    if exp_args.llm_name is None:
        exp_args.llm_name = await get_model(exp_args.api_config)
        wandb.log({'model': exp_args.llm_name})

    async with ClientSession(
        connector=aiohttp.TCPConnector(force_close=True),
        timeout=ClientTimeout(total=None, sock_connect=30, sock_read=1200),
    ) as session:
        await warm_up(exp_args, v_args, session)
        responses = await send_prompts(exp_args, v_args, prompts, chief_complaints, session)
        return extract_text_from_responses(responses, exp_args.llm_name)
