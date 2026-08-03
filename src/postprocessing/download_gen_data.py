import json
import os
import shutil
from datetime import datetime, timezone, timedelta
from pathlib import Path

import wandb

ENTITY = "anon-entity"
PROJECT = "merlin-generation"

MODELS = [
    "Qwen3-32B",
    "Llama-3.3-70B-Instruct",
    "medgemma-27b-it",
]

CHIEF_COMPLAINTS = [
    "back_pain",
    "dyspnea",
    "cough",
    "diarrhea",
    "chest_pain",
    "headache",
    "abdominal_pain",
]


def resolve_api_key(api_key: str | None = None, config_json: str = "config.json") -> str:
    if api_key:
        return api_key
    if os.environ.get("WANDB_API_KEY"):
        return os.environ["WANDB_API_KEY"]
    config_path = Path(config_json)
    if config_path.exists():
        with open(config_path) as f:
            cfg = json.load(f)
        if "WANDB_API_KEY" in cfg:
            print(f"Using API key from {config_path}")
            return cfg["WANDB_API_KEY"]
    raise RuntimeError(
        "No wandb API key found. Set API_KEY in the script, "
        "set WANDB_API_KEY env var, or place it in config.json."
    )


def _cutoff_date() -> datetime:
    return datetime.now(timezone.utc) - timedelta(days=61)


def _get_cc(tags: set) -> str | None:
    for cc in CHIEF_COMPLAINTS:
        if f"cc_{cc}" in tags:
            return cc
    return None


def _filter_runs(api: wandb.Api) -> list[tuple]:
    cutoff = _cutoff_date()
    matched = []
    for run in api.runs(f"{ENTITY}/{PROJECT}", per_page=200):
        tags = set(run.tags or [])
        model = next((m for m in MODELS if m in tags), None)
        cc = _get_cc(tags)
        if not model or not cc:
            continue
        created = datetime.fromisoformat(run.created_at.replace("Z", "+00:00"))
        if created > cutoff:
            continue
        matched.append((run, model, cc, created))
    return matched


def download(output_dir: str, dry_run: bool = False) -> None:
    api = wandb.Api()
    cutoff = _cutoff_date()
    output_root = Path(output_dir)

    print(f"Project : {ENTITY}/{PROJECT}")
    print(f"Cutoff  : runs created before {cutoff.date()} (>= 2 months old)")
    print(f"Output  : {output_dir}")
    print()

    matched = _filter_runs(api)

    if not matched:
        print("No matching runs found.")
        return

    print(f"Found {len(matched)} matching run(s):\n")
    for run, model, cc, created in matched:
        print(f"  [{model}] cc_{cc}  —  {run.name}  (created {created.date()}, state: {run.state})")

    if dry_run:
        print("\n[dry-run] No files downloaded.")
        return

    print()
    for run, model, cc, created in matched:
        print(f"\n→ {model} / cc_{cc}  [{run.name}]")

        artifacts = [a for a in run.logged_artifacts() if a.type == "dataset"]
        if not artifacts:
            print("    [no dataset artifact logged — skipping]")
            continue

        artifact = artifacts[0]
        dest = output_root / model
        dest.mkdir(parents=True, exist_ok=True)

        tmp_dir = dest / f"_tmp_cc_{cc}"
        print(f"    downloading artifact: {artifact.name}")
        artifact_dir = artifact.download(root=str(tmp_dir))

        files = [f for f in Path(artifact_dir).rglob("*") if f.is_file()]
        if not files:
            print("    [artifact empty — skipping]")
            shutil.rmtree(tmp_dir, ignore_errors=True)
            continue

        src = files[0]
        final_path = dest / f"cc_{cc}{src.suffix}"
        shutil.move(str(src), str(final_path))
        shutil.rmtree(tmp_dir, ignore_errors=True)
        print(f"    saved: {final_path.relative_to(output_root)}")

    print(f"\nDone. Files saved to: {output_root.resolve()}")
