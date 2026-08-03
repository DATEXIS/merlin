"""WandB eval-results download + local re-evaluation (the logic behind the
`wandb_check` and `new_evals` stages of scripts/eval_analysis.py; formerly
scripts/evaluate_eval_results.py).

Why this exists: some eval runs uploaded their results artifact successfully
but then crashed while computing metrics (NaN/None bug in
`evaluate_experiment`). The artifacts are intact, so we just re-evaluate them
locally here.

Resumable by default: a collection is skipped entirely (no download, no
re-evaluation) if (a) its `.pq` is already sitting under `download_dir` and
(b) it already has a row in the existing metrics CSV. Re-running after a
partial run -- e.g. picking up newly-finished eval_checkpoints.py /
eval_base_models.py jobs -- only fetches + evaluates what's new. `force=True`
ignores both caches.

Robustness:
  * OpenMP guard env vars are set *before* numpy/torch import (see
    scripts/eval_analysis.py) to avoid the macOS "libomp double-load"
    segfault when torch + scikit-learn are imported together.
  * Each artifact is evaluated in its own subprocess by default (invoked as
    `python -m src.eval.eval_results --single <collection>`): a native crash
    (segfault) on one artifact cannot take down the whole run. isolate=False
    runs everything in-process instead.
  * Collections without a `:latest` alias fall back to their newest version;
    empty collections are skipped with a warning.
"""
import json
import logging
import os
import re
import subprocess
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_ENTITY = "anon-entity"
DEFAULT_PROJECT = "merlin-eval-1.4"
ARTIFACT_TYPE = "eval_results"
CONFIG_JSON = REPO_ROOT / "config.json"
METRICS_MARKER = "__METRICS__ "  # subprocess protocol

# 2026-07-27: std (eval-time error-bar) seeds are done and 14b-full is stable
# again -- restrict the pipeline to the seeds we actually want feeding the
# metrics CSV: default seed 42 (no "-seed{N}" suffix at all) plus the std
# seeds 43/44 (see EVAL_SEED_TEST_FAMILY_RE / EVAL_SEED_TEST_BASE_RE in
# src/eval/checkpoint_metrics.py for why those show up as a doubled
# "-seed43-...-seed43" suffix). Anything else (e.g. a stray debug-seed
# collection left over from the 14b-full fix) is skipped before it's ever
# downloaded.
ALLOWED_SEEDS = {42, 43, 44}
SEED_RE = re.compile(r"-seed(\d+)")

log = logging.getLogger("eval_results")


def collection_seed(name: str) -> int:
    """Seed encoded in a collection name, e.g. 'test-06b-full-seed43-e4-seed43'
    -> 43. No '-seed{N}' suffix means the default seed-42 run."""
    m = SEED_RE.search(name)
    return int(m.group(1)) if m else 42


def filter_allowed_seeds(collections: list) -> list:
    """Keep only default-seed (42) and std (43/44) collections."""
    return [c for c in collections if collection_seed(c) in ALLOWED_SEEDS]


def default_metrics_csv(project: str, data_dir: Path = None) -> Path:
    """Derived from `project` so bumping it (e.g. for the next merlin
    version) can't silently leave the filename pointing at stale data."""
    return (data_dir or REPO_ROOT / "data") / f"eval_metrics_{project}.csv"


def default_download_dir(project: str) -> Path:
    # "merlin-eval-1.4" -> data/results/evaluation/1_4
    version = project.rsplit("-", 1)[-1].replace(".", "_")
    return REPO_ROOT / "data" / "results" / "evaluation" / version


def resolve_api_key() -> str:
    if os.environ.get("WANDB_API_KEY"):
        return os.environ["WANDB_API_KEY"]
    if CONFIG_JSON.exists():
        cfg = json.loads(CONFIG_JSON.read_text())
        if cfg.get("WANDB_API_KEY"):
            return cfg["WANDB_API_KEY"]
    raise RuntimeError(
        "No WandB API key found. Set WANDB_API_KEY env var or put it in config.json."
    )


def get_api():
    os.environ["WANDB_API_KEY"] = resolve_api_key()
    import wandb  # imported after the key is set
    return wandb.Api()


def experiment_name(collection_name: str) -> str:
    """Strip the `eval_results_` prefix for a cleaner row label."""
    prefix = "eval_results_"
    return collection_name[len(prefix):] if collection_name.startswith(prefix) else collection_name


def list_collections(api, entity: str, project: str):
    art_type = api.artifact_type(ARTIFACT_TYPE, project=f"{entity}/{project}")
    return [c.name for c in art_type.collections()]


def resolve_artifact(api, entity: str, project: str, collection_name: str):
    """Return a downloadable artifact for a collection, tolerating a missing
    `:latest` alias by falling back to the newest concrete version."""
    ref = f"{entity}/{project}/{collection_name}:latest"
    try:
        return api.artifact(ref, type=ARTIFACT_TYPE)
    except Exception as e:  # noqa: BLE001
        log.debug("  :latest lookup failed for %s (%s); trying explicit versions", collection_name, e)

    art_type = api.artifact_type(ARTIFACT_TYPE, project=f"{entity}/{project}")
    for c in art_type.collections():
        if c.name == collection_name:
            versions = list(c.artifacts())  # newest first
            if not versions:
                raise RuntimeError("collection has no versions (all deleted?)")
            return versions[0]
    raise RuntimeError("collection not found")


def _cached_pq_path(download_dir: Path, collection_name: str) -> Path | None:
    """Local .pq path for `collection_name` if already downloaded, else None.
    Checked before touching the WandB API at all, so a fully-cached run makes
    zero network calls."""
    art_dir = download_dir / collection_name
    if not art_dir.exists():
        return None
    pq_path = art_dir / f"{collection_name}.pq"
    if pq_path.exists():
        return pq_path
    candidates = list(art_dir.glob("*.pq"))
    return candidates[0] if candidates else None


def load_artifact_df(api, entity: str, project: str, download_dir: Path,
                     collection_name: str, force: bool = False) -> pd.DataFrame:
    pq_path = None if force else _cached_pq_path(download_dir, collection_name)
    if pq_path is not None:
        log.info("  using cached download for %s (%s)", collection_name, pq_path)
    else:
        artifact = resolve_artifact(api, entity, project, collection_name)
        art_dir = Path(artifact.download(root=str(download_dir / collection_name)))
        pq_path = art_dir / f"{collection_name}.pq"
        if not pq_path.exists():
            candidates = list(art_dir.glob("*.pq"))
            if not candidates:
                raise FileNotFoundError(f"No .pq file inside artifact for {collection_name}")
            pq_path = candidates[0]
    try:
        return pd.read_parquet(pq_path, engine="fastparquet")
    except Exception:
        return pd.read_parquet(pq_path)


def evaluate_one(api, entity: str, project: str, download_dir: Path,
                 collection_name: str, as_percent: bool, force: bool = False) -> dict:
    """Download (or reuse a cached download of) + evaluate a single
    collection; return a metrics dict."""
    from src.eval.eval_experiments import evaluate_experiment  # lazy: imports torch
    df = load_artifact_df(api, entity, project, download_dir, collection_name, force)
    metrics = dict(evaluate_experiment(df))
    if as_percent:
        metrics = {k: round(v * 100, 2) for k, v in metrics.items()}
    metrics["n_samples"] = len(df)
    return metrics


def evaluate_isolated(entity: str, project: str, download_dir: Path,
                      collection_name: str, as_percent: bool,
                      force: bool = False) -> tuple[dict | None, str | None]:
    """Run one evaluation in a child process (segfault isolation).
    Returns (metrics, error)."""
    cmd = [sys.executable, "-m", "src.eval.eval_results",
           "--single", collection_name,
           "--entity", entity, "--project", project,
           "--download-dir", str(download_dir),
           "--as-percent" if as_percent else "--raw"]
    if force:
        cmd.append("--force")
    proc = subprocess.run(cmd, capture_output=True, text=True,
                          env=os.environ.copy(), cwd=str(REPO_ROOT))
    for line in proc.stdout.splitlines():
        if line.startswith(METRICS_MARKER):
            return json.loads(line[len(METRICS_MARKER):]), None
    # No metrics -> figure out why.
    if proc.returncode < 0:
        err = f"child crashed with signal {-proc.returncode} (native/torch crash)"
    else:
        tail = (proc.stderr.strip().splitlines() or ["<no stderr>"])[-1]
        err = f"exit {proc.returncode}: {tail}"
    return None, err


def _load_done(out_csv: Path, force: bool):
    """Existing metrics CSV (or None) + set of already-evaluated row names."""
    if force or not out_csv.exists():
        return None, set()
    old_df = pd.read_csv(out_csv, index_col=0)
    return old_df, set(old_df.index)


def find_new_collections(entity: str = DEFAULT_ENTITY, project: str = DEFAULT_PROJECT,
                         out_csv: Path = None, name_filter: str = None,
                         force: bool = False) -> tuple[list, list]:
    """The `wandb_check` stage: scan wandb for eval_results collections and
    split them into (new, already_evaluated) vs the local metrics CSV."""
    out_csv = out_csv or default_metrics_csv(project)
    _, done_names = _load_done(out_csv, force)
    api = get_api()
    log.info("Scanning %s/%s for '%s' artifacts...", entity, project, ARTIFACT_TYPE)
    collections = list_collections(api, entity, project)
    if name_filter:
        collections = [c for c in collections if name_filter in c]
    collections = filter_allowed_seeds(collections)
    collections.sort()
    new = [c for c in collections if experiment_name(c) not in done_names]
    done = [c for c in collections if experiment_name(c) in done_names]
    return new, done


def run_new_evals(entity: str = DEFAULT_ENTITY, project: str = DEFAULT_PROJECT,
                  out_csv: Path = None, download_dir: Path = None,
                  name_filter: str = None, limit: int = None,
                  isolate: bool = True, as_percent: bool = True,
                  force: bool = False) -> pd.DataFrame | None:
    """The `new_evals` stage: download + evaluate every collection not yet in
    `out_csv`, append the rows, and return the full metrics DataFrame."""
    out_csv = out_csv or default_metrics_csv(project)
    download_dir = download_dir or default_download_dir(project)
    old_df, done_names = _load_done(out_csv, force)
    if old_df is not None:
        log.info("Found existing %s with %d row(s); already-evaluated collections "
                 "will be skipped (use --force to redo everything).", out_csv, len(old_df))

    new, done = find_new_collections(entity, project, out_csv, name_filter, force)
    if done:
        log.info("Skipping %d collection(s) already evaluated in %s.", len(done), out_csv)
    collections = new[:limit] if limit else new
    log.info("Evaluating %d collection(s).", len(collections))

    api = None if isolate else get_api()
    rows, failures = {}, {}
    for i, coll in enumerate(collections, 1):
        name = experiment_name(coll)
        log.info("[%d/%d] Evaluating %s", i, len(collections), name)
        try:
            if isolate:
                metrics, err = evaluate_isolated(entity, project, download_dir,
                                                 coll, as_percent, force)
                if err:
                    raise RuntimeError(err)
            else:
                metrics = evaluate_one(api, entity, project, download_dir,
                                       coll, as_percent, force)
            rows[name] = pd.Series(metrics, name=name)
        except Exception as e:  # noqa: BLE001
            log.error("  FAILED on %s: %s", name, e)
            failures[name] = str(e)

    if not rows and old_df is None:
        log.error("No artifacts were evaluated successfully.")
        for name, err in failures.items():
            log.error("  %s -> %s", name, err)
        return None

    new_df = pd.DataFrame(rows).T if rows else None
    if old_df is not None and new_df is not None:
        metrics_df = pd.concat([old_df, new_df])
    else:
        metrics_df = new_df if new_df is not None else old_df
    ordered = [c for c in metrics_df.columns if c != "n_samples"] + ["n_samples"]
    metrics_df = metrics_df[ordered].sort_index()

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    metrics_df.to_csv(out_csv)

    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 220)
    unit = "%" if as_percent else "raw"
    print(f"\n=== Eval metrics ({unit}) — {entity}/{project} ===\n")
    print(metrics_df.to_string())
    print(f"\nEvaluated {len(rows)} new artifact(s) ({len(done)} skipped as already-cached), "
          f"{len(metrics_df)} total rows. Saved to: {out_csv}")
    if failures:
        print(f"\n{len(failures)} artifact(s) failed:")
        for name, err in failures.items():
            print(f"  - {name}: {err}")
    return metrics_df


def _single_main():
    """Internal subprocess entrypoint: evaluate one collection, emit the
    metrics as a JSON marker line (see evaluate_isolated)."""
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--single", required=True)
    p.add_argument("--entity", default=DEFAULT_ENTITY)
    p.add_argument("--project", default=DEFAULT_PROJECT)
    p.add_argument("--download-dir", default=None)
    p.add_argument("--as-percent", dest="as_percent", action="store_true", default=True)
    p.add_argument("--raw", dest="as_percent", action="store_false")
    p.add_argument("--force", action="store_true")
    args = p.parse_args()
    download_dir = Path(args.download_dir) if args.download_dir else default_download_dir(args.project)
    api = get_api()
    metrics = evaluate_one(api, args.entity, args.project, download_dir,
                           args.single, args.as_percent, args.force)
    print(METRICS_MARKER + json.dumps(metrics))


if __name__ == "__main__":
    _single_main()
