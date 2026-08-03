"""Per-case response checkpointing for the QA stages.

A stage-1 run is ~5,500 judge calls; without checkpoints every hiccup costs a
full rerun. The cache stores raw judge responses keyed by "model|hadm_id" in a
parquet file and is flushed after every request chunk, so:

  * a rerun only requests cases that are missing (or whose cached response never
    parsed -- callers should only `put` validated responses),
  * a crash or Ctrl-C loses at most one chunk (QA.checkpoint_every cases).

Stage 2 caches are fingerprinted with a hash of the taxonomy (names + definitions
+ decision rules): editing the taxonomy invalidates the cache automatically
instead of silently reusing verdicts made against old class definitions.
"""
from __future__ import annotations

import hashlib
import json
import os
from typing import Dict, List, Optional

import pandas as pd


class ResponseCache:
    def __init__(self, path: str):
        self.path = path
        self._data: Dict[str, str] = {}
        if os.path.exists(path):
            df = pd.read_parquet(path)
            self._data = dict(zip(df["key"], df["raw"]))
            print(f"[cache] loaded {len(self._data)} cached responses from {path}")

    @staticmethod
    def key(model_name: str, hadm_id) -> str:
        return f"{model_name}|{hadm_id}"

    def get(self, key: str) -> Optional[str]:
        return self._data.get(key)

    def put(self, key: str, raw: str) -> None:
        if raw is not None:
            self._data[key] = raw

    def flush(self) -> None:
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        df = pd.DataFrame({"key": list(self._data), "raw": list(self._data.values())})
        tmp = self.path + ".tmp"
        df.to_parquet(tmp)
        os.replace(tmp, self.path)  # atomic -- a crash mid-write can't corrupt


def taxonomy_fingerprint(classes: List) -> str:
    """Short stable hash over the parts of the taxonomy that affect verdicts."""
    payload = [(c.name, c.definition, c.decision_rule) for c in classes]
    return hashlib.sha1(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:10]


def text_fingerprint(text: str) -> str:
    """Short stable hash over a prompt template. Used to fingerprint the stage-1
    RCA cache by STAGE1_RCA_PROMPT: editing the prompt (e.g. removing/adding
    keyword-style examples) changes what the judge produces, so old cached
    responses must not be silently reused under a different prompt -- the same
    reasoning as taxonomy_fingerprint for stage 2, applied one step earlier."""
    return hashlib.sha1(text.encode()).hexdigest()[:10]


def chunked(indices: List[int], size: int) -> List[List[int]]:
    return [indices[i:i + size] for i in range(0, len(indices), size)]
