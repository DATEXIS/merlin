import ast
import json
from pathlib import Path

import numpy as np
import pandas as pd

# ── Helpers (ported from src/pipeline/instruction_builder.py) ─────────────────


def _convert_code_to_short_code(code: str, icd_version: int = 10) -> str:
    return code[:4] if icd_version == 9 and code.lower().startswith("e") else code[:3]


def _convert_text_to_thinking(text: str | None) -> str:
    """Extract the raw reasoning: drop everything after </think> or ``` and remove
    any leading/inner <think>/</think> tags so only untagged reasoning remains."""
    if text is None or (isinstance(text, float) and pd.isna(text)):
        return ""
    thinking = str(text).split("</think>")[0].split("```")[0]
    return thinking.replace("<think>", "").replace("</think>", "").strip("\n")


def _placeholder_reason(icd_code: str) -> str:
    """Fallback 'reason' for a ground-truth ICD code the model never predicted."""
    return f"Based on the admission note code {icd_code} is suspected."


def _placeholder_disease_reason(disease: str) -> str:
    """Fallback 'reason' for a ground-truth disease the model never predicted."""
    return f"Based on the admission note {disease} is suspected."


def _manipulate_icd_json(data, icd_list: list, icd_version: int = 10) -> list:
    """Filter/add entries in the ICD JSON to match the ground-truth icd_list."""
    if isinstance(data, str):
        data = ast.literal_eval(data)

    normalized = {_convert_code_to_short_code(c, icd_version) for c in icd_list}
    filtered, seen = [], set()
    for entry in data:
        short = _convert_code_to_short_code(entry["icd_code"].replace(".", ""), icd_version)
        if short in normalized:
            filtered.append(entry)
            seen.add(short)

    for code in icd_list:
        if _convert_code_to_short_code(code, icd_version) in (normalized - seen):
            filtered.append({"icd_code": code, "reason": _placeholder_reason(code)})

    return filtered


def _manipulate_diagnose_pred(json_list: list, target_name: str) -> list:
    """Move the disease matching target_name to the front of the list. If the
    model never predicted it at all, inject it at the front with a placeholder
    reason -- mirrors _manipulate_icd_json's handling of missing ground-truth
    codes, so the final target always contains the ground truth."""
    for i, item in enumerate(json_list):
        if item.get("name") == target_name:
            return [item] + json_list[:i] + json_list[i + 1:]
    return [{"name": target_name, "reason": _placeholder_disease_reason(target_name)}] + json_list


def _get_icd_thinking_string(icd_list: list[dict], thinking: str) -> str:
    thinking += "I think the patient has the following ICD codes:\n"
    thinking += "\n".join(f"{i+1}: {d['icd_code']}" for i, d in enumerate(icd_list))
    thinking += "\nAfter a final consideration, I think the patient has the following ICD codes:\n"
    return thinking


def _get_disease_thinking_string(disease_list: list[dict], thinking: str) -> str:
    thinking = thinking.strip("\n")
    thinking += "I think in order of likelihood this is the order of diseases:\n"
    thinking += "\n".join(f"{i+1}: {d['name']}" for i, d in enumerate(disease_list))
    thinking += "\nAfter final reranking the diseases, I think this is the order of diseases:\n"
    return thinking


# Maps verifier step -> column in the patients DataFrame that holds the ground-truth label
LABELS_STR = {
    1: "disease_vector",
    2: "disease",
    3: "disease",
    4: "ICD_CODES",
}

OUTPUT_COLUMNS = [
    "hadm_id",
    "subject_id",
    "split",
    "chief_complaint",
    "model",
    "Verifier",
    "Score",
    "trace",          # True = reasoning arm, False = mimic-style fallback (no <think>)
    "Input",
    "Thinking",       # "" on fallback rows -> data_formatting emits an empty <think></think> block
    "Output",
]

# Default per-verifier acceptance thresholds (reuse the generation thresholds so
# this is not a new free parameter). A row keeps its reasoning trace iff its
# score is > 0 and >= the threshold for its verifier; everything else (below
# threshold, score == 0, or score/JSON missing) falls back to the mimic-style
# no-reasoning target built from the ground-truth label.
GEN_THRESHOLDS = {1: 0.6, 2: 0.5, 3: 0.8, 4: 0.55}
GEN_THRESHOLDS_HALF = {v: t / 2 for v, t in GEN_THRESHOLDS.items()}
# thresholds=None means "only nan/0 fall back" (i.e. an effective threshold just
# above 0). Implemented as 0.0 combined with the strict score > 0 rule below.
NO_THRESHOLD = {1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0}


# ── JSON / ground-truth helpers ───────────────────────────────────────────────


def _parse_json(x):
    """Return a parsed list/dict, or None if the cell holds no usable JSON."""
    if x is None:
        return None
    if isinstance(x, float):          # bare NaN
        return None
    if isinstance(x, str):
        if x.strip() in ("", "None", "nan", "null", "NaN"):
            return None
        try:
            return ast.literal_eval(x)
        except Exception:
            try:
                return json.loads(x)
            except Exception:
                return None
    if isinstance(x, (list, dict)):
        return x
    if isinstance(x, np.ndarray):
        return list(x)
    return None


def _ground_truth_output(v: int, row) -> str | None:
    """Build a mimic-style (no-reasoning) target straight from the ground truth,
    used when a fallback row has no usable model JSON at all (true-nan rows).

    V4 -> ground-truth ICD codes, V2/V3 -> ground-truth disease. V1 has no clean
    schema to reconstruct from the label vector, so it is skipped (returns None
    and the row is dropped)."""
    if v == 4:
        codes = row.get("ICD_CODES")
        codes = list(codes) if isinstance(codes, np.ndarray) else codes
        if codes is None or len(codes) == 0:
            return None
        return json.dumps([{"icd_code": str(c), "reason": _placeholder_reason(str(c))} for c in codes],
                          ensure_ascii=False)
    if v in (2, 3):
        disease = row.get("disease")
        if disease is None or (isinstance(disease, float) and pd.isna(disease)):
            return None
        return json.dumps([{"name": str(disease), "reason": _placeholder_disease_reason(str(disease))}],
                          ensure_ascii=False)
    # v == 1: no clean reconstruction from disease_vector -> drop
    return None


# ── Row gathering / rendering ─────────────────────────────────────────────────


def _gather_raw_rows(
        patients: pd.DataFrame,
        model: str,
        duplicate_verifiers: tuple[int, ...] = (),
) -> list[dict]:
    """Flatten a merged per-model DataFrame into one raw record per
    (patient, verifier step), keeping the score and the raw materials needed to
    render either the reasoning or the fallback arm later.

    duplicate_verifiers: verifier steps to emit twice per patient (e.g. (4,)
    to get two independent V4/ICD-prediction rows per patient, for an
    ablation that oversamples that step). Each duplicate is rendered
    identically from the same underlying row -- this isn't resampling a
    different generation, just repeating the step's weight in the flat
    instruction set.
    """
    rows = []
    for _, row in patients.iterrows():
        verifier_steps = list(range(1, 5)) + list(duplicate_verifiers)
        for v in verifier_steps:
            rows.append(
                {
                    "hadm_id": row["hadm_id"],
                    "subject_id": row["subject_id"],
                    "split": row["split"],
                    "chief_complaint": row.get("_cc", ""),
                    "model": model,
                    "Verifier": v,
                    "Score": row.get(f"v{v}_score"),
                    "Input": row.get(f"v{v}_prompt"),
                    "_v_json": row.get(f"v{v}_json"),
                    "_v_text": row.get(f"v{v}_text"),
                    # ground-truth pieces for fallback synthesis
                    "ICD_CODES": row.get("ICD_CODES"),
                    "disease": row.get("disease"),
                }
            )
    return rows


def _render_row(raw: dict, thresholds: dict[int, float], drop_instead_of_fallback: bool = False) -> dict | None:
    """Turn a raw record into a final instruction row, choosing the reasoning or
    the mimic-style fallback arm. Returns None if the row must be dropped.

    drop_instead_of_fallback: if True, any row that would have gone to the
    mimic-style fallback (nan/0 score, or below threshold) is dropped entirely
    instead -- an ablation that removes the below-threshold/no-answer signal
    rather than replacing it with a no-reasoning ground-truth target."""
    v = raw["Verifier"]
    score = raw["Score"]
    obj = _parse_json(raw["_v_json"])
    has_score = pd.notna(score)

    keep_trace = (
        obj is not None
        and has_score
        and score > 0
        and score >= thresholds.get(v, 0.0)
    )

    if not keep_trace and drop_instead_of_fallback:
        return None

    if keep_trace:
        thinking = _convert_text_to_thinking(raw["_v_text"])
        output_obj = obj
        if score < 1.0:
            if v in (2, 3):
                thinking = _get_disease_thinking_string(output_obj, thinking)
                output_obj = _manipulate_diagnose_pred(output_obj, raw["disease"])
            elif v == 4:
                thinking = _get_icd_thinking_string(output_obj, thinking)
                output_obj = _manipulate_icd_json(output_obj, list(raw["ICD_CODES"]))
        thinking = thinking.strip("\n")
        output = json.dumps(output_obj, ensure_ascii=False)
        trace = True
    else:
        # ── mimic-style fallback: no reasoning, direct ground-truth target ──
        # Empty string (not None) -> renders as Qwen3's native empty
        # <think>\n\n</think>\n\n block. This is what lets you force
        # non-thinking generation at eval time (prime the prompt with that
        # same empty block after "<|im_start|>assistant\n"); the model only
        # reliably continues straight to the answer there if it actually saw
        # that pattern during SFT.
        thinking = ""
        if obj is not None:
            # We have a model answer: correct it toward the ground truth (same
            # alignment as the reasoning arm) but emit NO reasoning string.
            output_obj = obj
            if v in (2, 3):
                output_obj = _manipulate_diagnose_pred(output_obj, raw["disease"])
            elif v == 4:
                output_obj = _manipulate_icd_json(output_obj, list(raw["ICD_CODES"]))
            output = json.dumps(output_obj, ensure_ascii=False)
        else:
            # No usable model JSON (true-nan) -> synthesize from ground truth.
            output = _ground_truth_output(v, raw)
        trace = False

    if output is None:
        return None
    inp = raw["Input"]
    if inp is None or (isinstance(inp, float) and pd.isna(inp)):
        # No verifier prompt to condition on -> cannot build the row.
        return None

    return {
        "hadm_id": raw["hadm_id"],
        "subject_id": raw["subject_id"],
        "split": raw["split"],
        "chief_complaint": raw["chief_complaint"],
        "model": raw["model"],
        "Verifier": v,
        "Score": score,
        "trace": trace,
        "Input": inp,
        "Thinking": thinking,
        "Output": output,
    }


# ── Public API ────────────────────────────────────────────────────────────────


def build_instruction_dataset(
    gen_data_dir: str,
    output_path: str,
    thresholds: dict[int, float] | None = None,
    models: list[str] | None = None,
    best_per_id: bool = False,
    duplicate_verifiers: tuple[int, ...] = (),
    drop_below_threshold: bool = False,
) -> pd.DataFrame:
    """Build one flat instruction dataset with per-row reasoning/fallback routing.

    Parameters
    ----------
    gen_data_dir: str
        Directory with the merged per-model ``<model>.pq`` files (output of
        merge_gen_data.merge_all).
    output_path: str
        Destination parquet path.
    thresholds: dict[int, float] | None
        Per-verifier acceptance thresholds. A row keeps its reasoning trace iff
        ``score > 0 and score >= thresholds[verifier]``; otherwise it becomes a
        mimic-style no-reasoning row (or is dropped, see ``drop_below_threshold``).
        ``None`` means "only nan/0 fall back" (effective threshold just above 0).
    models: list[str] | None
        Which teacher models to include (per-model file stems). ``None`` = all.
    best_per_id: bool
        If True, keep only the highest-scoring model's row per
        (hadm_id, verifier) before rendering.
    duplicate_verifiers: tuple[int, ...]
        Verifier steps to include twice per patient per model (e.g. (4,) to
        oversample the ICD-prediction step). Incompatible with
        best_per_id=True: that dedups on (hadm_id, Verifier), which would
        just collapse the duplicates back down to one row.
    drop_below_threshold: bool
        Ablation switch. If True, every row that would otherwise fall back to
        the mimic-style no-reasoning target (nan/0 score, or score below its
        verifier's threshold) is dropped from the dataset instead of kept as a
        fallback row. Expected to underperform (imbalances verifier-step
        counts -- V4/ICD is hit hardest -- and drops difficult-sample signal
        entirely) but useful as a control for what the mimic fallback is
        actually buying.
    """
    if duplicate_verifiers and best_per_id:
        raise ValueError(
            "duplicate_verifiers is incompatible with best_per_id=True: "
            "the (hadm_id, Verifier) dedup would collapse the duplicates."
        )
    thresholds = NO_THRESHOLD if thresholds is None else thresholds
    root = Path(gen_data_dir)
    # Skip combine_models' output: combined.pq has no v*_prompt columns.
    model_files = [p for p in sorted(root.glob("*.pq")) if p.stem != "combined"]
    if models is not None:
        model_files = [p for p in model_files if p.stem in set(models)]
    if not model_files:
        raise FileNotFoundError(f"No matching per-model .pq files found in {root}")

    raw_rows: list[dict] = []
    for model_file in model_files:
        model = model_file.stem
        patients = pd.read_parquet(model_file)
        # Drop rows where all verifier scores are missing (pipeline never ran).
        score_cols = [f"v{v}_score" for v in range(1, 5)]
        present = [c for c in score_cols if c in patients.columns]
        if present:
            patients = patients.dropna(subset=present, how="all")
        raw_rows.extend(_gather_raw_rows(patients, model, duplicate_verifiers))
        print(f"  gathered {len(patients):,} patients from {model}")

    raw = pd.DataFrame(raw_rows)

    if best_per_id:
        before = len(raw)
        # NaN scores sort first, keep='last' keeps the highest non-nan score.
        raw = (
            raw.sort_values("Score", na_position="first")
            .drop_duplicates(subset=["hadm_id", "Verifier"], keep="last")
            .reset_index(drop=True)
        )
        print(f"  best-per-id: {before:,} -> {len(raw):,} rows")

    rendered = [r for r in (_render_row(row, thresholds, drop_below_threshold) for row in raw.to_dict("records"))
                if r is not None]
    out = pd.DataFrame(rendered, columns=OUTPUT_COLUMNS)

    n_trace = int(out["trace"].sum())
    n_fb = len(out) - n_trace
    print(f"  rendered {len(out):,} rows | reasoning={n_trace:,} fallback={n_fb:,}")
    print("  per-verifier reasoning/fallback:")
    for v in range(1, 5):
        sub = out[out.Verifier == v]
        print(f"    V{v}: total={len(sub):5d}  reasoning={int(sub.trace.sum()):5d}  "
              f"fallback={int((~sub.trace).sum()):5d}")

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(output_path, index=False)
    print(f"  saved -> {output_path}")
    return out


# ── Back-compat wrapper (old signature used by run_postprocessing) ────────────


def build_all_instructions(gen_data_dir: str, output_path: str) -> pd.DataFrame:
    """Legacy entry point: all models, no thresholding beyond nan/0 fallback."""
    return build_instruction_dataset(
        gen_data_dir=gen_data_dir,
        output_path=output_path,
        thresholds=None,
        models=None,
        best_per_id=False,
    )
