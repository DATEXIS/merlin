#!/usr/bin/env python3
"""Diagnostic: per-verifier-stage token counts across the MeRLIn instruction
datasets, to check whether the V2 (diagnose) vs V4 (ICD) tradeoff we flagged
could be explained (even partly) by token-count imbalance between verifier
stages during SFT rather than by task difficulty or label quality alone.

Background: all four verifier stages -- V1 EXTRACT (manifestations), V2
DIAGNOSE (differential diagnoses), V3 RERANK (re-rank differential, still
DiagnosesModel), V4 ICD (ICD-10 code assignment) -- see
src/pipeline/verifier_args.py's `prompt_types` / `schemas_` -- are trained
together in a single SFT run per checkpoint. src/fine_tuning/data_formatting.
to_text() renders every row, regardless of which verifier it came from, to
the same ChatML `text` column (system + user=Input + assistant=<think>
{Thinking}</think>\n\n{Output}), and src/fine_tuning/training.py concatenates
all of them into one train_data set for SFTTrainer. Loss is computed
token-by-token, assistant-span only (unsloth's train_on_responses_only masks
up to the "<|im_start|>assistant\n" marker). So if one verifier's rows are
systematically longer (more input context, longer <think> traces, longer
JSON outputs) than another's, that stage silently receives a larger share of
the total gradient signal per epoch purely from token volume -- independent
of how many *examples* of each stage are in the dataset. That's a mechanical
explanation for a V2-vs-V4 tradeoff worth ruling in/out before reaching for
a task-difficulty story.

This script reproduces the *exact* text each row is rendered to at train
time (same SYSTEM_PROMPT / build_assistant_content / chat template as
src/fine_tuning/data_formatting.to_text -- imported directly, not
reimplemented, so there's no risk of the two drifting apart) and tokenizes
it with the real SFT tokenizer, so the counts here are the real per-row
training sequence lengths, not an approximation from e.g. whitespace
splitting.

Requires network access to the tokenizer's HF repo (default
"Qwen/Qwen3-8B" -- Qwen3's tokenizer is shared across the 0.6b/8b/14b/32b
family, see src/fine_tuning/training.py's FastLanguageModel.from_pretrained
usage / scripts/eval_config.yaml's base_model list). Point --tokenizer at a
local checkpoint dir instead if HF Hub isn't reachable from where this runs.

Usage:
    python -m src.eval.verifier_token_counts
    python -m src.eval.verifier_token_counts \
        --inputs data/results/instructions/merlin_thr_full.pq \
                 data/results/instructions/merlin_thr_half.pq \
        --tokenizer Qwen/Qwen3-8B --split train
"""
import argparse
import sys
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from src.fine_tuning.data_formatting import (  # noqa: E402
    SYSTEM_PROMPT, build_assistant_content, set_qwen3_chat_template,
)

VERIFIER_NAMES = {
    1: "V1 extract (manifestations)",
    2: "V2 diagnose (differential)",
    3: "V3 rerank (differential)",
    4: "V4 icd (ICD-10 codes)",
}

# The multi-verifier MeRLIn instruction dataset variants (excludes the
# mimic_*.pq files, which are the note->ICD-only ablation with no V1-V3
# stages and so aren't relevant to a V2-vs-V4 *tradeoff* question).
DEFAULT_INPUTS = sorted(
    str(p) for p in (REPO / "data" / "results" / "instructions").glob("merlin_*.pq")
)

OUT_CSV = REPO / "data" / "results" / "instructions" / "verifier_token_counts.csv"


def load_tokenizer(name: str):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(name)
    set_qwen3_chat_template(tok)
    return tok


def render_full_text(system: str, user: str, assistant: str) -> str:
    """Manual reconstruction of QWEN3_CHAT_TEMPLATE's output (no
    add_generation_prompt). Kept as plain string formatting -- not a Jinja
    render per row -- purely for speed over ~100k-row datasets; a startup
    self-check (see `_verify_template_match`) confirms this matches
    tokenizer.apply_chat_template exactly before any counting happens, so a
    future edit to QWEN3_CHAT_TEMPLATE that this drifts from will fail loud
    rather than silently under/over-count."""
    return (
        f"<|im_start|>system\n{system}<|im_end|>\n"
        f"<|im_start|>user\n{user}<|im_end|>\n"
        f"<|im_start|>assistant\n{assistant}<|im_end|>\n"
    )


def _verify_template_match(tokenizer):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": "sample user turn"},
        {"role": "assistant", "content": "<think>\nsample\n</think>\n\nsample answer"},
    ]
    via_template = tokenizer.apply_chat_template(messages, tokenize=False)
    via_manual = render_full_text(SYSTEM_PROMPT, "sample user turn",
                                   "<think>\nsample\n</think>\n\nsample answer")
    if via_template != via_manual:
        raise RuntimeError(
            "render_full_text() no longer matches QWEN3_CHAT_TEMPLATE -- "
            "src/fine_tuning/data_formatting.py's chat template changed. "
            "Fix render_full_text() before trusting these token counts.\n"
            f"template : {via_template!r}\nmanual   : {via_manual!r}"
        )


def batch_token_lens(tokenizer, texts: list[str], batch_size: int = 2000) -> list[int]:
    lens = []
    for i in range(0, len(texts), batch_size):
        chunk = texts[i:i + batch_size]
        enc = tokenizer(chunk, add_special_tokens=False)
        lens.extend(len(ids) for ids in enc["input_ids"])
    return lens


def compute_row_token_counts(df: pd.DataFrame, tokenizer) -> pd.DataFrame:
    """One row per instruction example -> token counts for Input, Thinking,
    Output, the full assistant content (<think>...</think>\n\n + Output, or
    bare Output when Thinking is None), and the full rendered ChatML
    sequence (what actually gets loaded into the model's context)."""
    has_thinking_col = "Thinking" in df.columns
    inputs = df["Input"].astype(str).tolist()
    outputs = df["Output"].astype(str).tolist()

    if has_thinking_col:
        thinking_raw = df["Thinking"].tolist()
    else:
        thinking_raw = [None] * len(df)

    assistant_content = [
        build_assistant_content(t, o) if t is not None else o
        for t, o in zip(thinking_raw, outputs)
    ]
    full_text = [
        render_full_text(SYSTEM_PROMPT, i, a) for i, a in zip(inputs, assistant_content)
    ]
    # Thinking-only tokens: tokenize just the raw thinking string (None/""
    # both -> 0 tokens; this undercounts the literal <think></think> wrapper
    # tokens by a small constant, which full_tokens / assistant_tokens
    # already capture).
    thinking_texts = [t if t else "" for t in thinking_raw]

    out = pd.DataFrame({
        "Verifier": df["Verifier"].values,
        "split": df["split"].values if "split" in df.columns else "unknown",
        "input_tokens": batch_token_lens(tokenizer, inputs),
        "thinking_tokens": batch_token_lens(tokenizer, thinking_texts),
        "output_tokens": batch_token_lens(tokenizer, outputs),
        "assistant_tokens": batch_token_lens(tokenizer, assistant_content),
        "full_tokens": batch_token_lens(tokenizer, full_text),
    })
    return out


def summarize(counts: pd.DataFrame, dataset_name: str, split_filter: str | None) -> pd.DataFrame:
    df = counts if split_filter is None else counts[counts["split"] == split_filter]
    if df.empty:
        return pd.DataFrame()

    total_assistant = df["assistant_tokens"].sum()
    total_rows = len(df)

    rows = []
    for v, g in df.groupby("Verifier"):
        n = len(g)
        assistant_sum = g["assistant_tokens"].sum()
        row_share = n / total_rows
        token_share = assistant_sum / total_assistant if total_assistant else 0.0
        rows.append({
            "dataset": dataset_name,
            "verifier": VERIFIER_NAMES.get(v, f"V{v}"),
            "n_rows": n,
            "row_share_%": round(100 * row_share, 2),
            "total_assistant_tokens": int(assistant_sum),
            "assistant_token_share_%": round(100 * token_share, 2),
            # >1 means this verifier eats a bigger slice of the training
            # token budget than its example count alone would predict.
            "token_vs_row_share_ratio": round(token_share / row_share, 2) if row_share else float("nan"),
            "mean_input_tokens": round(g["input_tokens"].mean(), 1),
            "mean_thinking_tokens": round(g["thinking_tokens"].mean(), 1),
            "mean_output_tokens": round(g["output_tokens"].mean(), 1),
            "mean_assistant_tokens": round(g["assistant_tokens"].mean(), 1),
            "median_full_tokens": int(g["full_tokens"].median()),
            "p90_full_tokens": int(g["full_tokens"].quantile(0.9)),
            "max_full_tokens": int(g["full_tokens"].max()),
        })
    return pd.DataFrame(rows).sort_values(["dataset", "verifier"])


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--inputs", nargs="+", default=DEFAULT_INPUTS,
                         help="instruction .pq files to analyze (default: all merlin_*.pq "
                              "under data/results/instructions/)")
    parser.add_argument("--tokenizer", default="Qwen/Qwen3-8B",
                         help="HF tokenizer repo or local path (Qwen3 tokenizer is shared "
                              "across the 0.6b/8b/14b/32b family)")
    parser.add_argument("--split", default="train",
                         help="restrict to this split for the training-token-budget view "
                              "('train', or 'all' to include dev/val/test rows too)")
    parser.add_argument("--out-csv", default=str(OUT_CSV))
    args = parser.parse_args()

    if not args.inputs:
        print("No input files found/given.", file=sys.stderr)
        sys.exit(1)

    print(f"Loading tokenizer {args.tokenizer} ...")
    tokenizer = load_tokenizer(args.tokenizer)
    _verify_template_match(tokenizer)

    split_filter = None if args.split == "all" else args.split
    all_summaries = []
    all_counts = []

    for path in args.inputs:
        name = Path(path).stem
        print(f"\n{name}: loading {path}")
        wanted = ["Verifier", "split", "Input", "Thinking", "Output"]
        available = pq.read_schema(path).names
        needed = [c for c in wanted if c in available]
        missing = set(wanted) - set(needed) - {"Thinking"}  # Thinking is optional
        if missing:
            print(f"  WARNING: {name} is missing required column(s) {missing}, skipping.")
            continue
        df = pd.read_parquet(path, columns=needed)
        print(f"  {len(df):,} rows, verifiers present: "
              f"{sorted(df['Verifier'].unique().tolist())}")

        counts = compute_row_token_counts(df, tokenizer)
        counts["dataset"] = name
        all_counts.append(counts)

        summary = summarize(counts, name, split_filter)
        all_summaries.append(summary)

    counts_df = pd.concat(all_counts, ignore_index=True)
    summary_df = pd.concat(all_summaries, ignore_index=True)

    # Pooled view across all selected datasets together (helps answer "does
    # V2 eat more of the *overall* token budget than V4 across the board",
    # not just within one dataset variant).
    pooled = summarize(counts_df.assign(dataset="ALL_POOLED"), "ALL_POOLED", split_filter)
    summary_df = pd.concat([summary_df, pooled], ignore_index=True)

    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 20)
    print("\n" + "=" * 100)
    print(f"Per-verifier token counts (split={args.split})")
    print("=" * 100)
    print(summary_df.to_string(index=False))

    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(out_path, index=False)
    print(f"\nWrote {out_path}")

    # Quick flag: verifiers whose share of total training tokens is
    # meaningfully out of proportion to their share of training rows.
    imbalanced = summary_df[(summary_df["dataset"] == "ALL_POOLED")
                             & ((summary_df["token_vs_row_share_ratio"] > 1.15)
                                | (summary_df["token_vs_row_share_ratio"] < 0.87))]
    if not imbalanced.empty:
        print("\nToken budget is disproportionate to example count for (pooled):")
        print(imbalanced[["verifier", "row_share_%", "assistant_token_share_%",
                           "token_vs_row_share_ratio"]].to_string(index=False))


if __name__ == "__main__":
    main()
