"""Data formatting for MERLIN-DDX SFT.

We render each example to a single ``text`` column with a plain ChatML template
(what Unsloth's SFTTrainer expects). Assistant-only loss is applied afterwards by
unsloth.chat_templates.train_on_responses_only, which masks everything up to and
including the ``<|im_start|>assistant\\n`` marker. The Qwen3 thinking block is
emitted explicitly as ``<think>\\n...\\n</think>\\n\\n{answer}``.
"""

from datasets import load_from_disk


SYSTEM_PROMPT = ("You are a medical expert with advanced knowledge in clinical reasoning, "
                 "diagnostics, and ICD-10-CM coding.")

# Plain ChatML / Qwen3 template. Masking is handled by train_on_responses_only
# (marker-based), so no {% generation %} tags are needed here.
QWEN3_CHAT_TEMPLATE = (
    "{%- for message in messages %}"
    "{%- if message['role'] == 'system' %}"
    "{{- '<|im_start|>system\n' + message['content'] + '<|im_end|>\n' }}"
    "{%- elif message['role'] == 'user' %}"
    "{{- '<|im_start|>user\n' + message['content'] + '<|im_end|>\n' }}"
    "{%- elif message['role'] == 'assistant' %}"
    "{{- '<|im_start|>assistant\n' + message['content'] + '<|im_end|>\n' }}"
    "{%- endif %}"
    "{%- endfor %}"
    "{%- if add_generation_prompt %}{{- '<|im_start|>assistant\n' }}{%- endif %}"
)

# Markers consumed by train_on_responses_only for assistant-only loss.
INSTRUCTION_PART = "<|im_start|>user\n"
RESPONSE_PART = "<|im_start|>assistant\n"

EVAL_SPLITS = ("dev", "val")


def set_qwen3_chat_template(tokenizer):
    """Install the plain ChatML template on the tokenizer (also saved with the model)."""
    tokenizer.chat_template = QWEN3_CHAT_TEMPLATE
    return tokenizer


def build_assistant_content(thinking: str, output: str) -> str:
    """Qwen3 thinking format: a single think block followed by the answer."""
    thinking = (thinking or "").strip("\n")
    return f"<think>\n{thinking}\n</think>\n\n{output}"


def to_text(batch, tokenizer):
    """Render a raw instruction batch to a single ChatML ``text`` column.

    A row can be rendered *without* a <think> block (bare answer, no tags at
    all) only when the whole dataset has no ``Thinking`` column (e.g. the
    MIMIC no-reasoning ablation: note -> ICD directly).

    Per-row "mimic-style / trace dropped" fallback rows in the mixed MERLIN
    datasets (built by build_instructions.py) carry ``Thinking = ""``
    (empty string, not None) and render as an empty ``<think>\n\n</think>``
    block followed by the answer — Qwen3's native non-thinking convention.
    This is deliberate: it's what lets you force non-thinking generation at
    eval time by priming the prompt with that same empty block after
    ``<|im_start|>assistant\n`` — the model only reliably continues straight
    to the answer there if it saw that exact pattern during SFT. A truly
    ``None`` Thinking value still drops the block entirely (bare answer),
    kept for the no-``Thinking``-column case above.
    """
    inputs, outputs = batch["Input"], batch["Output"]
    thinking = batch.get("Thinking")
    texts = []
    for i in range(len(inputs)):
        row_thinking = thinking[i] if thinking is not None else None
        content = (build_assistant_content(row_thinking, str(outputs[i]))
                   if row_thinking is not None else str(outputs[i]))
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": str(inputs[i])},
            {"role": "assistant", "content": content},
        ]
        texts.append(tokenizer.apply_chat_template(messages, tokenize=False))
    return {"text": texts, "split": batch["split"]}


def _assert_clean_thinking(dataset):
    """Boundary check: stored Thinking must be raw (no baked-in think tags).
    Datasets without a Thinking column (MIMIC ablation) are skipped."""
    if "Thinking" not in dataset.column_names:
        return
    sample = dataset.select(range(min(2000, len(dataset))))
    for t in sample["Thinking"]:
        if t and ("</think>" in t or "<think>" in t):
            raise ValueError(
                "Stored 'Thinking' contains <think>/</think> tags. Rebuild the "
                "instruction dataset with the fixed build_instructions (raw "
                "thinking only) before training."
            )


def load_sft_data_splits(args, tokenizer):
    """Return (train_text, eval_text_dict).

    - train_text: Dataset with a single ``text`` column for SFTTrainer.
    - eval_text_dict: {"dev": ds, "val": ds} for per-split eval loss.

    Downstream task metrics are computed separately by the post-hoc
    checkpoint-eval orchestrator (scripts/eval_checkpoints.py).
    """
    dataset = load_from_disk(args.dataset_path)
    _assert_clean_thinking(dataset)

    formatted = dataset.map(
        to_text,
        batched=True,
        num_proc=4,
        remove_columns=[c for c in dataset.column_names if c != "split"],
        fn_kwargs={"tokenizer": tokenizer},
        keep_in_memory=False,
    )

    train_data = formatted.filter(lambda x: x["split"] == "train", num_proc=32)
    train_data = train_data.remove_columns(["split"])

    # Subsample each eval split for cheap per-epoch eval LOSS (downstream task
    # metrics come from the post-hoc pipeline eval). eval_samples <= 0 == full.
    # Empty splits are dropped (e.g. MIMIC has no 'val') — an empty eval dataset
    # crashes SFTTrainer's _prepare_dataset.
    n = getattr(args, "eval_samples", -1)
    counts = []
    eval_text = {}
    for split in EVAL_SPLITS:
        ds = formatted.filter(lambda x, s=split: x["split"] == s, num_proc=32)
        ds = ds.remove_columns(["split"])
        if n and n > 0:
            ds = ds.select(range(min(n, len(ds))))
        counts.append(f"{split}: {len(ds):,}")
        if len(ds) > 0:
            eval_text[split] = ds

    print(f"Splits — train: {len(train_data):,} | " + " | ".join(counts)
          + (f" (eval capped at {n}/split)" if n and n > 0 else ""))
    return train_data, eval_text
