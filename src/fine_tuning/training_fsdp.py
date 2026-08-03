"""Full-parameter fine-tuning via Accelerate FSDP, for models whose optimizer
state doesn't fit replicated on a single GPU (32B+ full-FT, per merlin.md's
gotchas: "Unsloth is DDP-only ... full-parameter 32B cannot run on Unsloth --
it needs FSDP/DeepSpeed ZeRO-3, a non-Unsloth stack").

Deliberately does NOT import unsloth (see entrypoint.py's lazy-import comment):
Unsloth monkey-patches transformers process-wide on import, which is exactly
what makes its DDP-only multi-GPU path incompatible with FSDP sharding/hooks.
This module loads a plain transformers model instead, and lets Accelerate's
FSDP plugin (configured via `accelerate launch --use_fsdp ...` CLI flags in
src/kubernetes/fine_tuning_template.py) do the sharding.

Reuses data_formatting.py and manifest_callback.py from the Unsloth path
unchanged -- neither has an unsloth dependency.

UNTESTED end-to-end (no cluster access to run it). Sanity-checked against
transformers/accelerate/TRL's documented FSDP integration and against this
repo's proven training.py, but the first real run should be watched closely
for the same class of issues training.py's gotchas section documents for the
DDP path (NaN loss, OOM, checkpointing-callback interaction).
"""

import os
import torch

from accelerate import Accelerator
from accelerate.utils import set_seed
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTTrainer, SFTConfig
import wandb

from src.fine_tuning.data_formatting import load_sft_data_splits, set_qwen3_chat_template
from src.fine_tuning.manifest_callback import CheckpointManifestCallback


def setup_hf_cache(args):
    """Sets up the Hugging Face cache directory. Duplicated from training.py
    (not imported from there) to keep this module unsloth-free."""
    hf_cache_dir = args.hf_cache_dir
    os.environ["HF_HOME"] = hf_cache_dir
    for subdir in ["transformers", "datasets", "hub"]:
        os.makedirs(os.path.join(hf_cache_dir, subdir), exist_ok=True)


def load_model(args):
    """Load a plain (non-Unsloth) HF model + tokenizer for full-parameter
    fine-tuning under FSDP. No device_map/`.to(device)` here -- Accelerate's
    FSDP plugin (fsdp_cpu_ram_efficient_loading + fsdp_sync_module_states,
    set at accelerate-launch time) materializes real weights on rank 0 only
    and broadcasts shards to the other ranks when the trainer calls
    accelerator.prepare(); manually placing the model first would defeat that
    and try to materialize the full model on every rank's GPU/CPU."""
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenizer.padding_side = "right"
    set_qwen3_chat_template(tokenizer)

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16,
        # packing=True (padding-free, samples flattened into one sequence) needs a
        # flash-attention variant to (a) mask correctly so packed samples don't
        # attend to each other, and (b) run at flash-attn speed instead of a dense/
        # unoptimized fallback. SDPA logged both a correctness warning and a ~10x
        # slowdown (516 s per micro-step) on the first real run -- switched to FA2,
        # which the fine_tuning image should already carry (Unsloth depends on it
        # and it's used successfully by the existing LoRA/Unsloth full-FT paths).
        attn_implementation="flash_attention_2",
    )
    model.config.use_cache = False  # required for gradient checkpointing

    for p in model.parameters():
        p.requires_grad_(True)
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Full-parameter fine-tuning (FSDP, non-Unsloth): {n_train / 1e9:.2f}B trainable.")
    assert n_train > 0, "Full FT but no trainable params -- model is frozen."
    return model, tokenizer


def train(args):
    assert not args.use_lora, "training_fsdp.py is full-parameter only -- use training.py for LoRA."
    assert args.packing, "FSDP path requires packing=True (full-sequence loss); matches the multi-GPU combo in run_plan.yaml."
    assert not args.assistant_only_loss, "assistant_only_loss is unsupported here (single-GPU-only trick, see merlin.md); use packing=True instead."

    set_seed(args.seed)
    setup_hf_cache(args)

    accelerator = Accelerator()
    is_main = accelerator.is_main_process
    print(f"Rank {os.environ.get('RANK')} | Local {accelerator.local_process_index} | "
          f"World {os.environ.get('WORLD_SIZE')} | mode=full (fsdp)")

    if is_main:
        wandb.init(project=args.wandb_project, entity=args.wandb_entity, name=args.job_name)
        wandb.config.update(vars(args), allow_val_change=True)
    else:
        os.environ["WANDB_MODE"] = "disabled"

    print(vars(args))
    torch.set_float32_matmul_precision("high")

    model, tokenizer = load_model(args)

    print(f"Loading dataset from {args.dataset_path}...")
    with accelerator.main_process_first():
        train_data, eval_text = load_sft_data_splits(args, tokenizer)

    print(f"Starting training for {args.epochs} epochs.")
    training_args = SFTConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        weight_decay=args.weight_decay,
        logging_steps=args.logging_steps,

        bf16=True,

        packing=args.packing,
        dataset_text_field="text",
        max_length=args.max_seq_length,

        dataloader_num_workers=4,
        # adamw_torch (not _fused): training.py uses the fused variant, but its
        # FSDP1 compatibility is version-dependent -- unverified on this stack.
        # Safer default for a first run; revisit if optimizer-step time dominates.
        optim="adamw_torch",
        # Must be True to ACTIVATE checkpointing (calls model.gradient_
        # checkpointing_enable()) -- without it the 8192-token activations OOM
        # even at batch_size=1 (see training.py's identical comment).
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},

        # Evaluate each non-empty eval split after every epoch.
        eval_strategy="epoch" if eval_text else "no",
        per_device_eval_batch_size=args.eval_batch_size,

        report_to=["wandb"],
        save_strategy="epoch",
    )

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=eval_text or None,   # dict -> eval_dev_*, eval_val_* in wandb
    )

    # Record epoch -> checkpoint mapping for the post-hoc downstream eval sweep.
    if is_main:
        trainer.add_callback(CheckpointManifestCallback(
            output_dir=args.output_dir,
            job_name=args.job_name,
            model_name=args.model_name,
            finetuning_mode="full",
            base_model=args.model_name,
            lora_r=-1,  # not applicable outside LoRA
        ))

    gpu = torch.cuda.get_device_properties(0)
    print(f"GPU: {gpu.name}, {gpu.total_memory / 1024 ** 3:.1f} GB")

    if args.resume_from_checkpoint:
        print(f"Resuming from the latest checkpoint in {args.output_dir} "
              "(optimizer/scheduler/step restored, not just weights).")
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint or None)

    print(f"Saving model to {args.output_dir}")
    # Unlike training.py's model.save_pretrained(...) gated on is_main: under
    # FSDP each rank only holds a shard, so gathering the full state dict is a
    # COLLECTIVE op. trainer.save_model() must be called on every rank (it
    # gates the actual file write internally) -- gating this call itself on
    # is_main would make the other ranks skip the collective and hang forever.
    trainer.save_model(args.output_dir)
    if is_main:
        tokenizer.save_pretrained(args.output_dir)
