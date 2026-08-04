import unsloth
from unsloth import FastLanguageModel, is_bfloat16_supported


import os
import torch

from accelerate import Accelerator
from accelerate.utils import set_seed
from trl import SFTTrainer, SFTConfig
import wandb

from unsloth.chat_templates import train_on_responses_only

from src.fine_tuning.data_formatting import (
    load_sft_data_splits, set_qwen3_chat_template, INSTRUCTION_PART, RESPONSE_PART,
)
from src.fine_tuning.manifest_callback import CheckpointManifestCallback


def setup_hf_cache(args):
    """Sets up the Hugging Face cache directory."""
    hf_cache_dir = args.hf_cache_dir
    os.environ["HF_HOME"] = hf_cache_dir
    for subdir in ["transformers", "datasets", "hub"]:
        os.makedirs(os.path.join(hf_cache_dir, subdir), exist_ok=True)


def assert_assistant_masking(trainer):
    """Entry-point validation: confirm the loss mask covers part (not all) of the
    sequence, i.e. assistant-only masking is actually active."""
    batch = next(iter(trainer.get_train_dataloader()))
    labels = batch["labels"]
    frac = (labels != -100).float().mean().item()
    print(f"Assistant-only loss check: {frac:.3f} of tokens are supervised.")
    if not (0.0 < frac < 1.0):
        raise RuntimeError(
            f"Loss mask is degenerate (frac={frac:.3f}). train_on_responses_only "
            "did not mask as expected (check the ChatML markers)."
        )


def load_model(args, accelerator):
    """Load model + tokenizer. use_lora selects LoRA vs full-parameter."""
    full = not args.use_lora
    # Two full-FT paths :
    # - STANDARD (default): load normally + requires_grad on all params. Works
    # multi-GPU (DDP-safe), fast, stable for 8B+. This is the proven path.
    # - UNSLOTH (unsloth_full_finetuning=True): Unsloth's full_finetuning. Use only
    # for models where STANDARD breaks (0.6B NaNs + crawls — likely its tied
    # embeddings). It enables gradient offload that deadlocks DDP -> SINGLE-GPU.
    use_uft = full and args.unsloth_full_finetuning
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model_name,
        max_seq_length=args.max_seq_length,
        dtype=None,                 # auto bf16 on B200/H200
        load_in_4bit=args.load_in_4bit,
        full_finetuning=use_uft,
        device_map={"": accelerator.local_process_index},
    )
    tokenizer.padding_side = "right"
    set_qwen3_chat_template(tokenizer)
    model.config.use_cache = False  # required for gradient checkpointing

    # Unsloth's "unsloth" checkpointing uses reentrant backward, which DDP cannot
    # support ("marked ready twice"). For multi-GPU LoRA use standard non-reentrant
    # checkpointing (SFTConfig.gradient_checkpointing=True, use_reentrant=False).
    gc = "unsloth" if accelerator.num_processes == 1 else False

    if not full:
        print(f"LoRA enabled (r={args.lora_r}). gradient_checkpointing={gc!r}")
        model = FastLanguageModel.get_peft_model(
            model,
            r=args.lora_r,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj"],
            lora_alpha=args.lora_r * 2,
            lora_dropout=args.lora_dropout,
            bias="none",
            use_gradient_checkpointing=gc,
            random_state=args.seed,
        )
    else:
        if not use_uft:
            for p in model.parameters():
                p.requires_grad_(True)
        elif accelerator.num_processes > 1:
            print("WARNING: unsloth_full_finetuning on >1 GPU deadlocks DDP. Use 1 GPU.")
        n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
        path = "unsloth" if use_uft else "standard"
        print(f"Full-parameter fine-tuning ({path}): {n_train / 1e9:.2f}B trainable.")
        assert n_train > 0, "Full FT but no trainable params — model is frozen."
    return model, tokenizer


def train(args):
    set_seed(args.seed)
    setup_hf_cache(args)

    accelerator = Accelerator()
    is_main = accelerator.is_main_process
    mode = "lora" if args.use_lora else "full"
    print(f"Rank {os.environ.get('RANK')} | Local {accelerator.local_process_index} | "
          f"World {os.environ.get('WORLD_SIZE')} | mode={mode}")

    if is_main:
        wandb.init(project=args.wandb_project, entity=args.wandb_entity, name=args.job_name)
        wandb.config.update(vars(args), allow_val_change=True)
    else:
        os.environ["WANDB_MODE"] = "disabled"

    print(vars(args))
    torch.set_float32_matmul_precision("high")

    model, tokenizer = load_model(args, accelerator)

    print(f"Loading dataset from {args.dataset_path}...")
    with accelerator.main_process_first():
        train_data, eval_text = load_sft_data_splits(args, tokenizer)

    FastLanguageModel.for_training(model)

    # Multi-GPU + assistant-only loss don't mix on this Unsloth stack: the
    # unpacked train_on_responses_only path trips DDP's gradient-checkpointing
    # reducer. Use packing + full-sequence loss for multi-GPU, or run 1 GPU.
    if args.assistant_only_loss and accelerator.num_processes > 1:
        print("WARNING: assistant_only_loss on >1 GPU is unsupported here (DDP + "
              "Unsloth checkpointing). Expect a reducer error — use 1 GPU, or set "
              "assistant_only_loss=False with packing=True for multi-GPU.")

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

        bf16=is_bfloat16_supported(),

        packing=args.packing,
        dataset_text_field="text",
        max_length=args.max_seq_length,

        dataloader_num_workers=4,
        optim="adamw_torch_fused",
        # Must be True to ACTIVATE checkpointing: it calls model.gradient_
        # checkpointing_enable(), which for LoRA is Unsloth's patched ("unsloth")
        # method set in get_peft_model. Without this the 8192-token activations
        # don't fit.
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        ddp_find_unused_parameters=False,

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

    # Assistant-only loss: mask everything except the assistant response tokens.
    if args.assistant_only_loss:
        trainer = train_on_responses_only(
            trainer,
            instruction_part=INSTRUCTION_PART,
            response_part=RESPONSE_PART,
        )

    # Record epoch -> checkpoint mapping for the post-hoc downstream eval sweep.
    if is_main:
        trainer.add_callback(CheckpointManifestCallback(
            output_dir=args.output_dir,
            job_name=args.job_name,
            model_name=args.model_name,
            finetuning_mode=mode,
            base_model=args.model_name,
            lora_r=args.lora_r,
        ))

    if args.assistant_only_loss:
        assert_assistant_masking(trainer)

    gpu = torch.cuda.get_device_properties(0)
    print(f"GPU: {gpu.name}, {gpu.total_memory / 1024 ** 3:.1f} GB")

    if args.resume_from_checkpoint:
        print(f"Resuming from the latest checkpoint in {args.output_dir} "
              "(optimizer/scheduler/step restored, not just weights).")
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint or None)

    print(f"Saving model to {args.output_dir}")
    if accelerator.is_main_process:
        model.save_pretrained(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)
