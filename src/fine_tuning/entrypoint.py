import argparse


def str2bool(v: str) -> bool:
    return str(v).lower() == "true"


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    # Path & Project Args
    p.add_argument("--model_name", required=True)
    p.add_argument("--dataset_path", required=True)
    p.add_argument("--dataset_models", default="")  # currently unused (no model filter)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--job_name", default="finetune-run")  # also used as the wandb run name

    # Training Hyperparams
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--grad_accum", type=int, default=4)
    p.add_argument("--learning_rate", type=float, default=2e-4)
    p.add_argument("--warmup_steps", type=int, default=20)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--max_seq_length", type=int, default=2048)
    p.add_argument("--logging_steps", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--eval_batch_size", type=int, default=8)
    p.add_argument("--eval_samples", type=int, default=-1)  # cap per-split eval; -1 = full
    # If True, auto-detects and resumes from the latest checkpoint-N in output_dir
    # (HF Trainer's get_last_checkpoint) -- restores optimizer/scheduler/step, not
    # just weights. Use after a job was killed/preempted mid-run and output_dir
    # still has its checkpoint-N dirs on the PVC (same job_name -> same output_dir).
    p.add_argument("--resume_from_checkpoint", type=str2bool, default=False)

    # Fine-tuning mode: use_lora is the single switch (True = LoRA, False = full-param).
    p.add_argument("--use_lora", type=str2bool, default=True)
    # Full-FT path: standard requires_grad (default, multi-GPU) vs Unsloth
    # full_finetuning (single-GPU; only needed where standard NaNs, e.g. 0.6B).
    p.add_argument("--unsloth_full_finetuning", type=str2bool, default=False)
    # Third full-FT path, for models whose optimizer state doesn't fit replicated
    # on one GPU (32B+, note: Unsloth is DDP-only... needs
    # FSDP/DeepSpeed ZeRO-3, a non-Unsloth stack"). Routes to training_fsdp.py
    # instead of training.py below -- ignores unsloth_full_finetuning/lora_*.
    # Only meaningful with use_lora=False and gpu_count > 1 launched via
    # accelerate's --use_fsdp (see src/kubernetes/fine_tuning_template.py).
    p.add_argument("--fsdp_full_finetuning", type=str2bool, default=False)
    p.add_argument("--load_in_4bit", type=str2bool, default=False)  # True for QLoRA on tight VRAM
    p.add_argument("--packing", type=str2bool, default=True)        # ignored if assistant_only_loss
    p.add_argument("--assistant_only_loss", type=str2bool, default=True)

    # LoRA
    p.add_argument("--lora_r", type=int, default=16)
    p.add_argument("--lora_dropout", type=float, default=0.05)

    # WandB
    p.add_argument("--wandb_project", default="MERLIN-DDX-SFT")
    p.add_argument("--wandb_entity", default="anon-entity")

    # Cache folders
    p.add_argument("--hf_cache_dir", default="/tmp/hf_cache")
    p.add_argument("--wandb_cache_dir", default="/tmp/wandb_cache")

    args = p.parse_args()

    # Lazy import: training.py does `import unsloth` at module scope, which
    # monkey-patches transformers process-wide on import. The FSDP full-FT
    # path must never trigger that (Unsloth's patches are what make DDP-only
    # training break under FSDP ), so only import training.py
    # when we're actually taking the Unsloth-based path.
    if args.fsdp_full_finetuning:
        from src.fine_tuning.training_fsdp import train
    else:
        from src.fine_tuning.training import train
    train(args)
