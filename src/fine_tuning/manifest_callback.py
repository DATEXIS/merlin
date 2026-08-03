"""Writes a checkpoint manifest during SFT so the post-hoc downstream-eval
orchestrator knows which epoch maps to which checkpoint dir.

No torch/unsloth deps — just records (epoch, global_step, path) on every save.
"""

import json
import os

from transformers import TrainerCallback


class CheckpointManifestCallback(TrainerCallback):
    def __init__(self, output_dir: str, job_name: str, model_name: str,
                 finetuning_mode: str, base_model: str, lora_r: int):
        self.path = os.path.join(output_dir, "checkpoints.json")
        # On a resumed run (resume_from_checkpoint: True), a fresh callback
        # instance would otherwise start "checkpoints": [] and overwrite the
        # file on the first on_save, silently dropping every checkpoint
        # recorded before the kill/resume. Load prior history if present.
        prior_checkpoints = []
        if os.path.exists(self.path):
            with open(self.path) as f:
                prior_checkpoints = json.load(f).get("checkpoints", [])
        self.meta = {
            "job_name": job_name,
            "model_name": model_name,
            "base_model": base_model,
            "finetuning_mode": finetuning_mode,   # "lora" or "full"
            "lora_r": lora_r,                      # -> max_lora_rank when serving
            "output_dir": output_dir,
            "checkpoints": prior_checkpoints,
        }

    def on_save(self, args, state, control, **kwargs):
        if not state.is_world_process_zero:
            return
        ckpt_dir = os.path.join(args.output_dir, f"checkpoint-{state.global_step}")
        self.meta["checkpoints"].append({
            "epoch": round(state.epoch, 4),
            "global_step": state.global_step,
            "path": ckpt_dir,
        })
        with open(self.path, "w") as f:
            json.dump(self.meta, f, indent=2)
        print(f"[manifest] epoch {state.epoch:.2f} -> {ckpt_dir}")
