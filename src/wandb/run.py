import json
import logging
import os
import re

import wandb
from src.exp_args import ExpArgs


def init_wandb(run_name: str, eval_mode=False, tags=None, qa=False):
    project_name = 'merlin-eval-1.4' if eval_mode else 'merlin-generation'  # merlin-generation
    if qa:
        project_name = 'merlin_qa'
    key = os.getenv("WANDB_API_KEY") or json.load(open("config.json"))["WANDB_API_KEY"]

    wandb.login(key=key)
    return wandb.init(
        project=project_name,
        entity='anon-entity',
        name=run_name,
        tags=tags,
    )


def extract_sft_epoch(job_name: str):
    """Pull the epoch out of an eval job_name. Handles both naming schemes
    produced by eval_job_name() in scripts/eval_checkpoints.py:
      - default: eval-<run>-ep<N>-s<step>, e.g.
        'eval-ft-qwen3-8b-full-merlin-v1-ep1-s372' -> 1
      - short (a manifest entry sets `short_name`, e.g. v1.4's
        eval_config.yaml): <short_name>-e<N>, e.g.
        '8b-full-mimic-icd2-e1' -> 1
    Returns None if neither pattern matches (not a checkpoint-eval job_name)."""
    m = re.search(r'-ep?(\d+)(?:-s\d+)?$', job_name)
    return int(m.group(1)) if m else None


def get_hardware_tag(config: dict) -> str:
    num_gpus = config['Hardware'].get('gpu_count', '')
    gpu_type = config['Hardware'].get('server_gpu', '')
    # server_gpu is a plain string ("b200") when Hardware.server_gpu is a single
    # type, but can be a list (["b200","h200","h100"]) when the config uses the
    # nodeAffinity "any of these" eligible-set form (see server_template.py) --
    # str + list raised an uncaught TypeError here and crashed the client Job
    # before it did any real work (e.g. eval-8b-full-thrfull-icd2-e1/-mimic-icd2-e4).
    if isinstance(gpu_type, list):
        gpu_type = '-or-'.join(gpu_type)
    return str(num_gpus) + "x" + gpu_type


def update_wandb_name_tags(run, exp_args: ExpArgs):
    """Update the name of an existing WandB run."""
    cfg = exp_args.config_str
    run.name = exp_args.short_llm_name
    hardware_tag = get_hardware_tag(cfg)
    chief_complaint = 'cc_' + cfg['Client_Job']['chief_complaint']
    run.tags += (exp_args.short_llm_name, hardware_tag, chief_complaint)

    # Checkpoint-eval runs serve a model called 'checkpoint-<step>', so the bare
    # name is ambiguous across runs/epochs (e.g. 'checkpoint-372-full-no-labs').
    # The eval job_name encodes model + epoch (+ step, for the default scheme),
    # so use it as the run name — works for full and LoRA checkpoints, and for
    # both the default 'eval-<run>-ep<N>-s<step>' naming AND the short
    # '<short_name>-e<N>' naming (see eval_job_name() in
    # scripts/eval_checkpoints.py / eval_config.yaml) --
    # anything without the 'eval-' prefix, if present, is used as-is.
    job_name = cfg.get('job_name', '')
    if exp_args.eval_mode and job_name:
        run.name = job_name[len('eval-'):] if job_name.startswith('eval-') else job_name
        sft_epoch = extract_sft_epoch(job_name)
        if sft_epoch is not None:
            exp_args.sft_epoch = sft_epoch
            run.config.update({'sft_epoch': sft_epoch})  # lets you group/filter runs by it
            wandb.log({'sft_epoch': sft_epoch})          # also a logged metric, usable as
                                                          # custom x-axis on line/scatter charts
    elif cfg['Model'].get('full_finetuning'):
        run.name += '-full'
    elif cfg['Model'].get('lora'):
        mode = 'mimic' if 'mimic' in cfg['Model']['lora_modules'] else 'merlin'
        run.name += f'-{mode}-{job_name[-2:]}'

    if cfg['Model'].get('full_finetuning'):
        run.tags += ('full', )
    if cfg['Model'].get('lora'):
        run.tags += ('lora', )
    if exp_args.config_str['Model'].get('thinking'):
        run.name += '-think'
    if not exp_args.config_str['Client_Job'].get('merlin_mode'):
        run.name += '-no-merlin'
    elif exp_args.guided_decoding:
        run.name += '-JSON'
        run.tags += ('gui_dec', )
    if exp_args.seed != 42:
        run.name += f'-seed{exp_args.seed}'



    exp_args.run_name = run.name

    if exp_args.eval_mode and exp_args.merlin_mode:
        run.tags += ('merlin_eval', )
    elif exp_args.eval_mode and not exp_args.merlin_mode:
        run.tags += ('mimic_eval', )
    else:
        chief_complaint = 'cc_' + exp_args.config_str['Client_Job']['chief_complaint']
        run.tags += (f'data_gen', chief_complaint)

    logging.info(f'Updated WandB run name to: {run.name}')
