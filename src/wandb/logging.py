import json
import logging
from datetime import datetime

import numpy as np
import pandas as pd
import wandb

from src.exp_args import ExpArgs
from src.pipeline.verifier_args import VerifierArgs

LABELS_STR = {
    1: 'disease_vector',
    2: 'disease',
    3: 'disease',
    4: 'ICD_CODES',
}


def log_init(exp_args: ExpArgs, v_args: VerifierArgs):
    print(json.dumps(exp_args.config_str, indent=4))
    wandb.log({
        'concurrency': v_args.concurrency,
        'num_choices': v_args.num_choices,
        'num_samples': exp_args.num_samples,
    })


def log_budget_step_start(v_args: VerifierArgs):
    budget_step = v_args.budget - v_args.current_budget + 1
    logging.info(f'V{v_args.current_verifier} start {budget_step}/{v_args.budget} with '
                 f'Temperature: {round(v_args.temperature, 1)}, '
                 f'Max Tokens: {v_args.max_tokens}')


def log_budget_step_metrics(v_args: VerifierArgs, patients: pd.DataFrame):
    if v_args.budget <= 1:
        return
    budget_step = v_args.budget - v_args.current_budget + 1
    mean_score = round(np.mean(patients[f'v{v_args.current_verifier}_score'].fillna(0)), 3)
    mean_valid_score = round(np.mean(patients[f'v{v_args.current_verifier}_score']), 3)
    wandb.log({
        f'B_step': budget_step,
        f'v{v_args.current_verifier}_score': mean_score,
        f'V{v_args.current_verifier}_valid_score': mean_valid_score,
    })
    logging.info(f'V{v_args.current_verifier} end {budget_step}/{v_args.budget} with '
                 f'Valid Jsons Score: {mean_valid_score}, overall Score: {mean_score}')


def log_verifier_metrics(exp_args: ExpArgs, v_args: VerifierArgs, patients: pd.DataFrame):
    v_step = v_args.current_verifier
    mean_score = round(np.mean(patients[f'v{v_step}_score'].fillna(0)), 3)
    mean_valid_score = round(np.mean(patients[f'v{v_step}_score']), 3)
    minutes = round((datetime.now() - exp_args.init_time).seconds / 60, 2)
    wandb.log({
        f'V{v_step}_score': mean_score,
        f'V{v_step}_valid_score': mean_valid_score,
        'v_step': v_step,
        'score': mean_score,
        'used_budget': v_args.generated_prompts,
        'minutes': minutes,
        'max_tokens': v_args.max_tokens,
        'temperature': v_args.temperature,
        'budget': v_args.budget,
    })
    logging.info(f'V{v_step} Top Scores:\n{patients[f"v{v_step}_score"]}')


def log_eval_metrics(patients: pd.DataFrame, sft_epoch: int = None):
    """Compute downstream task metrics and log them to the current WandB run.

    sft_epoch (if given) is logged alongside the metrics in the same call so it
    lands on the same step, letting you pick 'sft_epoch' as a custom x-axis
    for line/scatter charts across checkpoint-eval runs.
    """
    from src.eval.eval_experiments import evaluate_experiment
    try:
        metrics = evaluate_experiment(patients)
        log_dict = {f'eval/{k}': v for k, v in metrics.items()}
        if sft_epoch is not None:
            log_dict['sft_epoch'] = sft_epoch
        wandb.log(log_dict)
        logging.info(f'Eval metrics: {metrics}')
    except Exception as e:
        logging.error(f'Failed to compute eval metrics: {e}')


def log_dataframe(patients: pd.DataFrame, v_step: int):
    result_cols = [f'v{v_step}_prompt', f'v{v_step}_score', f'all_v{v_step}_scores',
                   f'v{v_step}_json', f'v{v_step}_text', f'v{v_step}_preds']
    v_columns = {
        0: ['Chief Complaint', 'admission_note', 'labs', 'Discharge Diagnosis'],
        1: [LABELS_STR[1]] + result_cols,
        2: [LABELS_STR[2]] + result_cols,
        3: [LABELS_STR[3]] + result_cols,
        4: [LABELS_STR[4]] + result_cols,
    }
    log_df = patients[v_columns[v_step]].map(str).reset_index().head(30) # Limit to 30 rows for logging
    wandb.log({f'patients_v{v_step}': wandb.Table(dataframe=log_df)})
