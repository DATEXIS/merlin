import ast
import logging

import pandas as pd
import wandb

from src.pipeline.instruction_builder import create_merlin_instructions
from src.exp_args import ExpArgs
from src.pipeline.verifier_args import VerifierArgs
from src.wandb.run import init_wandb


def load_schemes_from_wandb(wandb_run, exp_args: ExpArgs) -> dict:
    # ToDo: Implement
    file_name = f'dataset_{exp_args.num_samples}_{exp_args.short_llm_name.replace("-", "_")}'
    artifact = wandb_run.use_artifact(f"{file_name}:latest", type="dataset")


def load_merlin_instructions(wandb_run, model: str) -> pd.DataFrame:
    """ Load merlin instructions from WandB artifact. """
    model = model.replace("-", "_")
    artifact_ref = f"instructions_3055_{model}"
    artifact = wandb_run.use_artifact(artifact_ref + ":latest", type="instructions")
    artifact_dir = artifact.download("data/results")
    file_path = f"{artifact_dir}/{artifact_ref}.pq"
    logging.info(f'Instructions locally stored at: {file_path}')

    return pd.read_parquet(file_path, engine='fastparquet')


def load_cross_validation_patients(wandb_run, experiment_name: str | list) -> list[pd.DataFrame]:
    """ Load up to 3 results for an experiment from WandB artifacts for evaluation. """
    dfs = []

    if isinstance(experiment_name, str):
        experiment_name = [experiment_name]
    print(f'Download {experiment_name}')
    for i in reversed(range(20)):  # try v19 → v0
        if len(dfs) == 3:
            break
        for exp_name in experiment_name:
            if len(dfs) == 3:
                break
            artifact_ref = f"{exp_name}:v{i}"
            try:
                artifact = wandb_run.use_artifact(artifact_ref, type="eval_results")
            except (ValueError, wandb.CommError):
                logging.debug(f"{artifact_ref} not found, skipping")
                continue

            artifact_dir = artifact.download()
            file_path = f"{artifact_dir}/{exp_name}.pq"
            df = pd.read_parquet(file_path, engine="fastparquet")
            preds_cols = ['v2_preds', 'v4_preds', 'v2_json', 'v4_json']
            df[preds_cols] = df[preds_cols].map(ast.literal_eval)

            if "hadm_id" in df.columns:
                df = df.set_index("hadm_id")

            dfs.append(df)

    if not dfs:
        raise RuntimeError(f"No artifacts found for {experiment_name}")

    return list(reversed(dfs))


def load_patients_from_wandb(wandb_run, exp_args: ExpArgs, v_step=1) -> pd.DataFrame:
    """ Load patients from WandB artifact. Files are patients_10 / patients_63 / patients_1411
    To start with v_step > 1 a run must have been completed with the llm. """
    file_name = f'patients_{exp_args.num_samples}'
    file_name = exp_args.file_name

    if exp_args.eval_mode:
        logging.info('Loading evaluation dataset from WandB.')
        file_name = 'test_dataset'

    if v_step > 1:
        llm_name = exp_args.short_llm_name.replace("-", "_")
        file_name = f'dataset_{exp_args.num_samples}_{llm_name}'
        artifact = wandb_run.use_artifact(f"{file_name}:latest", type="dataset")
    else:
        artifact = wandb_run.use_artifact(f"{file_name}:latest", type="dataset")

    artifact_dir = artifact.download()
    file_path = f"{artifact_dir}/{file_name}.pq"
    logging.info(f'Dataset locally stored at: {file_path}')

    df = pd.read_parquet(file_path, engine='fastparquet')

    if exp_args.splits:
        assert 'split' in df.columns, f"No 'split' column in {file_path}; cannot filter to {exp_args.splits}"
        df = df[df['split'].isin(exp_args.splits)]
        logging.info(f'Filtered to splits {exp_args.splits}: {len(df)} notes.')

    return df.head(exp_args.num_samples)


def store_checkpoint(exp_args: ExpArgs, v_args: VerifierArgs, patients: pd.DataFrame,
                     work_df: pd.DataFrame):

    if exp_args.eval_mode:
        return

    path = f'/checkpoints/{v_args.chief_complaint}_{exp_args.short_llm_name}'
    work_path = f'{path}_work_df.pq'
    patients_path = f'{path}_patients_df.pq'

    work_df.to_parquet(work_path)
    patients.to_parquet(patients_path)
    v_args.store_checkpoint(path)


def load_checkpoint(exp_args: ExpArgs, v_args: VerifierArgs):
    path = f'/checkpoints/{v_args.chief_complaint}_{exp_args.short_llm_name}' or f'/checkpoints/default_run'
    work_path = f'{path}_work_df.pq'
    patients_path = f'{path}_patients_df.pq'
    patients = pd.read_parquet(patients_path)
    try:
        work_df = pd.read_parquet(work_path)
    except FileNotFoundError:
        work_df = pd.read_parquet(patients_path)

    v_args.load_from_checkpoint(path)

    # If checkpoint was stored at the end of a verifier step, move to next step
    if len(work_df) == 0 or v_args.current_budget == 0:
        v_args.set_next_verifier(exp_args.eval_mode)
        work_df = patients.copy()

    return patients, work_df


def load_patients(wandb_run, exp_args: ExpArgs, v_args: VerifierArgs):
    """ Try to load patients from WandB, otherwise load from local file. """
    if exp_args.load_from_checkpoint:
        return load_checkpoint(exp_args, v_args)
    else:
        try:
            patient_df = load_patients_from_wandb(wandb_run, exp_args, v_args.current_verifier)
        except Exception as e:
            if v_args.start_verifier == 1:
                logging.warning(f'Could not load patients from WandB, loading from local file: {e}')
                path = f'data/reasoning/abdominal_pain/patients_{exp_args.num_samples}.pq'
                patient_df = pd.read_parquet(path, engine='fastparquet')
            else:
                raise FileNotFoundError("Could not load patients from WandB for verifier step > 1.")

    labs_na_string = 'No laboratory results available.'
    patient_df['labs'] = patient_df.get('labs', pd.Series()).fillna(labs_na_string)

    return patient_df, patient_df.copy()


def upload_df_to_wandb(wandb_run, df: pd.DataFrame, artifact_name: str, data_type: str):
    df.to_parquet(f'{artifact_name}.pq')
    artifact = wandb.Artifact(artifact_name, data_type)
    artifact.add_file(f'{artifact_name}.pq')
    wandb_run.log_artifact(artifact)
    logging.info(f'Uploaded {data_type} to WandB: {data_type}/{artifact_name}')


def upload_merlin_results(wandb_run, exp_args: ExpArgs, patients: pd.DataFrame):
    if exp_args.eval_mode:
        run_name = 'eval_results_' + exp_args.run_name
        upload_df_to_wandb(wandb_run, patients, run_name, 'eval_results')
    else:
        upload_df_to_wandb(wandb_run, patients, exp_args.run_name, 'dataset')
        instructions = create_merlin_instructions(patients)
        artifact_name = f'{len(patients)}_{exp_args.short_llm_name.replace("-", "_")}'
        upload_df_to_wandb(wandb_run, instructions, artifact_name, 'instructions')


def download_all_instructions(models: list[str]) -> pd.DataFrame:
    """Download instructions for all models from wandb"""
    wandb_run = init_wandb('Download Instructions', eval_mode=False)
    all_instructions = pd.DataFrame()

    # Workaround to add the splits
    path_for_split = 'data/reasoning/abdominal_pain/patients_3055.pq'
    splits = pd.read_parquet(path_for_split)['split']
    splits = [s for s in splits.values for _ in range(4)]  # each patient has 4 verifiers

    for model in models:
        instruction_df = load_merlin_instructions(wandb_run, model)
        instruction_df['model'] = model
        instruction_df['split'] = splits
        all_instructions = pd.concat([all_instructions, instruction_df], axis=0)

    wandb_run.finish()
    return all_instructions
