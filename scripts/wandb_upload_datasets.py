import wandb

from src.wandb.run import init_wandb

wandb_run = init_wandb('Upload Patients Dataset', eval_mode=True)
#
# artifact = wandb.Artifact('patients_3055', type='dataset')
# artifact.add_file('data/reasoning/abdominal_pain/patients_3055.pq')
# wandb_run.log_artifact(artifact)
#
# chief_complaints = ['abdominal_pain', 'back_pain', 'chest_pain', 'cough', 'diarrhea', 'dyspnea',
#                     'headache', 'dyspnea_part1', 'dyspnea_part2']
#
# for chief_complaint in ['headache']:
#     artifact = wandb.Artifact(f'patients_{chief_complaint}', type='dataset')
#     artifact.add_file(f'data/preprocessed_mimic/patients_{chief_complaint}.pq')
#     wandb_run.log_artifact(artifact)
#
# artifact = wandb.Artifact('test_dataset', type='dataset')
# artifact.add_file('data/reasoning/abdominal_pain/test_dataset.pq')
# wandb_run.log_artifact(artifact)
#
# artifact = wandb.Artifact('patients_ood_700', type='dataset')
# artifact.add_file('data/reasoning/patients_ood_700.pq')
# wandb_run.log_artifact(artifact)
#
# wandb_run.finish()

# wandb_run = init_wandb('Upload Patients Dataset', eval_mode=True)
#
artifact = wandb.Artifact('test_dataset', type='dataset')
artifact.add_file('data/results/test_dataset.pq')
wandb_run.log_artifact(artifact)
#
# wandb_run.finish()

# wandb_run = init_wandb('Upload Patients Dataset', qa=True)
#
# artifact = wandb.Artifact('eval_results_Qwen3-8B', type='dataset')
# artifact.add_file('data/sft_results/eval_results_Qwen3-8B.pq')
# wandb_run.log_artifact(artifact)
#
# artifact = wandb.Artifact('eval_results_Qwen3-8b-full-2e-JSON', type='dataset')
# artifact.add_file('data/sft_results/Qwen3-8b-full-2e-JSON.pq')
# wandb_run.log_artifact(artifact)

wandb_run.finish()
