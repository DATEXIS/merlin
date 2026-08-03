# MeRLIn-DDx — Anonymized Code Release

Code accompanying the submission *MeRLIn-DDx: A Clinically Grounded Framework for
Diagnostic Reasoning*. This repository is provided for review under double-blind
conditions: identifiers (usernames, registry hosts, tracking entities, cluster
namespaces) have been replaced with neutral placeholders. The cluster topology,
image structure and config schema are unchanged, so the pipeline is documented
end to end.

MeRLIn-DDx reframes differential diagnosis (DDx) as an explicit, *verifiable*
reasoning process. A four-stage generator–verifier pipeline turns real admission
notes into reasoning traces that are cross-checked against medical ground truth
at every step; the resulting traces are used for supervised fine-tuning (SFT) of
open-weight LLMs, which are then evaluated on discharge ICD-10 prediction.

The dataset comprises ~115k synthetic reasoning traces derived from ~10.5k
MIMIC-IV admissions across 7 chief complaints and 20 clinical specialties.

---

## 1. What is and is not in this repository

**Included**

| Path | Contents |
| :--- | :--- |
| `src/` | All library code: pipeline, preprocessing, postprocessing, fine-tuning, QA/error analysis, evaluation, k8s templating. |
| `scripts/` | Thin CLIs. Every script is driven by a YAML config; the command line only selects *what* to run. |
| `k8s/` | Standalone manifests (vLLM server/client, PVCs, debug pod, QA job). |
| `deployment/` | Dockerfiles and pinned requirements for the server / client / fine-tuning images. |
| `data/medical_schemes/` | WikiDoc-derived symptom and diagnosis schemas per chief complaint (not MIMIC-derived). |
| `data/results/` | **Aggregate** metric tables only — `eval_metrics_*.csv`, QA rate tables, encoder baseline summary. |
| `figures/` | Every figure in the paper and appendix, as emitted by `src/eval/`. |
| `notebooks/` | Diagram notebook used for the pipeline schematic. |

**Deliberately excluded**

- **All MIMIC-IV data and anything derived from it** — raw notes, labs, preprocessed
  splits, reasoning traces, instruction datasets, and per-admission prediction
  dumps. These fall under the PhysioNet Credentialed Health Data License and
  cannot be redistributed. Credentialed users can regenerate every one of them
  from the scripts here.
- Model checkpoints (size), WandB run directories, and credentials.

Row-level artifacts keyed by `subject_id` / `hadm_id` are excluded even where
they are small, for the same licensing reason. The metric tables under
`data/results/` are aggregates over the 2,184-admission test split and contain no
patient-level rows.

---

## 2. Requirements

**Data**

- **MIMIC-IV** (credentialed PhysioNet access) — admission notes, lab events
  within 12 h of admission, and discharge ICD-10 codes.
- **WikiDoc** — the clinical knowledge base behind the symptom/disease mappings.
  The distilled schemas are already checked in under `data/medical_schemes/`.

**Infrastructure**

- A **Kubernetes** cluster with GPU nodes. The generation, fine-tuning and
  evaluation loops are all launched as k8s Jobs against vLLM servers; namespace,
  PVC names and image references live in the YAML configs.
- A container registry reachable from the cluster (placeholder:
  `registry.example.com/anon/...`).
- **Weights & Biases** for experiment tracking and as the transport for
  generation artifacts and eval results.

Copy `.env.example` to `.env` and fill in `WANDB_API_KEY`,
`HUGGING_FACE_HUB_TOKEN`, and your entity/project.

---

## 3. The generator–verifier pipeline

Each stage runs an instruction → generation → verification loop. A trace is kept
only if the verifier score clears the configured threshold.

| Stage | Task | Verified against |
| :--- | :--- | :--- |
| **V1** | Symptom extraction from the admission note | WikiDoc-derived symptom schema |
| **V2** | Differential diagnosis ranking | WikiDoc symptom–disease mappings |
| **V3** | Laboratory-informed re-ranking (labs ≤12 h post-admission) | MIMIC-IV lab events |
| **V4** | Discharge ICD-10 code prediction | MIMIC-IV discharge codes |

---

## 4. Configuration

Settings live in YAML, one file per concern; flags only select stages.

| Config | Drives | Contents |
| :--- | :--- | :--- |
| `scripts/pipeline_config.yaml` | `pipeline_start` / `pipeline_stop` / `pipeline_restart_client` | Generation: model selection, verifier thresholds, deployment settings. |
| `scripts/run_plan.yaml` | `launch_all` | Every SFT run (LoRA and full-FT, incl. FSDP) as one plan: shared `defaults` plus per-run overrides. |
| `scripts/eval_config.yaml` | `eval_checkpoints`, `eval_base_models`, `eval_analysis` | Checkpoint sweep manifests, base-model list, shared serving/client params, and the `Analysis:` block for local post-processing. |
| `scripts/qa_config.yaml` | `qa_analysis`, `qa_deterministic`, `qa_full_code` | Error-taxonomy and ICD-granularity analyses. |

The `eval_*_config_*.yaml` and `watch_epochs_*.yaml` files are the concrete
manifests behind individual reported sweeps (seeds 43/44, the 8B LoRA reruns,
base-model and external-model baselines).

---

## 5. Workflow

| Step | Command | Description |
| :--- | :--- | :--- |
| 1. Preprocess | `python scripts/run_preprocesssing.py` | Cleans and formats raw MIMIC-IV data for the 7 chief complaints. |
| 2. Generate | `python scripts/pipeline_start.py` | Runs the 4-stage generator–verifier pipeline on the cluster. |
| 3. Download + postprocess | `python scripts/download_gen_data.py` then `python scripts/run_postprocessing.py` | Fetches generation artifacts from WandB, merges them, builds the instruction-dataset variants. |
| 4. Upload datasets | `python scripts/upload_datasets_to_pvc.py` | Pushes the instruction datasets to the cluster PVC. |
| 5. Fine-tune | `python scripts/launch_all.py [--only <substr>]` | Launches all (or a subset of) SFT runs from `run_plan.yaml` as parallel k8s Jobs. |
| 6. Evaluate | `python scripts/eval_checkpoints.py` · `eval_base_models.py` | Serves each checkpoint / base model with vLLM and runs the downstream eval (resumable). |
| 7. Analyze + plot | `python scripts/eval_analysis.py wandb_check new_evals create_plots` | Pulls new results, re-evaluates locally, writes the reshaped results table, renders quick-look and publication figures. |
| 8. Error analysis | `python scripts/qa_analysis.py` · `qa_deterministic.py` · `qa_full_code.py` | Error taxonomy, deterministic failure rates, ICD granularity breakdown. |

`eval_analysis.py` also exposes two independent stages: `encoder_results` (folds
in the encoder-classifier baselines) and `clinibench` (recomputes the encoder
incumbent's baselines from its released per-admission predictions on our test
split — see `src/eval/clinibench_headtohead.py` for the protocol and the
reproduction audit).

All analysis and plotting logic lives in `src/eval/`; the scripts are thin CLIs.

---

## 6. Container images

```bash
python scripts/docker_build.py          # builds + pushes server / client / fine-tuning images
```

Dockerfiles are in `deployment/`. Replace the registry placeholder in the YAML
configs (`server_image`, `client_image`) with your own before building.

---

## 7. Reproducing the reported numbers

With credentialed MIMIC-IV access and a GPU cluster, steps 1–7 regenerate every
table and figure in the paper. Without cluster access, the aggregate tables under
`data/results/` are the exact inputs to the plotting code, so
`python scripts/eval_analysis.py create_plots` reproduces the figures in
`figures/` directly.

---

## 8. License

Code is released for review purposes. The MeRLIn-DDx dataset itself is
distributed under the **PhysioNet Credentialed Health Data License** and is not
included in this repository.
