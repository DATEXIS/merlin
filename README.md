# MERLIN-DDX — Anonymized Code Release

Code accompanying the submission *MERLIN-DDX: Learning Diagnostic Reasoning
from Traces Verified Against External Clinical References*. Provided for review
under double-blind conditions: identifiers (usernames, registry hosts, tracking
entities, cluster namespaces) are replaced with neutral placeholders. Cluster
topology, image structure and config schema are unchanged, so the pipeline is
documented end to end.

Clinical notes record conclusions, not the reasoning behind them. MERLIN-DDX
builds that missing supervision: a four-stage generator–verifier pipeline
samples diagnostic reasoning traces over real hospital admissions and checks
every intermediate step against an external clinical reference (WikiDoc) and the
recorded outcome (MIMIC-IV). Traces that clear their stage threshold become
supervised fine-tuning data for Qwen3 students.

Three terms are used consistently here and in the paper: **the pipeline** is the
method; **MERLIN-DDX** is the dataset it produces; **+ MERLIN** marks a Qwen3
base model fine-tuned on that dataset.

---

## 1. The dataset in numbers

| | |
| :--- | :--- |
| Admissions | 10,400 (train 7,299 / dev 917 / test 2,184) |
| Four-stage chains | 31,200 (each admission run by all three generators) |
| Stage-level reasoning records | 124,510, of which 117,580 carry a verifier score |
| Chief complaints | 7 (dyspnea, abdominal pain, chest pain, cough, back pain, diarrhea, headache) |
| WikiDoc coverage | 155 distinct symptom features, 361 candidate diagnoses |
| Label space | 1,157 distinct three-character ICD codes, median 12 per admission |
| Train-split SFT records | 82,789 = 55,974 accepted traces + 26,815 label-only fallbacks |
| Generators | Qwen3-32B, MedGemma-27B, Llama-3.3-70B-Instruct |
| Students | Qwen3 0.6B / 8B / 14B / 32B |

Primary metric is **macro-F1 over three-character ICD codes** — the label space
is large and heavily imbalanced, so macro weights each category equally instead
of letting frequent codes dominate.

---

## 2. What is and is not in this repository

**Included**

| Path | Contents |
| :--- | :--- |
| `src/` | All library code: pipeline, preprocessing, postprocessing, fine-tuning, evaluation, error analysis, k8s templating. |
| `src/pipeline/prompts.py` | The full generation prompts for all four stages (Appendix H of the paper). |
| `scripts/` | Thin CLIs. Every script is driven by a YAML config; the command line only selects *what* to run. |
| `k8s/` | Standalone manifests (vLLM server/client, PVCs, debug pod). |
| `deployment/` | Dockerfiles and pinned requirements for the server / client / fine-tuning images. |
| `data/medical_schemes/` | WikiDoc-derived symptom and diagnosis schemas per chief complaint (not MIMIC-derived). |
| `data/results/` | **Aggregate** metric tables only — `eval_metrics_*.csv`, error-analysis rate tables, encoder baseline summary. |
| `figures/` | The figures the paper and appendix use, as emitted by `src/eval/`. |

This release is scoped to what the paper reports: `scripts/` carries one entry
point per pipeline stage, and `src/eval/` one module per paper figure or table
plus the shared machinery they build on (`plot_common.py`,
`checkpoint_metrics.py`, `classification_metrics.py`, `eval_results.py`).
Exploratory and superseded figure variants are not included.

The working tree also carried ~20 per-sweep config variants (one per seed ×
model-size × dataset arm), differing only in `job_name` / `short_name`, `seed`
and GPU fields. One representative of each kind is kept:
`eval_base_models_config_test.yaml` for a test-split baseline sweep and
`eval_error_bars_config_seed43.yaml` for a seed-repeat eval.

**Deliberately excluded**

- **All MIMIC-IV data and anything derived from it** — admission notes, labs,
  preprocessed splits, reasoning traces, instruction datasets, and per-admission
  prediction dumps. These fall under the PhysioNet Credentialed Health Data
  License and cannot be redistributed. Credentialed users can regenerate every
  one of them from the scripts here.
- Model checkpoints (size), experiment-tracking run directories, credentials.

Row-level artifacts keyed by `subject_id` / `hadm_id` are excluded even where
small, for the same reason. The tables under `data/results/` are aggregates over
the 2,184-admission test split and contain no patient-level rows.

---

## 3. Requirements

**Data**

- **MIMIC-IV** (credentialed PhysioNet access) — the ICD-10-coded `hosp` subset.
  Admission notes are reconstructed following van Aken et al. (2021); labs are
  those drawn within 12 h of admission; labels are the discharge ICD codes.
- **WikiDoc** — the external clinical reference. Each chief complaint maps to one
  WikiDoc differential-diagnosis table; the distilled schemas are checked in
  under `data/medical_schemes/`.

**Infrastructure**

- A **Kubernetes** cluster with GPU nodes. Generation, fine-tuning and evaluation
  all run as k8s Jobs against vLLM servers; namespace, PVC names and image
  references live in the YAML configs.
- A container registry reachable from the cluster (placeholder:
  `registry.example.com/anon/...`).
- **Weights & Biases** for experiment tracking and as the transport for
  generation artifacts and eval results.

Copy `.env.example` to `.env` and fill in `WANDB_API_KEY`,
`HUGGING_FACE_HUB_TOKEN`, and your entity/project.

---

## 4. The generator–verifier pipeline

Each stage pairs a generation prompt with a verifier that scores the output
against an external reference target. The generator samples **30 candidate
traces per round** and keeps the best-scoring one; if it falls below the stage
threshold the round repeats, up to **4 rounds**, after which the stage falls back
to a label-only target. Every accepted trace retains its verifier score, so the
dataset can be re-filtered at other thresholds.

| Stage | Task | Reference target | Verifier metric | Accept at |
| :--- | :--- | :--- | :--- | :--- |
| **V1** | Symptom extraction | WikiDoc symptom list for the confirmed disease | NormDot | ≥ 0.60 |
| **V2** | Diagnosis ranking | WikiDoc differential for the chief complaint | Reciprocal rank | ≥ 0.50 |
| **V3** | Lab-informed reranking (labs ≤ 12 h) | as V2 | Reciprocal rank | ≥ 0.80 |
| **V4** | Discharge ICD-code prediction | MIMIC-IV discharge codes | micro-F1 | ≥ 0.55 |

Thresholds come from a fixed construction rule — each sits at or just above its
stage's mean verifier score — not from tuning on downstream performance.

**At evaluation time only V2 and V4 are scored.** V3 is skipped: lab values are
not admission-time evidence, and using them would break the task definition. V1
still runs because V2 conditions on it, but NormDot is a construction-time
acceptance gate measuring agreement with a disease-level prototype, not a
per-note extraction metric — it is reported as a pipeline diagnostic only.

---

## 5. Configuration

Settings live in YAML, one file per concern; flags only select stages.

| Config | Drives | Contents |
| :--- | :--- | :--- |
| `pipeline_config.yaml` | `pipeline_start` / `pipeline_stop` | Generation: generator selection, verifier thresholds, round budget, deployment. |
| `run_plan.yaml` | `launch_all` | Every SFT run (LoRA and full-FT, incl. FSDP) as one plan: shared `defaults` plus per-run overrides. Backs Table 5. |
| `eval_config.yaml` | `eval_checkpoints`, `eval_base_models`, `eval_analysis` | Checkpoint sweep manifests, base-model list, shared serving/client params, and the `Analysis:` block for local post-processing. |
| `eval_base_models_config_test.yaml` | `eval_base_models` | Test-split eval of the untuned bases and external baselines — the baseline rows of Table 1. |
| `eval_error_bars_config_seed43.yaml` | `eval_base_models`, `eval_checkpoints` | Re-evaluates fixed checkpoints at a different client sampling seed; source of the ± in Table 1 and the whiskers in Figure 1. |

---

## 6. Workflow

| Step | Command | Description |
| :--- | :--- | :--- |
| 1. Preprocess | `python scripts/run_preprocesssing.py` | Builds the admission notes and splits from raw MIMIC-IV for the 7 chief complaints. |
| 2. Generate | `python scripts/pipeline_start.py` | Runs the four-stage generator–verifier loop on the cluster. |
| 3. Download + postprocess | `python scripts/download_gen_data.py` then `python scripts/run_postprocessing.py` | Fetches generation artifacts, merges the three generators, builds the instruction-dataset variants (accepted + fallback, ICD-row duplication, drop control). |
| 4. Upload datasets | `python scripts/upload_datasets_to_pvc.py` | Pushes the instruction datasets to the cluster PVC. |
| 5. Fine-tune | `python scripts/launch_all.py [--only <substr>]` | Launches all (or a subset of) SFT runs from `run_plan.yaml` as parallel k8s Jobs. |
| 6. Evaluate | `python scripts/eval_checkpoints.py` · `eval_base_models.py` | Serves each checkpoint / base model with vLLM and runs the admission-time eval (resumable). |
| 7. Analyze + plot | `python scripts/eval_analysis.py wandb_check new_evals create_plots` | Pulls new results, re-evaluates locally, writes the reshaped results table, renders every paper figure. |
| 8. Error analysis | `python scripts/qa_deterministic.py` · `qa_full_code.py` | The error decomposition behind Table 2 and the granularity breakdown. Every class is an exact set operation on gold vs. predicted three-character categories; no model scores them. |
| 9. Dataset analyses | `python scripts/dataset_analyses.py` · `scripts/analysis/*.py` | Corpus-level figures: label-frequency long tail, per-chief-complaint mapping quality. |

Epoch selection is on **dev** by ICD macro-F1; the selected checkpoint is
reported on the held-out **test** split.

`eval_analysis.py` also exposes two independent stages: `encoder_results` (folds
in the encoder classifiers retrained on our subset) and `clinibench` (rescores
the published encoder baselines on our test admissions under one stated
protocol — see `src/eval/clinibench_headtohead.py` for the protocol and the
reproduction audit).

Figures are written to `figures/`, LaTeX tables to `tables/`, CSV outputs to
`data/results/`.

---

## 7. Where each paper figure and table comes from

| Paper | Generated by |
| :--- | :--- |
| Figure 1 — ICD macro-F1 by scale, base vs. + MERLIN | `src/eval/main_figure_plots_macro.py` |
| Figures 2, 5 — pipeline schematic and implementation view | Hand-drawn (TikZ); PDFs in `figures/` |
| Figure 3 — data-construction ablation | `src/eval/ablation_plots.py` |
| Figure 4 — WikiDoc mapping quality per chief complaint | `scripts/analysis/match_category_by_chief_complaint.py` |
| Figure 6 — verifier scores across generation rounds | `src/eval/generation_plots.py` |
| Figure 7 — epoch-selection trade-off | `src/eval/epoch_tradeoff_plots.py` |
| Figure 8 — training-seed robustness (8B, both arms) | `src/eval/robustness_plots.py` |
| Figure 9 — label-frequency distribution | `src/eval/dataset_analyses/icd_longtail.py` |
| Table 1 — main test-split results | `src/eval/paper_tables.py` (`table_main`) |
| Table 2 — where the ICD gain comes from | `src/eval/paper_tables.py` (`table_qa`), `scripts/qa_deterministic.py` |
| Tables 3, 4 — dataset composition and stage accounting | Counts emitted by `scripts/run_postprocessing.py` |
| Table 5 — per-run fine-tuning configuration | `scripts/run_plan.yaml` |
| Table 6 — chief-complaint breakdown (32B + MERLIN) | `scripts/analysis/cc_metrics_table_32b_full.py` |
| Table 7 — encoder ICD-prediction baselines | `src/eval/encoder_metrics.py`, `src/eval/clinibench_headtohead.py` |
| Table 8 — head / body / tail detail | `src/eval/longtail_strata.py` |
| Table 9 — data-construction ablation, exact values | `src/eval/paper_tables.py` (`table_ablation`) |
| Table 10 — LoRA vs. full fine-tuning | `src/eval/paper_tables.py` (`table_lora`) |
| Appendix E.2 — NormDot gate values | `src/eval/paper_tables.py` (`table_symptom`) |

---

## 8. Container images

```bash
python scripts/docker_build.py          # builds + pushes server / client / fine-tuning images
```

Dockerfiles are in `deployment/`. Replace the registry placeholder in the YAML
configs (`server_image`, `client_image`) with your own before building.

---

## 9. Reproducing the reported numbers

With credentialed MIMIC-IV access and a GPU cluster, steps 1–9 regenerate every
table and figure. Without cluster access, the aggregate tables under
`data/results/` are the exact inputs to the plotting code, so
`python scripts/eval_analysis.py create_plots` reproduces the figures in
`figures/` directly.

---

## 10. Licensing

MIMIC-IV is used under the **PhysioNet Credentialed Health Data License 1.5.0**.
MERLIN-DDX is derived from it and will be released under the same terms,
requiring the same CITI training and credentialing — which is why no
MIMIC-derived artifact appears in this repository. The WikiDoc content used as
the external reference is licensed **CC BY-SA 3.0** and is used and attributed
under those terms.

MERLIN-DDX is a research resource, not a deployed diagnostic tool, and must not
inform clinical decisions without prospective validation and regulatory review.
