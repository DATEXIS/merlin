import subprocess
from pathlib import Path

import pandas as pd
from datasets import Dataset

# Upload instruction datasets to a Kubernetes PVC in HuggingFace dataset format.
# Reads parquet files from data/results/instructions/, converts each to a HF
# Dataset (arrow format), saves locally, then kubectl cp to the PVC.
#
# Datasets are namespaced by version on the PVC so different rounds/lineages
# don't mix:
# /ft_models/datasets/<version>/<dataset_name>
# Edit DATASETS below per upload round (mirrors the MERLIN_VARIANTS
# config-at-top-of-file pattern in run_postprocessing.py). Old versions are never
# touched by a later run — nothing here overwrites or deletes prior uploads.
# Re-running with the same name+version is safe (idempotent overwrite of just
# that dataset's folder).

# .pq stem (data/results/instructions/<name>.pq) -> PVC version folder.
# merlin_* and mimic_instructions are separate lineages (see run_plan job_names:
# ...-merlin-vX vs. ...-mimic-vX), so they get independent version tags even
# when built in the same postprocessing pass.
DATASETS = {
    # "merlin_replace": "merlin-v1.3",
    # "merlin_thr_full": "merlin-v1.3",
    # "merlin_thr_half": "merlin-v1.3",
    # "merlin_best_per_id": "merlin-v1.3",
    # "merlin_icd_2x": "merlin-v1.3",
    # "mimic_instructions": "mimic-v1.3", # OLD single-stage (V4-only) mimic — already uploaded, don't re-touch

    # Added after seeing the initial v1.3 ablation results (thr_full and
    # icd_2x each improved fine-tuning individually) — still the v1.3 round,
    # no underlying source data changed, just two new derived variants:
    # merlin_thrfull_icd2x — thr_full's thresholds + icd_2x's V4 duplication, combined.
    # mimic_fixed — full V1-V4 mimic (old mimic_instructions was V4/ICD only).
    # mimic_fixed_icd_2x — mimic_fixed with V4 duplicated (icd_2x carried over to mimic).
    # "merlin_thrfull_icd2x": "merlin-v1.3",
    # "mimic_fixed": "mimic-v1.3",
    # "mimic_fixed_icd_2x": "mimic-v1.3",

    # v1.4 round: rebuilt from the corrected postprocessing (match_category rank
    # filter now actually applied before the single-cc-per-subject assignment,
    # see src/postprocessing/merge_gen_data.py -- headache in particular was
    # previously ~50% unmatched/other rows that leaked through). Re-run
    # scripts/run_postprocessing.py first so these.pq files reflect the fix,
    # then upload here. New version folder (not v1.3) so the old, uncorrected
    # checkpoints' datasets are never touched.
    # merlin_thrfull_icd2x / mimic_fixed_icd_2x -- the two winning combos,
    # re-run on corrected data.
    # merlin_icd_2x / mimic_fixed -- added: for run_plan.yaml's
    # ablation-arm runs (icd_2x alone vs. thrfull_icd2x combined; mimic_fixed
    # alone vs. mimic_fixed_icd_2x). These standalone variants already exist
    # locally (data/results/instructions/*.pq, rebuilt ) but were
    # never pushed to a v1.4 PVC folder until now.
    "merlin_thrfull_icd2x": "merlin-v1.4",
    "merlin_icd_2x":        "merlin-v1.4",
    "mimic_fixed_icd_2x":   "mimic-v1.4",
    "mimic_fixed":          "mimic-v1.4",

    # New ablation added: same thr_full thresholds, but
    # below-threshold/nan/0 rows are dropped instead of mimic-fallback'd (see
    # MERLIN_VARIANTS["merlin_thrfull_drop"] in run_postprocessing.py). Not
    # expected to outperform thr_full -- control for what the fallback buys.
    # Same lineage/round as the rest of v1.4, just a new derived variant name.
    "merlin_thrfull_drop":  "merlin-v1.4",

    # for the 4-run extension of run_plan.yaml (runs
    # 9-11: 8B full-FT on the remaining v1.3 dataset ablations under corrected
    # v1.4 postprocessing). The.pq files already exist locally from the v1.4
    # run_postprocessing.py rebuild (it builds every MERLIN_VARIANTS entry);
    # they just were never pushed to a v1.4 PVC folder.
    "merlin_replace":       "merlin-v1.4",
    "merlin_thr_full":      "merlin-v1.4",
    "merlin_thr_half":      "merlin-v1.4",
}

if __name__ == "__main__":
    namespace = 'merlin'
    tool_name = 'tool-anon'
    local_instructions_dir = Path('data/results/instructions')
    # tool-anon mounts merlin-ddx-ckpts-pvc at /merlin_ddx; training/server mount the
    # SAME pvc at /ft_models. So this writes PVC-relative 'datasets/<version>/<name>',
    # which the training job loads as /ft_models/datasets/<version>/<name> (matches
    # each run's dataset_path in run_plan.yaml). Keep these
    # in sync.

    for name, version in DATASETS.items():
        pq_file = local_instructions_dir / f"{name}.pq"
        if not pq_file.exists():
            raise FileNotFoundError(
                f"{pq_file} not found — run scripts/run_postprocessing.py first"
            )

        remote_dir = f'/merlin_ddx/datasets/{version}/'
        # Local staging mirrors the remote layout: instructions/<version>/<name>/
        local_hf_dir = local_instructions_dir / version / name

        # Convert parquet -> HuggingFace Dataset (arrow format)
        print(f"Converting {pq_file} -> {local_hf_dir} ...")
        df = pd.read_parquet(pq_file)
        ds = Dataset.from_pandas(df, preserve_index=False)
        ds.save_to_disk(str(local_hf_dir))
        print(f"  {len(ds):,} rows saved.")

        # Ensure the PARENT dir exists, then cp the local dir INTO it. kubectl cp of
        # a directory into an existing target dir nests it, so we must target the
        # parent (remote_dir) — copying into remote_path would create
        # remote_path/<dataset_name>/<dataset_name>.
        remote_path = f"{remote_dir}{name}"
        mkdir_command = [
            "kubectl", "exec", "-n", namespace, tool_name,
            "--", "mkdir", "-p", remote_dir
        ]
        try:
            subprocess.run(mkdir_command, check=True, capture_output=True, text=True)
        except subprocess.CalledProcessError as e:
            print(f"mkdir failed: {e.stderr}")

        # Upload to PVC (cp into the parent dir -> creates remote_dir/<dataset_name>)
        command = [
            "kubectl", "cp",
            str(local_hf_dir),
            f"{namespace}/{tool_name}:{remote_dir.rstrip('/')}"
        ]

        print(f"Uploading {local_hf_dir} -> {remote_path} ...")
        try:
            result = subprocess.run(command, check=True, capture_output=True, text=True)
            print("Upload successful!")
            print(result.stdout)
        except subprocess.CalledProcessError as e:
            print(f"Upload failed: {e.stderr}")

    versions = sorted(set(DATASETS.values()))
    print(f"\nDone. Remote layout: /merlin_ddx/datasets/<version>/<name> "
          f"for version in {versions} "
          f"(training sees /ft_models/datasets/<version>/<name>)")
