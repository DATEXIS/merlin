"""
Diagnoses Recall@3 (V2) broken out by chief complaint x model, on the TEST
split. Companion to match_category_by_chief_complaint.py -- same "per chief
complaint" cut, but for the headline V2 Recall@3 metric instead of retrieval
match_category.

Columns are the paper's base-vs-full-FT-across-scale family (06b/8b/14b/32b),
same models plot_heatmap_size_x_mode_test draws in paper_plots.py, read
directly from the row-level TEST eval_results parquets (each row already
carries Chief Complaint + gold `disease` + ranked `v2_preds`). Recall@3 here
is computed the same way as calculate_accuracy_at_k in
src/eval/classification_metrics.py: hit if gold appears in the top-3 of
v2_preds; a missing/unparseable v2_preds (V2 request/parse failure, see
memory v2_request_failure_vs_parse_failure) counts as a miss, not excluded --
this matches how the aggregate metric in eval_metrics_merlin-eval-1.4.csv is
computed, so a checkpoint with a JSON-validity outlier (e.g. 14b-full-e3)
shows it as depressed Recall@3 rather than silently dropping those rows.

External baselines (llama-3-70b-instruct, medgemma-27b-it, baichuan-m2-32b)
and LoRA checkpoints are NOT included -- add them to MODELS below if wanted.

Run from repo root: python scripts/analysis/recall3_by_chief_complaint.py
"""
import ast
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

EVAL_DIR = Path("data/results/evaluation/1_4")
OUT_CSV = Path("data/results/evaluation/recall3_by_chief_complaint.csv")
OUT_FIG_DIR = Path("paper/figures/dataset_analyses")
K = 3

# (column label, size, mode, checkpoint dir suffix)
MODELS = [
    ("06b\nbase", "06b", "base", "test-06b-base"),
    ("06b\nfull", "06b", "full", "test-06b-full-e4"),
    ("8b\nbase", "8b", "base", "test-8b-base"),
    ("8b\nfull", "8b", "full", "test-8b-full-e3"),
    ("14b\nbase", "14b", "base", "test-14b-base"),
    ("14b\nfull", "14b", "full", "test-14b-full-e3"),
    ("32b\nbase", "32b", "base", "test-32b-base"),
    ("32b\nfull", "32b", "full", "test-32b-full-e3"),
]


def _parse_preds(v):
    if not isinstance(v, str):
        return []
    try:
        parsed = ast.literal_eval(v)
        return parsed if isinstance(parsed, list) else []
    except (ValueError, SyntaxError):
        return []


def recall_at_k_by_cc(ckpt_name: str, k: int = K) -> pd.Series:
    path = EVAL_DIR / f"eval_results_{ckpt_name}" / f"eval_results_{ckpt_name}.pq"
    df = pd.read_parquet(path, columns=["Chief Complaint", "disease", "v2_preds"])
    df["hit"] = [
        gold in _parse_preds(preds)[:k]
        for gold, preds in zip(df["disease"], df["v2_preds"])
    ]
    by_cc = df.groupby("Chief Complaint")["hit"].mean()
    by_cc["All"] = df["hit"].mean()
    return by_cc


def main():
    cols = {}
    n_by_cc = None
    for label, size, mode, ckpt in MODELS:
        by_cc = recall_at_k_by_cc(ckpt)
        cols[label] = by_cc
        if n_by_cc is None:
            path = EVAL_DIR / f"eval_results_{ckpt}" / f"eval_results_{ckpt}.pq"
            n_df = pd.read_parquet(path, columns=["Chief Complaint"])
            n_by_cc = n_df["Chief Complaint"].value_counts()

    table = pd.DataFrame(cols) * 100  # percent
    cc_order = [c for c in n_by_cc.sort_values(ascending=False).index if c in table.index]
    table = table.loc[cc_order + ["All"]]
    table.insert(0, "n", [n_by_cc.get(cc, n_by_cc.sum()) for cc in table.index])

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    table.round(1).to_csv(OUT_CSV)
    print(table.round(1))
    print(f"\nsaved table -> {OUT_CSV}")

    # --- heatmap: rows=chief complaint (+ All), cols=model, cell=Recall@3 % --
    grid = table.drop(columns="n").to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(8, 5.5))
    im = ax.imshow(grid, cmap="YlOrRd", aspect="auto", vmin=0, vmax=100)
    ax.set_xticks(range(grid.shape[1]))
    ax.set_xticklabels(table.drop(columns="n").columns, fontsize=8)
    ax.set_yticks(range(grid.shape[0]))
    ax.set_yticklabels(table.index, fontsize=9)

    finite = grid[~np.isnan(grid)]
    vmin, vmax = finite.min(), finite.max()
    span = vmax - vmin
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            v = grid[i, j]
            frac = (v - vmin) / span if span > 0 else 0.5
            color = "white" if frac > 0.6 else "black"
            ax.text(j, i, f"{v:.1f}", ha="center", va="center", fontsize=8, color=color)
        # separator line above "All" row
    ax.axhline(grid.shape[0] - 1.5, color="black", linewidth=1)

    ax.set_title(f"Diagnoses Recall@{K} by chief complaint x model (TEST)")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Recall@3 (%)")
    fig.tight_layout()

    OUT_FIG_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(OUT_FIG_DIR / f"recall3_by_chief_complaint.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"saved heatmap -> {OUT_FIG_DIR / 'recall3_by_chief_complaint.png'}")


if __name__ == "__main__":
    main()
