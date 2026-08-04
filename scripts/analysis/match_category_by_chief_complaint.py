"""
Match-category quality breakdown per chief complaint, from data/results/gen_data/combined.pq.

match_category (see src/preprocessing/note_ranking.py::categorize):
    0 exact match
    1 near match
    2 fuzzy match
    3 anything else
   -1 not found in predictions (should have been filtered out pre-generation)

Run from repo root: python scripts/analysis/match_category_by_chief_complaint.py
"""
import matplotlib.pyplot as plt
import pandas as pd

COMBINED_PATH = "data/results/gen_data/combined.pq"
CATEGORY_ORDER = [0, 1, 2, 3, -1]
CATEGORY_LABELS = {0: "0 exact", 1: "1 near", 2: "2 fuzzy", 3: "3 other", -1: "-1 unmatched"}
CATEGORY_COLORS = {0: "#2c7fb8", 1: "#7fcdbb", 2: "#edf8b1", 3: "#feb24c", -1: "#e31a1c"}

df = pd.read_parquet(COMBINED_PATH, columns=["Chief Complaint", "match_category"])

# --- Table: counts + row-% per chief complaint x match_category -----------------
counts = pd.crosstab(df["Chief Complaint"], df["match_category"])
counts = counts.reindex(columns=[c for c in CATEGORY_ORDER if c in counts.columns])
counts["total"] = counts.sum(axis=1)

pct = counts.drop(columns="total").div(counts["total"], axis=0).mul(100).round(1)
pct.columns = [f"{c}_%" for c in pct.columns]

table = pd.concat([counts, pct], axis=1).sort_values("total", ascending=False)
print(table)
table.to_csv("data/results/evaluation/match_category_by_chief_complaint.csv")

# --- Diagram: 100%-stacked bar of match_category share per chief complaint ------
share = counts.drop(columns="total").div(counts["total"], axis=0).sort_values(0)
raw = counts.drop(columns="total").loc[share.index]  # same row order, absolute counts for labels

fig, ax = plt.subplots(figsize=(9, 5))
bottom = pd.Series(0.0, index=share.index)
for cat in CATEGORY_ORDER:
    if cat not in share.columns:
        continue
    bars = ax.barh(share.index, share[cat], left=bottom, label=CATEGORY_LABELS[cat],
                    color=CATEGORY_COLORS[cat])
    for bar, cc in zip(bars, share.index):
        n = raw.loc[cc, cat]
        if n == 0:
            continue
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_y() + bar.get_height() / 2,
                f"{n:,}", ha="center", va="center", fontsize=8)
    bottom += share[cat]

ax.set_xlabel("Share of rows")
ax.set_title("match_category distribution by chief complaint (combined.pq)")
ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
fig.tight_layout()
fig.savefig("figures/dataset_analyses/match_category_by_chief_complaint.png", dpi=150)
plt.show()
