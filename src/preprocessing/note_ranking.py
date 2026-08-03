import logging
import re

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def match_disease_from_codes(df: pd.DataFrame, wikidoc_df: pd.DataFrame) -> None:
    """
    Adds the following columns to df by matching ICD code regexes from wikidoc_df
    against the 'ICD_CODES' list in each row of df:
        - 'diseases_by_icd_match': list of matched diseases
        - 'icd_codes_rank': list of indices of matching ICD codes
        - 'icd_codes_match_count': number of matched codes

    Parameters:
        df: DataFrame with column 'ICD_CODES' (list of ICD codes)
        wikidoc_df: DataFrame with columns 'Disease' and 'icd10_regex'

    Returns:
        DataFrame with additional columns
    """
    # Precompile regex patterns for speed
    disease_patterns = [
        (disease, re.compile(pattern))
        for disease, pattern in zip(wikidoc_df['Disease'], wikidoc_df['icd10_regex'])
    ]

    all_matched_diseases = []
    all_matched_ranks = []
    match_counts = []

    for codes in df['ICD_CODES']:
        matched_diseases = []
        matched_ranks = []

        for idx, code in enumerate(codes):
            for disease, pattern in disease_patterns:
                if pattern.match(code):
                    matched_diseases.append(disease)
                    matched_ranks.append(idx)
                    break  # Stop at first matching disease for this code

        if matched_diseases:
            all_matched_diseases.append(matched_diseases)
            all_matched_ranks.append(matched_ranks)
            match_counts.append(len(matched_diseases))
        else:
            all_matched_diseases.append([None])
            all_matched_ranks.append([-1])
            match_counts.append(0)

    # df = df.copy()
    df['diseases_by_icd_match'] = all_matched_diseases
    df['icd_codes_rank'] = all_matched_ranks
    df['icd_codes_match_count'] = match_counts

    # pick the first match if it exists, else “no match” sentinel
    df["disease_by_icd_match"] = df["diseases_by_icd_match"].apply(
        lambda lst: lst[0] if lst else None
    )
    df["disease_icd_rank"] = df["icd_codes_rank"].apply(
        lambda lst: lst[0] if lst else -1
    )


def calculate_disease_cossim_rank(df):
    """
    For each row, finds the rank (index) of the value in 'disease' within
    the list 'diseases'. Stores result in 'disease_cossim_rank'.

    Returns:
        Modified df with new column 'disease_cossim_rank'.
    """
    def find_rank(row):
        disease = row['disease_by_icd_match']
        candidates = row['diseases']
        if pd.isna(disease) or not isinstance(candidates, list):
            return -1
        try:
            return candidates.index(disease)
        except ValueError:
            return -1

    df['disease_cossim_rank'] = df.apply(find_rank, axis=1)
    # return df


def categorize(row):
    # | Category               | Condition                                            |
    # | ---------------------- | ---------------------------------------------------- |
    # | 0                      | Exact match (0, 0)                                   |
    # | 1                      | Near match  (0, 1), (1, 0), (1, 1)                   |
    # | 2                      | Fuzzy match (involving rank 2)                       |
    # | 3                      | Anything else                                        |
    # | -1                     | Any rank is `-1` (not found in predictions)          |
    icd_rank = row['disease_icd_rank']
    cossim_rank = row['disease_cossim_rank']

    if (icd_rank, cossim_rank) == (0, 0):
        return 0
    elif (icd_rank, cossim_rank) in {(0, 1), (1, 0), (1, 1)}:
        return 1
    elif (icd_rank, cossim_rank) in {(0, 2), (2, 0), (1, 2), (2, 1), (2, 2)}:
        return 2
    elif icd_rank == -1 or cossim_rank == -1:
        return -1
    else:
        return 3


def get_summary_df(note_df: pd.DataFrame) -> pd.DataFrame:
    # Group and count
    summary = (
        note_df
        .groupby(['split', 'match_category'])
        .size()
        .unstack(fill_value=0)
    )

    # Add total per split (optional)
    summary['total'] = summary.sum(axis=1)
    summary.loc['total'] = summary.sum(numeric_only=True)

    row_order = ['train', 'dev', 'test', 'total']
    col_order = [0, 1, 2, 3, -1, 'total']

    # Filter to existing rows in summary
    existing_rows_order = [row for row in row_order if row in summary.index]
    existing_cols_order = [col for col in col_order if col in summary.columns]

    # Reorder the rows
    summary = summary.loc[existing_rows_order]
    summary = summary[existing_cols_order]

    return summary


def plot_rank_heatmap(df, max_rank=5, title=None):
    """
    Plots a heatmap of counts of (disease_icd_rank, disease_cossim_rank) pairs,
    with proper numerical sorting and separate display of -1 (unmatched cases).

    Parameters:
        df: DataFrame with 'disease_icd_rank' and 'disease_cossim_rank' columns
        max_rank: max integer rank to display individually; higher values grouped into '10+'
        title: title of the plot
    """
    title = title or "Joint Distribution of ICD Code Rank and Diagnosis Similarity Rank"

    # Modified bucket function to preserve -1
    def bucket(rank):
        try:
            rank = int(rank)
            if rank == -1:
                return "-1"
            elif 0 <= rank < max_rank:
                return str(rank)
            else:
                return f"{max_rank}+"
        except:
            return f"{max_rank}+"

    df_filtered = df.copy()
    df_filtered['icd_bucket'] = df_filtered['disease_icd_rank'].apply(bucket)
    df_filtered['cossim_bucket'] = df_filtered['disease_cossim_rank'].apply(bucket)

    # Count occurrences
    heatmap_data = (
        df_filtered
        .groupby(['icd_bucket', 'cossim_bucket'])
        .size()
        .unstack(fill_value=0)
    )

    # Custom sort key (puts -1 at top/left, 0–9 in order, 10+ last)
    def sort_key(x):
        return -1 if x == "-1" else int(x.replace('+', '999'))

    # Apply sorted order to rows and columns
    heatmap_data = heatmap_data.reindex(
        index=sorted(heatmap_data.index, key=sort_key),
        columns=sorted(heatmap_data.columns, key=sort_key)
    )

    # Plot
    plt.figure(figsize=(7, 6))
    sns.heatmap(heatmap_data, annot=True, fmt="d", cmap="Blues")
    plt.title(title)
    plt.xlabel("disease_cossim_rank")
    plt.ylabel("disease_icd_rank")
    plt.gca().invert_yaxis()  # Show -1 and 0 at top
    plt.tight_layout()
    plt.show()


def add_rank_columns(note_df: pd.DataFrame, wikidoc_df: pd.DataFrame, show_plot=False):
    logging.info('Adding rank columns based on ICD code matching and disease cosine similarity')
    work_df = note_df.copy()
    match_disease_from_codes(work_df, wikidoc_df)
    calculate_disease_cossim_rank(work_df)

    work_df['match_category'] = work_df.apply(categorize, axis=1)
    note_df['match_category'] = work_df.apply(categorize, axis=1)

    if show_plot:
        plot_rank_heatmap(work_df)

    return get_summary_df(work_df)

