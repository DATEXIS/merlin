import itertools
import logging

import pandas as pd

from src.exp_args import ExpArgs


def load_mimic_notes(exp_args: ExpArgs) -> pd.DataFrame:
    icds = [9, 10]
    data_sources = ['hosp', 'icu']
    splits = ['train', 'dev', 'test']

    all_notes = []

    for icd, data_source in itertools.product(icds, data_sources):

        # Load discharge notes
        discharge_dfs = []
        for split in splits:
            path = exp_args.get_mimic_note_path('discharge', split, icd, data_source)
            df = pd.read_parquet(path)
            df.rename(columns={'text': 'discharge_note'}, inplace=True)
            discharge_dfs.append(df[['hadm_id', 'discharge_note']])

        discharge_notes = pd.concat(discharge_dfs, ignore_index=True)

        # Load admission notes and merge with discharge notes
        for split in splits:
            path = exp_args.get_mimic_note_path('admission', split, icd, data_source)
            notes = pd.read_parquet(path)
            notes.rename(columns={'TEXT': 'admission_note'}, inplace=True)

            notes = notes.merge(discharge_notes, on='hadm_id', how='left')
            notes = notes[['subject_id', 'hadm_id', 'charttime',
                           'LONG_CODES', 'admission_note', 'discharge_note']]

            notes['ICD_version'] = icd
            notes['data_source'] = data_source
            notes['split'] = split

            all_notes.append(notes)

    note_df = pd.concat(all_notes, ignore_index=True)
    # Rename to ICD_CODES as SHORT_CODES are deprecated
    note_df.rename(columns={'LONG_CODES': 'ICD_CODES'}, inplace=True)

    print(f"Loaded {len(note_df)} notes. Columns: {note_df.columns.tolist()}")
    return note_df


def filter_mimic_by_meta_data(
        df: pd.DataFrame,
        icd_version: int = None,
        data_source: str = None,
        split: str = None,
        chief_complaint: str = None,
) -> pd.DataFrame:
    """
    Filter MIMIC notes DataFrame by metadata columns.
    """
    filtered_df = df.copy()

    if icd_version is not None:
        print(f'Filtering by ICD version: {icd_version}')
        filtered_df = filtered_df[filtered_df['ICD_version'] == icd_version]

    if data_source is not None:
        print(f'Filtering by data source: {data_source}')
        filtered_df = filtered_df[filtered_df['data_source'] == data_source]

    if split is not None:
        print(f'Filtering by split: {split}')
        filtered_df = filtered_df[filtered_df['split'] == split]

    if chief_complaint is not None and 'cc_list' in filtered_df.columns:
        print(f'Filtering by chief complaint: {chief_complaint}')
        filtered_df = filtered_df[filtered_df['cc_list'].apply(
            lambda cc_list: chief_complaint in cc_list
        )]

    logging.info(f'Filtered mimic df from {len(df)} to {len(filtered_df)} notes')
    return filtered_df
