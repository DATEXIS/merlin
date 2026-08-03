import pandas as pd

LAB_DATA_PATH = 'data/mimic-iv/lab_data'
ADMISSIONS_PATH = f'{LAB_DATA_PATH}/admissions.csv'
LABEVENTS_PATH = f'{LAB_DATA_PATH}/labevents_hosp_icd10.pq'


def load_labevents() -> pd.DataFrame:
    admissions = pd.read_csv(ADMISSIONS_PATH)
    labevents = pd.read_parquet(LABEVENTS_PATH)

    admissions = admissions[['hadm_id', 'admittime', 'dischtime']]
    abd_pain_labevents = labevents.merge(admissions, on='hadm_id')
    abd_pain_labevents['storetime'] = pd.to_datetime(abd_pain_labevents['storetime'])
    abd_pain_labevents['admittime'] = pd.to_datetime(abd_pain_labevents['admittime'])
    abd_pain_labevents['dischtime'] = pd.to_datetime(abd_pain_labevents['dischtime'])
    abd_pain_labevents['time_diff'] = abd_pain_labevents['storetime'] - abd_pain_labevents['admittime']

    return abd_pain_labevents


def extract_events_within_hours(labevents, hadm_id_list, hours=12, first_only=True):
    """
    Extract lab events within the first `hours` of admission for given hadm_ids.

    Parameters
    ----------
    hadm_id_list : list-like
        List of hospital admission IDs to filter.
    hours : int
        Time window (in hours) from admission.
    first_only : bool, optional (default=False)
        If True, keep only the first event per (hadm_id, itemid).
    """
    mask = (
            labevents['hadm_id'].isin(hadm_id_list)
            & (labevents['time_diff'] >= pd.Timedelta(microseconds=0))
            & (labevents['time_diff'] <= pd.Timedelta(hours=hours))
            & (labevents['storetime'] <= labevents['dischtime'])
    )
    labevents_within_hours = labevents.loc[mask].copy()

    if first_only:
        # keep the earliest row per hadm_id & itemid
        labevents_within_hours = (
            labevents_within_hours.sort_values(["hadm_id", "itemid", "storetime"])
            .groupby(["hadm_id", "itemid"], as_index=False)
            .first()
        )

    return labevents_within_hours


def format_lab_events_for_note(labevents: pd.DataFrame) -> pd.DataFrame:
    def row_to_string(r):
        parts = [
            str(r['valuenum']),
            str(r['valueuom']),
            str(r['label']),
            str(r['flag']),
        ]
        # # include comments only when it is **not** NaN
        # if pd.notna(r['comments']):
        #     parts.append(str(r['comments']))

        # drop 'nan' / 'None' artefacts that come from str()-ing missing values
        return ' '.join(p for p in parts if p.lower() not in {'nan', 'none', ''})

    labevents['labs'] = labevents.apply(row_to_string, axis=1)

    df_labs_aggregated = labevents.groupby('hadm_id')['labs'].apply(lambda x: ' | '.join(x)).reset_index()
    return df_labs_aggregated


def add_lab_data_to_notes(patients: pd.DataFrame, hours: int) -> pd.DataFrame:
    print(f"Adding lab data to notes for {len(patients)} patients using a {hours}-hour window.")
    labevents = load_labevents()
    print(f"Loaded {len(labevents)} lab events. Extracting events within {hours} hours for "
          f"{len(patients)} patients.")
    patients_lab_df = extract_events_within_hours(labevents, patients['hadm_id'], hours=hours)
    patients_lab_df = format_lab_events_for_note(patients_lab_df)
    return patients.merge(patients_lab_df, on='hadm_id', how='left')
