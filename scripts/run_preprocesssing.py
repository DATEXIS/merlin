import ast
import json
from pathlib import Path

import pandas as pd

from src.exp_args import ExpArgs
from src.preprocessing.chief_complaint_mapping import add_system_and_chief_complaints_to_notes
from src.preprocessing.chief_complaint_mapping import extract_chief_complaint
from src.preprocessing.disease_mapping import add_primary_diagnosis_column, add_wikidoc_columns
from src.preprocessing.lab_extraction import add_lab_data_to_notes
from src.preprocessing.note_ranking import add_rank_columns
from src.preprocessing.utils import load_mimic_notes, filter_mimic_by_meta_data
from src.utils import load_diseases_for_chief_complaint


def preprocessing():
    exp_args = ExpArgs()
    notes = load_mimic_notes(exp_args)
    notes = add_lab_data_to_notes(notes, hours=12)

    # Add system and chief complaints to note
    print("Adding chief complaints and systems to notes...")
    cc_df = pd.read_csv('data/medical_schemes/cc_synonyms_systems.csv')
    cc_with_synonyms = dict(zip(cc_df['chief_complaint'], cc_df['synonyms'].map(ast.literal_eval)))
    cc_system_map = dict(zip(cc_df['chief_complaint'], cc_df['systems']))

    notes['cc_string'] = notes['discharge_note'].apply(extract_chief_complaint)
    annotated_df = add_system_and_chief_complaints_to_notes(
        df=notes,
        cc_with_synonyms=cc_with_synonyms,
        cc_system_map=cc_system_map
    )

    # Extract primary diagnosis string from discharge note and add as primary_diagnosis column
    print("Adding primary diagnosis column to notes...")
    add_primary_diagnosis_column(annotated_df)

    annotated_df.to_parquet("data/preprocessed_mimic/annotated_mimic_notes.pq", index=False)
    return annotated_df


def filter_by_rank():
    exp_args = ExpArgs()
    annotated_df = pd.read_parquet("data/preprocessed_mimic/annotated_mimic_notes.pq")

    # The seven most frequent chief complaints by admission count (paper Table 3).
    # data/medical_schemes/ also carries a vertigo schema; it falls outside this
    # top-seven cut and is not part of the dataset.
    chief_complaints = ['abdominal pain', 'back pain', 'chest pain', 'cough', 'diarrhea', 'dyspnea', 'headache']
    frequent_chief_complaints = {'abdominal pain', 'dyspnea'}
    with open('data/preprocessed_mimic/subject_complaint_map.json', 'r') as f:
        subject_complaint_map = json.load(f)

    for chief_complaint in chief_complaints:
        filtered_notes = filter_mimic_by_meta_data(
            df=annotated_df,
            icd_version=10,
            data_source='hosp',
            chief_complaint=chief_complaint
        )
        print(f"Filtered {len(filtered_notes)} notes for chief complaint: {chief_complaint}")
        filtered_notes['Chief Complaint'] = chief_complaint
        # Add wikidoc disease and disease_vector columns based on primary diagnoses to DataFrame.
        wikidoc_df = load_diseases_for_chief_complaint(exp_args, chief_complaint)
        add_wikidoc_columns(filtered_notes, wikidoc_df)

        add_rank_columns(filtered_notes, wikidoc_df, False)
        file_name = f"patients_{chief_complaint}.pq"
        filtered_notes.to_parquet(f"data/preprocessed_mimic/{file_name}_unfiltered", index=False)

        max_rank = 1 if chief_complaint in frequent_chief_complaints else 2
        filtered_notes = filtered_notes[(filtered_notes['match_category'] <= max_rank)
                                        & (filtered_notes['match_category'] >= 0)]
        print(f"Rank Filtered to {len(filtered_notes)}")

        chief_complaint = chief_complaint.replace(' ', '_')
        mask = filtered_notes['subject_id'].isin(subject_complaint_map[chief_complaint])
        filtered_notes = filtered_notes[mask]
        print(f"Rare Case Filtered to {len(filtered_notes)}")

        filtered_notes.to_parquet(f"data/preprocessed_mimic/{file_name}", index=False)


if __name__ == "__main__":
    # Stage 1 is the expensive one (loads and annotates the full MIMIC-IV note
    # set); it is skipped when its output is already on disk. Delete
    # annotated_mimic_notes.pq to force a rebuild.
    if not Path("data/preprocessed_mimic/annotated_mimic_notes.pq").exists():
        preprocessing()
    filter_by_rank()