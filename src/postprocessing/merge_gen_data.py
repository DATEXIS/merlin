import gc
import json
import pandas as pd
from pathlib import Path

# Mirrors `frequent_chief_complaints` / `max_rank` in
# scripts/run_preprocesssing.py::filter_by_rank. That filter is only ever
# reached for the *first* chief complaint in the loop (a stray `break` at
# run_preprocesssing.py:80 exits after 'abdominal pain'), so every other
# complaint's cc_*.pq -- including headache -- was generated from unranked
# data (match_category -1 "unmatched" and 3 "other" included). Applying the
# same filter here, before the cross-model / single-cc-per-subject
# assignment, corrects the already-generated data without re-running
# generation. Keep this set in sync with run_preprocesssing.py.
FREQUENT_CHIEF_COMPLAINTS = {"abdominal pain", "dyspnea"}


def _max_rank_for_cc(cc_stem: str) -> int:
    """cc_stem is a cc_*.pq file stem, e.g. 'cc_headache' or 'cc_abdominal_pain'."""
    chief_complaint = cc_stem.removeprefix("cc_").replace("_", " ")
    return 1 if chief_complaint in FREQUENT_CHIEF_COMPLAINTS else 2


def _read_cc_file(path: Path, columns: list[str] | None = None, verbose: bool = True) -> pd.DataFrame:
    """
    Read one cc_*.pq file and apply the match_category rank filter:
    keep 0 <= match_category <= max_rank (drops -1 unmatched and anything
    above max_rank). Must run before any cross-model / single-cc-per-subject
    assignment, since that assignment is resolved from whatever ids survive
    here.
    """
    read_cols = sorted(set(columns) | {"match_category"}) if columns is not None else None
    df = pd.read_parquet(path, columns=read_cols)
    max_rank = _max_rank_for_cc(path.stem)
    before = len(df)
    df = df[(df["match_category"] >= 0) & (df["match_category"] <= max_rank)]
    removed = before - len(df)
    if removed and verbose:
        print(f"    {path.stem}: rank-filtered {before:,} -> {len(df):,} "
              f"({removed:,} dropped, match_category outside [0, {max_rank}])")
    if columns is not None:
        df = df[columns]
    return df


def _load_id_index(model_dir: Path) -> dict[str, set[tuple]]:
    """Read only the id columns of every cc_*.pq file for one model, after
    the match_category rank filter.
    Returns {cc_name: {(subject_id, hadm_id), ...}}. Cheap: avoids loading
    the (large) text/embedding columns just to figure out membership."""
    files = sorted(model_dir.glob("cc_*.pq"))
    if not files:
        raise FileNotFoundError(f"No cc_*.pq files found in {model_dir}")
    idx = {}
    for path in files:
        cc = path.stem  # e.g. "cc_back_pain"
        df = _read_cc_file(path, columns=["subject_id", "hadm_id"], verbose=False)
        idx[cc] = set(map(tuple, df.itertuples(index=False, name=None)))
    return idx


def resolve_cross_model_assignment(
    model_dirs: list[Path],
) -> tuple[dict[tuple, str], dict[str, int], int]:
    """
    Decide, once, which single chief complaint each (subject_id, hadm_id)
    belongs to across ALL models combined.

    A hadm_id can appear in multiple cc_*.pq files (within one model, or with
    a different set of ccs per model). We only keep hadm_ids that have at
    least one cc for which *every* model has a generated row — otherwise a
    model would silently be missing that patient later on. Among the ccs that
    satisfy this, we pick the lowest-resource one (using the cross-model
    intersection size, not any single model's count, so every model makes
    the identical choice for a given hadm_id).

    Returns (assignment, global_size):
      assignment  : {(subject_id, hadm_id): chosen_cc}, only for ids kept
      global_size : {cc: # ids present in every model for that cc} — the
                    resource measure used for the low-resource tie-break
      n_total_ids : total distinct (subject_id, hadm_id) seen anywhere,
                    for reporting how many were dropped
    """
    per_model_idx = {d.name: _load_id_index(d) for d in model_dirs}
    models = list(per_model_idx.keys())
    all_ccs = sorted({cc for idx in per_model_idx.values() for cc in idx})

    # Cross-model resource size per cc: how many ids exist in EVERY model's
    # raw file for that cc. This is the true achievable ceiling for that cc,
    # not just one model's (possibly larger) raw count.
    global_size = {}
    for cc in all_ccs:
        sets = [per_model_idx[m].get(cc, set()) for m in models]
        global_size[cc] = len(set.intersection(*sets)) if all(sets) else 0

    # id -> {cc: {models that generated this id under this cc}}
    id_cc_models: dict[tuple, dict[str, set[str]]] = {}
    for m, idx in per_model_idx.items():
        for cc, ids in idx.items():
            for pid in ids:
                id_cc_models.setdefault(pid, {}).setdefault(cc, set()).add(m)

    assignment: dict[tuple, str] = {}
    for pid, cc_models in id_cc_models.items():
        valid = [cc for cc, mset in cc_models.items() if len(mset) == len(models)]
        if not valid:
            continue  # no cc has this id in ALL models -> drop entirely
        assignment[pid] = min(valid, key=lambda c: global_size[c])

    return assignment, global_size, len(id_cc_models)


def build_complaint_map(assignment: dict[tuple, str]) -> dict[str, list]:
    """
    Build {chief_complaint: [subject_id, ...]} from the cross-model cc
    assignment resolved once in resolve_cross_model_assignment().

    This is used by scripts/run_preprocesssing.py's "Rare Case Filter"
    (`filtered_notes['subject_id'].isin(subject_complaint_map[cc])`) to
    reproduce, on a later preprocessing run, the same cohort of subjects
    that ended up in the generated dataset for each chief complaint.

    Keyed by subject_id (not hadm_id) to match that existing filter. An
    earlier, manually-maintained version of this file was built by grouping
    on hadm_id instead, which silently zeroed out the Rare Case Filter for
    every complaint since subject_id and hadm_id ranges never overlap in
    MIMIC-IV. Building it here from `assignment` keeps it in sync with
    whatever actually got merged/deduped, instead of a separate ad-hoc script.
    """
    cc_to_subjects: dict[str, set] = {}
    for (subject_id, hadm_id), cc in assignment.items():
        cc_name = cc.removeprefix("cc_")  # "cc_back_pain" -> "back_pain"
        # ids come from pandas (numpy int64) via _load_id_index; cast to
        # plain int so the map is JSON-serializable.
        cc_to_subjects.setdefault(cc_name, set()).add(int(subject_id))
    return {cc: sorted(ids) for cc, ids in sorted(cc_to_subjects.items())}


def merge_model(
    model_dir: Path,
    assignment: dict[tuple, str],
) -> tuple[pd.DataFrame, dict, dict]:
    """
    Load all cc_*.pq files for one model, apply the match_category rank
    filter (see `_read_cc_file`), and keep only the rows selected by
    `assignment` (built once, across all models, by
    resolve_cross_model_assignment -- itself computed from rank-filtered
    ids, via `_load_id_index`). This guarantees:
      - each (subject_id, hadm_id) is used once for this model
      - the cc it lands under is identical for every model
      - the id is guaranteed to exist under that cc for every model
      - match_category is within [0, max_rank] for that cc

    Returns (merged_df, before_sizes, after_sizes) where the size dicts
    map cc_name -> count. `before_sizes` is the count *after* the rank
    filter but before the single-cc-per-subject assignment.
    """
    files = sorted(model_dir.glob("cc_*.pq"))
    if not files:
        raise FileNotFoundError(f"No cc_*.pq files found in {model_dir}")

    frames = []
    cc_sizes = {}
    for path in files:
        cc = path.stem  # e.g. "cc_back_pain"
        df = _read_cc_file(path)
        cc_sizes[cc] = len(df)
        ids = list(map(tuple, df[["subject_id", "hadm_id"]].itertuples(index=False, name=None)))
        keep = [assignment.get(pid) == cc for pid in ids]
        df = df.loc[keep].copy()
        df["_cc"] = cc
        frames.append(df)

    combined = pd.concat(frames, ignore_index=True)
    after_sizes = combined.groupby("_cc").size().to_dict()

    # Print before/after table
    all_ccs = sorted(cc_sizes.keys())
    col_w = max(len(cc) for cc in all_ccs)
    num_w = 8
    sep = f"  +-{'-' * col_w}-+-{'-' * num_w}-+-{'-' * num_w}-+-{'-' * num_w}-+"
    print(sep)
    print(f"  | {'Chief Complaint':<{col_w}} | {'Before':>{num_w}} | {'After':>{num_w}} | {'Removed':>{num_w}} |")
    print(sep)
    total_before = total_after = 0
    for cc in all_ccs:
        before = cc_sizes[cc]
        after = after_sizes.get(cc, 0)
        removed = before - after
        total_before += before
        total_after += after
        print(f"  | {cc:<{col_w}} | {before:>{num_w}} | {after:>{num_w}} | {removed:>{num_w}} |")
    print(sep)
    print(f"  | {'TOTAL':<{col_w}} | {total_before:>{num_w}} | {total_after:>{num_w}} | {total_before - total_after:>{num_w}} |")
    print(sep)

    return combined, cc_sizes, after_sizes


def _print_cross_model_table(
    summary: dict[str, tuple[dict, dict]],  # model -> (before, after)
) -> None:
    """Print a cross-model comparison: rows = CCs, columns = models (before → after)."""
    all_ccs = sorted({cc for b, _ in summary.values() for cc in b})
    models = sorted(summary.keys())

    cc_w = max(len(cc) for cc in all_ccs)
    # each model gets a "before→after" column
    col_w = max(max(len(m) for m in models), 13)  # min width "XXXXX→XXXXX"

    def col(m: str) -> str:
        return m[:col_w]

    sep_mid = "-+-".join(["-" * cc_w] + ["-" * col_w] * len(models))
    sep = f"+-{sep_mid}-+"

    header_cells = [f"{col(m):^{col_w}}" for m in models]
    print("\n── Cross-model distribution (before → after dedup)\n")
    print(sep)
    print("| " + f"{'Chief Complaint':<{cc_w}}" + " | " + " | ".join(header_cells) + " |")
    print(sep)
    for cc in all_ccs:
        cells = []
        for m in models:
            before, after = summary[m]
            b = before.get(cc, 0)
            a = after.get(cc, 0)
            cell = f"{b}→{a}"
            cells.append(f"{cell:^{col_w}}")
        print("| " + f"{cc:<{cc_w}}" + " | " + " | ".join(cells) + " |")
    print(sep)
    # totals row
    total_cells = []
    for m in models:
        before, after = summary[m]
        b = sum(before.values())
        a = sum(after.values())
        cell = f"{b}→{a}"
        total_cells.append(f"{cell:^{col_w}}")
    print("| " + f"{'TOTAL':<{cc_w}}" + " | " + " | ".join(total_cells) + " |")
    print(sep)


def merge_all(
        gen_data_dir: str,
        output_dir: str | None = None,
        complaint_map_path: str | None = "data/preprocessed_mimic/subject_complaint_map.json",
) -> None:
    """
    Merge cc files for every model folder found under gen_data_dir.
    Saves one <model>.pq per model in output_dir (defaults to gen_data_dir).

    The cc assignment (which hadm_ids are kept, and which single cc they're
    filed under) is resolved once, across all models, before any model's
    full data is loaded. This guarantees every hadm_id that survives is
    present under the *same* cc for every model.

    Also (re)builds subject_complaint_map.json from that same assignment, so
    it always reflects exactly what was merged here. Pass
    complaint_map_path=None to skip writing it.
    """
    root = Path(gen_data_dir)
    out_root = Path(output_dir) if output_dir else root

    model_dirs = sorted(d for d in root.iterdir() if d.is_dir())
    if not model_dirs:
        print("No model directories found.")
        return

    print("── Resolving cross-model cc assignment (id columns only)")
    assignment, global_size, n_total_ids = resolve_cross_model_assignment(model_dirs)
    print(f"  {len(assignment):,} hadm_ids kept (have a cc present in ALL models), "
          f"{n_total_ids - len(assignment):,} dropped (no cc common to all models)")

    if complaint_map_path:
        complaint_map = build_complaint_map(assignment)
        out_map_path = Path(complaint_map_path)
        out_map_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_map_path, "w") as f:
            json.dump(complaint_map, f)
        print(f"\n  Saved subject_complaint_map -> {out_map_path}")
        for cc, ids in complaint_map.items():
            print(f"    {cc}: {len(ids)} subjects")

    summary: dict[str, tuple[dict, dict]] = {}

    for model_dir in model_dirs:
        model = model_dir.name
        print(f"\n── {model}")
        try:
            merged, before, after = merge_model(model_dir, assignment)
        except FileNotFoundError as e:
            print(f"  [skipping: {e}]")
            continue

        summary[model] = (before, after)
        out_path = out_root / f"{model}.pq"
        merged.to_parquet(out_path, index=False)
        print(f"  saved: {out_path}")
        del merged
        gc.collect()

    if len(summary) > 1:
        _print_cross_model_table(summary)

    print("\nDone.")


COMBINED_COLUMNS = [
    "subject_id", "hadm_id", "ICD_CODES", "admission_note", "discharge_note",
    "split", "diagnose_string", "Chief Complaint", "diseases",
    "diseases_cossim_scores", "disease", "disease_vector", "match_category",
    "labs",
    "v1_score", "v1_json", "v1_text", "v1_preds",
    "v2_score", "v2_json", "v2_text", "v2_preds",
    "v3_score", "v3_json", "v3_text", "v3_preds",
    "v4_score", "v4_json", "v4_text", "v4_preds",
    "gen_model",
]


_VALID_SPLITS = {"train": "train", "dev": "dev", "val": "dev", "test": "test"}


def combine_models(gen_data_dir: str, output_path: str) -> None:
    """
    Concatenate all per-model .pq files into a single combined parquet.

    Adds a 'gen_model' column with the model name and keeps only the
    columns defined in COMBINED_COLUMNS.
    """
    root = Path(gen_data_dir)
    out_path = Path(output_path).resolve()

    # combine_models writes its output inside gen_data_dir (by design, so
    # downstream tooling finds it there) — exclude that path itself from the
    # input glob, otherwise a rerun re-ingests its own previous output as a
    # phantom 4th "model" (with whatever stale data was in it at the time).
    model_files = sorted(p for p in root.glob("*.pq") if p.resolve() != out_path)
    if not model_files:
        raise FileNotFoundError(f"No .pq files found in {root}")

    frames = []
    for path in model_files:
        df = pd.read_parquet(path)
        df["gen_model"] = path.stem
        # Normalize split naming — 'val' and 'dev' refer to the same split;
        # 'dev' is canonical everywhere else in the pipeline.
        if "split" in df.columns:
            unknown = set(df["split"].unique()) - set(_VALID_SPLITS)
            if unknown:
                print(f"  [WARNING] {path.stem}: unrecognized split value(s) {unknown}, left as-is")
            df["split"] = df["split"].map(lambda s: _VALID_SPLITS.get(s, s))
        # Keep only columns that exist (guard against optional ones)
        cols = [c for c in COMBINED_COLUMNS if c in df.columns]
        frames.append(df[cols])
        print(f"  {path.stem}: {len(df):,} rows")

    combined = pd.concat(frames, ignore_index=True)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(output_path, index=False)
    print(f"  combined: {len(combined):,} rows → {output_path}")
