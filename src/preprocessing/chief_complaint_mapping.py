import logging
import re

import pandas as pd


def extract_chief_complaint(note):
    """
    Extracts the Chief Complaint section from an admission or discharge note.
    Returns None if not found.
    """

    if not isinstance(note, str):
        return None

    # Corrected pattern:
    # - All alternatives include a capturing group
    # - Allows variable section titles (chief complaint, ___ complaint, chief ___)
    # - Stops before a section separator (no space: \n\n, one space: \n \n, one space with dot: \n.
    # \n or multiple spaces or tabs: \n \n) + next ALL CAPS header OR end of text
    pattern = (
        r"(?:"
        r"chief complaint\s*:|"
        r"___ complaint\s*:|"
        r"chief\s+___\s*:)"
        r"(.*?)(?="
        r"\n\s*\.?\s*\n"          # first blank(-ish) line after CC
        r"|"                      # OR
        r"\n[A-Z][A-Za-z ]+:"     # next line is a header (mixed case allowed)
        r"|\Z)"                   # OR end of note
    )

    match = re.search(pattern, note, re.DOTALL | re.IGNORECASE)

    if not match:
        return None

    extracted = match.group(1)
    if not extracted:
        return None

    return extracted.strip()


def _normalize_syn_list(syns):
    """Deduplicate & strip, keep order (longer first for greedy matching)."""
    uniq = []
    seen = set()
    for s in syns:
        s = s.strip().lower()
        if s and s not in seen:
            seen.add(s)
            uniq.append(s)
    # sort by length desc to prefer longest phrase first
    return sorted(uniq, key=len, reverse=True)


def _to_regex_phrase(s: str) -> str:
    """
    Turn a phrase into a safe regex with flexible whitespace.
    Example: 'shortness of breath' -> 'shortness\\s+of\\s+breath'
    """
    esc = re.escape(s)
    return re.sub(r'\\\s+', r'\\s+', esc)


def _compile_matchers(key_to_syns: dict):
    """
    Build a list of (compiled_regex, key) with word boundaries, case-insensitive.
    Longer phrases first to avoid partial overshadowing.
    """
    pairs = []
    for key, syns in key_to_syns.items():
        # include the key itself among phrases
        for phrase in _normalize_syn_list(syns + [key]):
            patt = rf'(?<!\w){_to_regex_phrase(phrase)}(?!\w)'
            pairs.append((phrase, patt, key))
    # sort by phrase length desc so longest substrings match first
    pairs.sort(key=lambda x: len(x[0]), reverse=True)
    compiled = [(re.compile(patt, flags=re.IGNORECASE), key) for _, patt, key in pairs]
    return compiled


def _all_key_matches(text: str, matchers) -> list[str]:
    """
    Return ALL keys whose patterns match the text, as a deduplicated list.
    (Keys are already lowercased in the matchers.)
    """
    if not isinstance(text, str):
        return []
    hits: list[str] = []
    for rgx, key in matchers:
        if rgx.search(text):
            if key not in hits:
                hits.append(key)
    return hits


def add_system_and_chief_complaints_to_notes(
        df: pd.DataFrame,
        cc_with_synonyms: dict[str, list[str]],
        cc_system_map: dict[str, str],
) -> pd.DataFrame:
    """
    Annotate rows with (cc_list, system_list) based on canonical CC keys & their synonyms.
       - If a synonym matches, store the KEY (not the synonym).
       - If multiple keys match a note, they are stored as a list in cc_list.

    Expected columns in df: 'cc_string', 'split'
    Returns: annotated_df
      - annotated_df: original rows + 'cc_list' and 'system_list' (lists of matches)
    """

    # basic column sanity check
    required_cols = {"cc_string", "split"}
    missing_cols = required_cols - set(df.columns)
    if missing_cols:
        raise ValueError(f"df is missing required columns: {sorted(missing_cols)}")

    logging.info("Map discharge note to following chief complaints and systems:")
    for cc, system in cc_system_map.items():
        print(f"  CC: '{cc}' -> System: '{system}'")

    # --- normalize keys (lowercase) ---
    cc_syn_norm = {k.lower(): v for k, v in cc_with_synonyms.items()}
    cc_system_map_norm = {k.lower(): v for k, v in cc_system_map.items()}

    # check consistency between synonym keys and system keys
    keys_syn = set(cc_syn_norm.keys())
    keys_sys = set(cc_system_map_norm.keys())

    missing_in_system_map = keys_syn - keys_sys
    missing_in_synonyms = keys_sys - keys_syn

    msg_parts = []
    if missing_in_system_map:
        msg_parts.append(
            f"keys in cc_with_synonyms but not in cc_system_map: {sorted(missing_in_system_map)}"
        )
    if missing_in_synonyms:
        msg_parts.append(
            f"keys in cc_system_map but not in cc_with_synonyms: {sorted(missing_in_synonyms)}"
        )
    if msg_parts:
        raise ValueError("Inconsistent keys between cc_with_synonyms and cc_system_map: "
                         + " | ".join(msg_parts))

    # compile regex matchers (keys are already lowercase)
    matchers = _compile_matchers(cc_syn_norm)

    # --- annotate with lists of CCs ---
    df_annot = df.copy()
    df_annot["cc_list"] = df_annot["cc_string"].apply(
        lambda s: _all_key_matches(s, matchers)
    )

    # keep only rows where at least one CC matched
    df_annot = df_annot[df_annot["cc_list"].map(bool)].copy()

    # map to system list
    df_annot["system_list"] = df_annot["cc_list"].apply(
        lambda keys: [cc_system_map_norm[k] for k in keys]
    )

    # --- build exploded view for summary ---
    df_exploded = df_annot.explode("cc_list").rename(columns={"cc_list": "cc"})
    df_exploded["system"] = df_exploded["cc"].map(cc_system_map_norm)

    # annotated_df keeps the list-columns (cc_list, system_list)
    return df_annot
