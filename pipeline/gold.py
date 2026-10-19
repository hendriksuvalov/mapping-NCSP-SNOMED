"""Load the expert NCSP -> SNOMED CT gold mappings (NCSP_AUG26 release)."""
import csv

import pandas as pd

from config import ATHENA_DIR, GOLD_DIR

GOLD_FILES = [
    GOLD_DIR / "NCSP_AUG26" / "source_to_concept_map_NCSP_L_M_chapters.csv",
    GOLD_DIR / "NCSP_AUG26" / "source_to_concept_map_NCSP_T_chapter.csv",
]


def index_concept_ids() -> set:
    """Concept ids eligible for the index: standard SNOMED concepts of class Procedure (any OMOP domain)."""
    concepts = pd.read_csv(
        ATHENA_DIR / "CONCEPT.csv", sep="\t", dtype=str, keep_default_na=False, quoting=csv.QUOTE_NONE,
        usecols=["concept_id", "vocabulary_id", "concept_class_id", "standard_concept"],
    )
    mask = (
        (concepts.vocabulary_id == "SNOMED")
        & (concepts.concept_class_id == "Procedure")
        & (concepts.standard_concept == "S")
    )
    return set(concepts.loc[mask, "concept_id"])


def load_gold_rows() -> pd.DataFrame:
    """One row per (NCSP code, target concept); unmapped codes (target_concept_id 0) are dropped."""
    rows = pd.concat(
        [pd.read_csv(f, dtype=str, keep_default_na=False) for f in GOLD_FILES], ignore_index=True
    )
    rows = rows[rows.target_concept_id != "0"].copy()
    rows["in_index"] = rows.target_concept_id.isin(index_concept_ids())
    return rows


def load_gold_codes(rows: pd.DataFrame | None = None) -> pd.DataFrame:
    """One row per NCSP code with its target sets and mapping type (1:1, 1:n, unanswerable)."""
    rows = load_gold_rows() if rows is None else rows
    codes = rows.groupby("source_code").agg(
        description_et=("source_code_description", "first"),
        all_targets=("target_concept_id", lambda s: set(s)),
        index_targets=("target_concept_id", lambda s: set(s[rows.loc[s.index, "in_index"]])),
    ).reset_index()
    codes["chapter"] = codes.source_code.str[0]
    codes["group"] = codes.source_code.str[:3]
    codes["n_index_targets"] = codes.index_targets.map(len)
    codes["mapping_type"] = codes.n_index_targets.map(
        lambda n: "unanswerable" if n == 0 else ("1:1" if n == 1 else "1:n")
    )
    return codes
