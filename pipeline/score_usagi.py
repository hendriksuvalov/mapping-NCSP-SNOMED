"""Score a Usagi mapping file (saved from Usagi, "usagi.csv" format) on the test codes, like the other systems.

Usagi's automatically assigned top suggestion (column conceptId) is the answer; nothing needs approving in Usagi.
  python score_usagi.py usagi/usagi_output_en.csv usagi/usagi_output_et_translated.csv
At the end it prints the values in the form used by the manuscript source (LaTeX macros).
"""
import argparse
from pathlib import Path

import pandas as pd

from evaluate import paired_bootstrap, per_code_scores, summarise
from gold import load_gold_codes
from hierarchy import Hierarchy, relations_for_run
from config import WORK_DIR


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("files", nargs="+", type=Path)
    args = parser.parse_args()

    gold = load_gold_codes()
    folds = pd.read_csv(WORK_DIR / "splits/folds.csv", dtype=str)
    test = gold[gold.source_code.isin(folds[folds.fold != "1"].source_code)].reset_index(drop=True)

    macros = []
    runs = {}
    for path in args.files:
        usagi = pd.read_csv(path, dtype=str, keep_default_na=False)
        if "targetConceptId" in usagi.columns and "conceptId" not in usagi.columns:  # Usagi's export format
            usagi = usagi.rename(columns={"targetConceptId": "conceptId"})
            classes = usagi.targetConceptClassId.value_counts()
            print(f"\n{path.name}: concept classes of the suggestions: {classes.to_dict()}")
        else:  # Usagi's "Save as" format has no concept class: look it up in the vocabulary
            classes = concept_classes(usagi.conceptId).value_counts()
            print(f"\n{path.name}: concept classes of the suggestions: {classes.to_dict()}")
        if "mappingStatus" in usagi.columns:
            print(f"  mapping status: {usagi.mappingStatus.value_counts().to_dict()} (UNCHECKED = automatic suggestion)")
        missing = {"sourceCode", "conceptId"} - set(usagi.columns)
        if missing:
            raise SystemExit(f"{path}: missing columns {missing}; save the file from Usagi in its own format")
        usagi = usagi[usagi.sourceCode.isin(test.source_code) & usagi.conceptId.str.isdigit() & (usagi.conceptId != "0")]
        runs[path.stem] = usagi.drop_duplicates("sourceCode").set_index("sourceCode").conceptId.to_dict()

    ids = set().union(*gold.index_targets) | {c for run in runs.values() for c in run.values()}
    hierarchy = Hierarchy.from_athena(ids)
    for name, predictions in runs.items():
        scores = per_code_scores({c: [p] for c, p in predictions.items()}, test)
        summary = summarise(scores)
        levels = relations_for_run(predictions, test, hierarchy).level
        print(f"\n{name}: {len(predictions)} of {len(test)} test codes have a Usagi suggestion")
        print(f"  exact {100 * summary.loc['all', 'hit']:.1f}% [{summary.loc['all', 'hit_ci']}]"
              f" | within R1 {100 * levels.isin(['R0', 'R1']).mean():.1f}%"
              f" | within R2 {100 * levels.isin(['R0', 'R1', 'R2']).mean():.1f}%")
        print(summary[["n", "hit"]].round(3).to_string())
        prefix = latex_prefix(name)
        if prefix:
            low, high = (100 * float(x) for x in summary.loc["all", "hit_ci"].split("-"))
            macros += [(f"{prefix}", f"{100 * summary.loc['all', 'hit']:.1f}"),
                       (f"{prefix}CI", f"{low:.1f}--{high:.1f}"),
                       (f"{prefix}Rone", f"{100 * levels.isin(['R0', 'R1']).mean():.1f}")]
            diff = paired_vs_pipeline(scores)
            print(f"  expanded-Estonian pipeline minus this run: {diff[0][1]} points [95% CI {diff[1][1]}]")
            if prefix == "usagiGoogle":
                macros += diff
    if macros:
        print("\n% ---- values for the manuscript ----")
        for macro, value in macros:
            print(f"\\renewcommand{{\\{macro}}}{{{value}}}")


def concept_classes(concept_ids: pd.Series) -> pd.Series:
    import csv
    from gold import ATHENA_DIR
    wanted = set(concept_ids)
    classes = {}
    for chunk in pd.read_csv(ATHENA_DIR / "CONCEPT.csv", sep="\t", dtype=str, keep_default_na=False,
                             quoting=csv.QUOTE_NONE, usecols=["concept_id", "concept_class_id"], chunksize=500_000):
        hit = chunk[chunk.concept_id.isin(wanted)]
        classes.update(zip(hit.concept_id, hit.concept_class_id))
    return concept_ids.map(classes).fillna("not in vocabulary")


def latex_prefix(name):
    for key, prefix in [("google", "usagiGoogle"), ("et_translated", "usagiLLM"), ("_en", "usagiEnglish")]:
        if key in name:
            return prefix
    return None


def paired_vs_pipeline(usagi_scores):
    """Primary comparison: pipeline with expanded Estonian input (per-code mean over three runs) minus Usagi."""
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "analysis"))
    from results import collect
    pipeline = collect()["repeats"]["et"]["mean_scores"]
    p = paired_bootstrap(pipeline, usagi_scores)
    return [("usagiDiff", f"{100 * p['diff']:+.1f}"),
            ("usagiDiffCI", f"{100 * p['low']:+.1f} to {100 * p['high']:+.1f}")]


if __name__ == "__main__":
    main()
