"""Hierarchy levels (R0 exact, R1, R2, R3, further) for every available run on the test codes.

Cut-off decided on the development set (fold 1): distances 1-2 are near misses, 3 is shown as "related",
4 and more are "further" (SNOMED steps are too uneven there to mean "close").

Writes runs/hierarchy/<run>.csv (per code) and runs/hierarchy/summary.csv; prints the summary.
  python hierarchy_report.py
"""
from pathlib import Path

import pandas as pd

from evaluate import candidates_to_predictions
from gold import load_gold_codes
from hierarchy import Hierarchy, relations_for_run
from rerank import candidates_file
from config import WORK_DIR

OUT = WORK_DIR / "runs/hierarchy"
MAIN = "large-3072__k30__low__rerank.v2_f2345"


def available_runs(test_codes) -> dict[str, dict[str, str]]:
    """name -> {source_code: predicted concept_id} for every run that exists."""
    runs = {}
    for route in ["et", "et_translate", "et_expand", "en", "en_expand"]:
        cands = pd.read_csv(candidates_file(route, "large-3072"), dtype={"source_code": str, "concept_id": str})
        cands = cands[cands.source_code.isin(test_codes)]
        runs[f"search only | {route}"] = {c: p[0] for c, p in candidates_to_predictions(cands, 1).items()}
    extra = {"RAG k10 | et_expand": "et_expand__large-3072__k10__low__rerank.v2_f2345",
             "RAG small-1536 | et_expand": "et_expand__small-1536__k30__low__rerank.v2_f2345"}
    names = {f"RAG | {r}": f"{r}__{MAIN}" for r in ["et", "et_translate", "et_expand", "en", "en_expand"]} | extra
    for name, stem in names.items():
        path = WORK_DIR / "runs/rerank" / f"{stem}.csv"
        if path.exists():
            run = pd.read_csv(path, dtype=str, keep_default_na=False)
            runs[name] = dict(zip(run.source_code, run.concept_id))
    for language in ["et", "en"]:
        path = WORK_DIR / "runs/zero_shot" / f"{language}__low.csv"
        if path.exists():
            run = pd.read_csv(path, dtype=str, keep_default_na=False)
            runs[f"no search | {language}"] = dict(zip(run.source_code, run.concept_id))
    return runs


def summarise(rel: pd.DataFrame) -> dict:
    n = len(rel)
    share = lambda mask: round(100 * mask.sum() / n, 1)
    near = rel.distance.isin([1, 2])
    return {
        "n": n,
        "R0 exact": share(rel.level == "R0"),
        "R1": share(rel.level == "R1"),
        "R2": share(rel.level == "R2"),
        "R3": share(rel.level == "R3"),
        "further/none": share(rel.level.isin(["further", "none"])),
        "within R1": share(rel.level.isin(["R0", "R1"])),
        "within R2": share(rel.level.isin(["R0", "R1", "R2"])),
        "near miss: more general": share(near & (rel.kind == "more_general")),
        "near miss: more specific": share(near & (rel.kind == "more_specific")),
        "near miss: lateral": share(near & (rel.kind == "lateral")),
    }


def main():
    gold = load_gold_codes()
    folds = pd.read_csv(WORK_DIR / "splits/folds.csv", dtype=str)
    test = gold[gold.source_code.isin(folds[folds.fold != "1"].source_code)].reset_index(drop=True)
    runs = available_runs(set(test.source_code))
    ids = set().union(*gold.index_targets) | {p for run in runs.values() for p in run.values() if p}
    hierarchy = Hierarchy.from_athena(ids)

    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for name, predictions in runs.items():
        rel = relations_for_run(predictions, test, hierarchy)
        rel.to_csv(OUT / (name.replace(" | ", "__").replace(" ", "_") + ".csv"), index=False)
        rows.append({"run": name, **summarise(rel)})
    summary = pd.DataFrame(rows).set_index("run")
    summary.to_csv(OUT / "summary.csv")
    pd.set_option("display.width", 250)
    print(summary.to_string())


if __name__ == "__main__":
    main()
