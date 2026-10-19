"""How far a predicted SNOMED CT concept is from the correct one in the is-a hierarchy.

Distance = steps from the prediction up to the nearest common ancestor + steps from the correct concept up to it
(shortest over all common ancestors; SNOMED concepts can have several parents). Uses OMOP's CONCEPT_ANCESTOR,
which stores the minimum number of levels between every standard concept and each of its ancestors.

  kind           meaning
  exact          the prediction is a correct concept                         (R0)
  more_general   the prediction is an ancestor of the correct concept
  more_specific  the prediction is a descendant of the correct concept
  lateral        neither: related through a common ancestor (e.g. a sibling, 1 up + 1 down = R2)
  unrelated      no common ancestor within the concepts' ancestor sets
  not_in_hierarchy / no_prediction
For codes with several correct concepts, the closest one counts.
"""
import csv
from pathlib import Path

import pandas as pd

from config import ATHENA_DIR


class Hierarchy:
    def __init__(self, ancestors: dict[str, dict[str, int]]):
        self.ancestors = ancestors  # concept_id -> {ancestor_id: min levels of separation}, incl. itself at 0

    @classmethod
    def from_athena(cls, concept_ids) -> "Hierarchy":
        wanted = {int(c) for c in concept_ids if str(c).isdigit()}
        table = pd.read_csv(ATHENA_DIR / "CONCEPT_ANCESTOR.csv", sep="\t", quoting=csv.QUOTE_NONE,
                            usecols=["ancestor_concept_id", "descendant_concept_id", "min_levels_of_separation"],
                            dtype="int64")
        table = table[table.descendant_concept_id.isin(wanted)]
        return cls.from_pairs(table.descendant_concept_id, table.ancestor_concept_id, table.min_levels_of_separation)

    @classmethod
    def from_pairs(cls, descendants, ancestors, levels) -> "Hierarchy":
        result: dict[str, dict[str, int]] = {}
        for d, a, n in zip(descendants, ancestors, levels):
            result.setdefault(str(d), {})[str(a)] = int(n)
        return cls(result)

    def relation(self, predicted: str | None, targets: set[str]) -> dict:
        if not predicted:
            return {"kind": "no_prediction", "distance": None, "target": None, "up_pred": None, "up_gold": None}
        if predicted in targets:
            return {"kind": "exact", "distance": 0, "target": predicted, "up_pred": 0, "up_gold": 0}
        pred_anc = self.ancestors.get(predicted)
        if not pred_anc:
            return {"kind": "not_in_hierarchy", "distance": None, "target": None, "up_pred": None, "up_gold": None}
        best = None
        for target in sorted(targets):
            gold_anc = self.ancestors.get(target, {})
            for common in pred_anc.keys() & gold_anc.keys():
                up_pred, up_gold = pred_anc[common], gold_anc[common]
                key = (up_pred + up_gold, max(up_pred, up_gold))
                if best is None or key < best[0]:
                    best = (key, target, up_pred, up_gold)
        if best is None:
            return {"kind": "unrelated", "distance": None, "target": None, "up_pred": None, "up_gold": None}
        (distance, _), target, up_pred, up_gold = best
        kind = "more_general" if up_pred == 0 else "more_specific" if up_gold == 0 else "lateral"
        return {"kind": kind, "distance": distance, "target": target, "up_pred": up_pred, "up_gold": up_gold}


def level_label(relation: dict, max_level: int = 3) -> str:
    """R0, R1, R2, R3 or 'further' (distance > max_level, unrelated, or not in the hierarchy), or 'none'."""
    if relation["kind"] == "no_prediction":
        return "none"
    d = relation["distance"]
    if d is None or d > max_level:
        return "further"
    return f"R{d}"


def relations_for_run(predictions: dict[str, str], gold: pd.DataFrame, hierarchy: Hierarchy) -> pd.DataFrame:
    rows = []
    for code, targets, chapter, mapping_type in zip(gold.source_code, gold.index_targets, gold.chapter, gold.mapping_type):
        r = hierarchy.relation(predictions.get(code) or None, targets)
        rows.append({"source_code": code, "chapter": chapter, "mapping_type": mapping_type,
                     "predicted": predictions.get(code, ""), **r, "level": level_label(r)})
    return pd.DataFrame(rows)
