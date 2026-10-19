"""Assign every gold NCSP code to one of K cross-validation folds.

Codes sharing the first three NCSP characters (e.g. all LBF*) stay in the same fold, so sibling
codes never sit on both sides of a split. Folds are stratified by chapter x mapping type.

  python make_folds.py            # writes splits/folds.csv
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from gold import load_gold_codes
from config import WORK_DIR

SEED = 20260925


def balanced_group_folds(codes: pd.DataFrame, k: int, seed: int) -> pd.Series:
    """Greedy stratified group assignment: place the largest groups first, each into the fold that
    keeps per-stratum and total counts closest to an even 1/k share. Ties are broken by a seeded shuffle."""
    rng = np.random.default_rng(seed)
    strata = codes.chapter + "|" + codes.mapping_type
    counts = pd.crosstab(codes.group, strata)                     # group x stratum
    counts = counts.loc[rng.permutation(counts.index)]            # seeded tie-break order
    counts = counts.loc[counts.sum(axis=1).sort_values(ascending=False, kind="stable").index]
    target = counts.sum() / k
    target_total = counts.values.sum() / k
    load = pd.DataFrame(0, index=range(1, k + 1), columns=counts.columns)
    assignment = {}
    def imbalance(v):
        return (((v - target) / target.clip(lower=1)) ** 2).sum() + ((v.sum() - target_total) / target_total) ** 2

    for group, row in counts.iterrows():
        # Choose the fold whose imbalance grows least when this group is added.
        best = min(load.index, key=lambda f: (imbalance(load.loc[f] + row) - imbalance(load.loc[f]), rng.random()))
        load.loc[best] += row
        assignment[group] = best
    return codes.group.map(assignment)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--out", type=Path, default=WORK_DIR / "splits/folds.csv")
    args = parser.parse_args()

    codes = load_gold_codes().sort_values("source_code").reset_index(drop=True)
    codes["fold"] = balanced_group_folds(codes, args.k, SEED)
    assert codes.groupby("group").fold.nunique().eq(1).all(), "a group was split across folds"

    args.out.parent.mkdir(parents=True, exist_ok=True)
    codes[["source_code", "chapter", "group", "mapping_type", "n_index_targets", "fold"]].to_csv(args.out, index=False)

    summary = codes.pivot_table(index="fold", columns=["chapter", "mapping_type"], values="source_code",
                                aggfunc="count", fill_value=0)
    summary["total"] = summary.sum(axis=1)
    summary["groups"] = codes.groupby("fold").group.nunique()
    print(summary.to_string())
    print(f"\nwrote {len(codes)} codes to {args.out} (seed {SEED})")


if __name__ == "__main__":
    main()
