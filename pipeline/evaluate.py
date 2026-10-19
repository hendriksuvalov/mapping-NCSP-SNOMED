"""Score predictions against the gold mappings, following PROTOCOL.md.

Predictions: for each NCSP code, an ordered list of concept_ids (usually one). Codes without a
prediction count as empty predictions (misses). Every gold code is in the denominator, including the
unanswerable ones.

Per-code metrics:
  hit        any predicted concept is a gold target              (liberal accuracy)
  precision  |P & G| / |P|  (0 when P is empty)
  recall     |P & G| / |G|  (0 for unanswerable codes)
  f1         harmonic mean of precision and recall
  complete   every gold target is predicted                       (all targets required)
G is the code's gold targets that exist in the index.
"""
import numpy as np
import pandas as pd

METRICS = ["hit", "precision", "recall", "f1", "complete"]


def per_code_scores(predictions: dict, gold: pd.DataFrame) -> pd.DataFrame:
    unknown = set(predictions) - set(gold.source_code)
    if unknown:
        raise ValueError(f"predictions for codes not in the gold set: {sorted(unknown)[:5]}")
    rows = []
    for code, targets in zip(gold.source_code, gold.index_targets):
        predicted = list(dict.fromkeys(predictions.get(code, [])))  # drop duplicates, keep order
        overlap = len(set(predicted) & targets)
        precision = overlap / len(predicted) if predicted else 0.0
        recall = overlap / len(targets) if targets else 0.0
        rows.append({
            "source_code": code,
            "hit": float(overlap > 0),
            "precision": precision,
            "recall": recall,
            "f1": 2 * precision * recall / (precision + recall) if overlap else 0.0,
            "complete": float(bool(targets) and targets <= set(predicted)),
        })
    scores = pd.DataFrame(rows)
    return gold[["source_code", "chapter", "group", "mapping_type"]].merge(scores, on="source_code")


def cluster_bootstrap_ci(scores: pd.DataFrame, metric: str, n_boot: int = 2000, seed: int = 0):
    """95% CI of the mean of `metric`, resampling whole NCSP groups (first three characters)."""
    by_group = scores.groupby("group")[metric].agg(["sum", "count"])
    sums, counts = by_group["sum"].to_numpy(), by_group["count"].to_numpy()
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(sums), size=(n_boot, len(sums)))
    means = sums[draws].sum(axis=1) / counts[draws].sum(axis=1)
    return np.percentile(means, [2.5, 97.5])


def summarise(scores: pd.DataFrame, metrics=("hit", "recall", "complete"), ci: bool = True) -> pd.DataFrame:
    """Mean of each metric overall, per chapter and per mapping type (with n)."""
    slices = [("all", scores)]
    slices += [(f"chapter {c}", g) for c, g in scores.groupby("chapter")]
    slices += [(t, g) for t, g in scores.groupby("mapping_type")]
    rows = []
    for name, part in slices:
        row = {"slice": name, "n": len(part)}
        for m in metrics:
            row[m] = part[m].mean()
            if ci and name == "all":
                low, high = cluster_bootstrap_ci(part, m)
                row[f"{m}_ci"] = f"{low:.3f}-{high:.3f}"
        rows.append(row)
    return pd.DataFrame(rows).set_index("slice")


def paired_bootstrap(scores_a: pd.DataFrame, scores_b: pd.DataFrame, metric: str = "hit", n_boot: int = 2000,
                     seed: int = 0) -> dict:
    """Difference a − b in the mean of `metric` on the same codes, with a 95% CI from resampling NCSP groups
    jointly for both systems (respects the dependence between related codes, unlike McNemar's test)."""
    merged = scores_a[["source_code", "group", metric]].merge(
        scores_b[["source_code", metric]], on="source_code", suffixes=("_a", "_b"))
    merged["diff"] = merged[f"{metric}_a"] - merged[f"{metric}_b"]
    by_group = merged.groupby("group")["diff"].agg(["sum", "count"])
    sums, counts = by_group["sum"].to_numpy(), by_group["count"].to_numpy()
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(sums), size=(n_boot, len(sums)))
    diffs = sums[draws].sum(axis=1) / counts[draws].sum(axis=1)
    low, high = np.percentile(diffs, [2.5, 97.5])
    return {"diff": merged["diff"].mean(), "low": low, "high": high,
            "excludes_zero": bool(low > 0 or high < 0), "n_codes": len(merged), "n_groups": len(sums)}


def mcnemar(scores_a: pd.DataFrame, scores_b: pd.DataFrame, metric: str = "hit") -> dict:
    """Exact McNemar test on paired per-code binary outcomes."""
    from scipy.stats import binomtest

    merged = scores_a[["source_code", metric]].merge(scores_b[["source_code", metric]], on="source_code")
    a, b = merged[f"{metric}_x"].astype(bool), merged[f"{metric}_y"].astype(bool)
    only_a, only_b = int((a & ~b).sum()), int((~a & b).sum())
    p = binomtest(only_a, only_a + only_b, 0.5).pvalue if only_a + only_b else 1.0
    return {"only_a": only_a, "only_b": only_b, "p_value": p}


def candidates_to_predictions(candidates: pd.DataFrame, k: int) -> dict:
    """Top-k concept_ids per code from a long candidates table (source_code, rank, concept_id)."""
    top = candidates[candidates["rank"] <= k].sort_values(["source_code", "rank"])
    return top.groupby("source_code").concept_id.apply(list).to_dict()


def recall_at_k(candidates: pd.DataFrame, gold: pd.DataFrame, ks=(1, 5, 10, 20, 30, 50, 100, 200, 300)) -> pd.DataFrame:
    """Share of codes with at least one gold target among the top-k candidates (hit@k), overall and per slice."""
    rows = []
    for k in ks:
        scores = per_code_scores(candidates_to_predictions(candidates, k), gold)
        row = {"k": k, "all": scores.hit.mean()}
        row.update({f"{c}": g.hit.mean() for c, g in scores.groupby("chapter")})
        row.update({t: g.hit.mean() for t, g in scores.groupby("mapping_type") if t != "unanswerable"})
        rows.append(row)
    return pd.DataFrame(rows).set_index("k")
