"""Retrieve the top-k SNOMED candidates for every query route and embedding variant.

Query embeddings are cached per (route, model) at full size, so dimension variants reuse them.
Codes with several query rows are scored per concept by their best-matching row (max pooling).

Output: runs/retrieval/<route>__<model>-<dims>.csv with source_code, rank, concept_id, score,
plus a .json manifest next to it.

  python retrieve.py --route en et --variants small-1536 large-1024 large-3072
"""
import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from config import EMBEDDING_DEPLOYMENTS, WORK_DIR, azure_client
from index_io import embed_texts, load_index, normalise

OUT_DIR = WORK_DIR / "runs/retrieval"
CACHE_DIR = WORK_DIR / "queries/cache"


def query_embeddings(route: str, model: str, queries: pd.DataFrame) -> np.ndarray:
    digest = hashlib.sha256("\n".join(queries.source_code + "\t" + queries.text).encode()).hexdigest()[:12]
    cache = CACHE_DIR / f"{route}__{model}__{digest}.npy"
    if cache.exists():
        return np.load(cache)
    cache.parent.mkdir(parents=True, exist_ok=True)
    vectors = embed_texts(azure_client(), EMBEDDING_DEPLOYMENTS[model], queries.text.tolist())
    np.save(cache, vectors)
    return vectors


POOLING = {
    # How several query rows of one code are combined into one score per concept.
    "max": lambda s: s.max(axis=0),     # best-matching row
    "mean": lambda s: s.mean(axis=0),   # average similarity over rows
}


def retrieve(route: str, model: str, dims: int, k: int, index_cache: dict, pooling: str = "max") -> Path:
    queries = pd.read_csv(WORK_DIR / "queries" / f"{route}.csv", dtype=str, keep_default_na=False)
    if (queries.text.str.strip() == "").any():
        raise SystemExit(f"{route}: empty query texts")
    if model not in index_cache:
        index_cache[model] = load_index(WORK_DIR / "index" / model)
    concepts, full_matrix, index_manifest = index_cache[model]
    matrix = normalise(full_matrix[:, :dims]) if dims < full_matrix.shape[1] else full_matrix
    vectors = query_embeddings(route, model, queries)
    vectors = normalise(vectors[:, :dims])

    rows = []
    for code, idx in queries.groupby("source_code", sort=False).indices.items():
        if pooling == "centroid":  # one query vector: the normalised mean of the rows
            scores = normalise(vectors[idx].mean(axis=0, keepdims=True))[0] @ matrix.T
        else:
            scores = POOLING[pooling](vectors[idx] @ matrix.T)
        order = np.argpartition(-scores, k - 1)[:k]
        order = order[np.argsort(-scores[order])]
        rows += [(code, r + 1, concepts.concept_id.iat[i], float(scores[i])) for r, i in enumerate(order)]

    suffix = "" if pooling == "max" else f"__{pooling}"
    out = OUT_DIR / f"{route}__{model}-{dims}{suffix}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=["source_code", "rank", "concept_id", "score"]).to_csv(out, index=False)
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "route": route, "model": model, "deployment": EMBEDDING_DEPLOYMENTS[model], "dimensions": dims, "k": k,
        "n_codes": queries.source_code.nunique(), "n_query_rows": len(queries),
        "pooling": pooling if len(queries) > queries.source_code.nunique() else "single query",
        "index_run_id": index_manifest["run_id"], "vocabulary_version": index_manifest["vocabulary_version"],
    }
    out.with_suffix(".json").write_text(json.dumps(manifest, indent=2))
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--route", nargs="+", required=True)
    parser.add_argument("--variants", nargs="+", default=["small-1536", "large-1024", "large-1536", "large-2048", "large-3072"])
    parser.add_argument("--k", type=int, default=300)
    parser.add_argument("--pooling", choices=["max", "mean", "centroid"], default="max")
    args = parser.parse_args()
    index_cache = {}
    for route in args.route:
        for variant in args.variants:
            model, dims = variant.split("-")
            print("wrote", retrieve(route, model, int(dims), args.k, index_cache, args.pooling), flush=True)


if __name__ == "__main__":
    main()
