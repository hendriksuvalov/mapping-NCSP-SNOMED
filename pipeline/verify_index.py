"""Sanity-check a saved index: reload it, check alignment with Athena, re-embed a sample, run queries.

Example:
  python verify_index.py index/test --query "Puncture of ovarian cyst" "Munasarjatsüsti punktsioon"
"""
import argparse
from pathlib import Path

import numpy as np

from build_index import load_procedures
from config import azure_client, work_path
from index_io import embed_texts, load_index, search


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("index_dir", type=Path)
    parser.add_argument("--dimensions", type=int, help="check a truncated view of the index")
    parser.add_argument("--recheck", type=int, default=50, help="re-embed this many random rows")
    parser.add_argument("--query", nargs="*", default=[])
    parser.add_argument("--top", type=int, default=3)
    args = parser.parse_args()
    args.index_dir = work_path(args.index_dir)

    concepts, matrix, manifest = load_index(args.index_dir, args.dimensions)
    deployment = manifest["deployment"]
    checks = {
        "row count matches concepts.csv and manifest": matrix.shape[0] == len(concepts) == manifest["n_concepts"],
        "dimensions as expected": matrix.shape[1] == (args.dimensions or manifest["dimensions"]),
        "dtype float32": matrix.dtype == np.float32,
        "no NaN/inf": bool(np.isfinite(matrix).all()),
        "rows unit-normalised": bool(np.allclose(np.linalg.norm(matrix, axis=1), 1, atol=1e-3)),
        "no duplicate concept_ids": concepts.concept_id.is_unique,
    }
    reference = load_procedures().set_index("concept_id")
    checks["ids and names match Athena CONCEPT.csv"] = bool(
        concepts.concept_id.isin(reference.index).all()
        and (reference.loc[concepts.concept_id, "concept_name"].values == concepts.concept_name.values).all()
    )

    # Re-embed a random sample of stored names: row i must match a fresh embedding of name i,
    # and searching with that fresh embedding must return row i itself.
    rows = np.random.default_rng(0).choice(len(concepts), min(args.recheck, len(concepts)), replace=False)
    client = azure_client()
    fresh = embed_texts(client, deployment, concepts.concept_name.iloc[rows].tolist(), args.dimensions)
    self_sim = (matrix[rows] * fresh).sum(axis=1)
    top, _ = search(matrix, fresh, 1)
    # Identical names (different concepts) can legitimately tie, so compare names, not rows.
    same_name = concepts.concept_name.values[top[:, 0]] == concepts.concept_name.values[rows]
    checks[f"stored row == fresh embedding for {len(rows)} sampled names (cos > 0.99)"] = bool((self_sim > 0.99).all())
    checks["search with a fresh embedding returns that concept first"] = bool(same_name.all())

    print(f"{args.index_dir}  ({deployment}, {matrix.shape[0]} x {matrix.shape[1]})")
    for name, ok in checks.items():
        print(f"  [{'OK' if ok else 'FAIL'}] {name}")
    print(f"  self-similarity min {self_sim.min():.5f}")

    if args.query:
        q = embed_texts(client, deployment, args.query, args.dimensions)
        top, scores = search(matrix, q, args.top)
        for text, idx, sc in zip(args.query, top, scores):
            hits = ", ".join(f"{concepts.concept_name[i]} ({concepts.concept_id[i]}) {s:.3f}" for i, s in zip(idx, sc))
            print(f"\n  query: {text}\n    -> {hits}")

    raise SystemExit(0 if all(checks.values()) else 1)


if __name__ == "__main__":
    main()
