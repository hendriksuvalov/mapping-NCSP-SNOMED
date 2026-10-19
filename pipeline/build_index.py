"""Embed standard SNOMED CT procedure-class concepts and save them as a local index.

The run is checkpointed: every batch is written to <out>/parts/ as soon as it is embedded, so an
interrupted run resumes where it stopped when started again with the same arguments.

Output directory contents (after a completed run):
  embeddings.npy  float32 matrix, one row per concept (full model dimensions; truncate at load time)
  concepts.csv    concept_id, concept_name, concept_code (row i <-> embeddings[i])
  manifest.json   run configuration, model snapshot, token usage, timestamps

Examples:
  python build_index.py --model small --out index/small            # full index
  python build_index.py --model large --sample 1200 --out index/test_large
  python build_index.py --model small --concept-ids 4127237 4032745 --out index/test
"""
import argparse
import csv
import hashlib
import json
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from config import ATHENA_DIR, AZURE_API_VERSION, EMBEDDING_DEPLOYMENTS, azure_client, work_path


def load_procedures() -> pd.DataFrame:
    concepts = pd.read_csv(
        ATHENA_DIR / "CONCEPT.csv",
        sep="\t",
        dtype=str,
        keep_default_na=False,
        quoting=csv.QUOTE_NONE,
        usecols=["concept_id", "concept_name", "concept_class_id", "vocabulary_id",
                 "standard_concept", "concept_code"],
    )
    # SNOMED procedure class, not the OMOP Procedure domain: OMOP files ~20k SNOMED procedures
    # under Measurement/Observation, and some gold targets are among them.
    mask = (
        (concepts.vocabulary_id == "SNOMED")
        & (concepts.concept_class_id == "Procedure")
        & (concepts.standard_concept == "S")
    )
    return concepts.loc[mask, ["concept_id", "concept_name", "concept_code"]].reset_index(drop=True)


def vocabulary_version() -> str:
    vocab = pd.read_csv(ATHENA_DIR / "VOCABULARY.csv", sep="\t", dtype=str, keep_default_na=False)
    return dict(zip(vocab.vocabulary_id, vocab.vocabulary_version)).get("None", "unknown")


def select_concepts(args) -> pd.DataFrame:
    concepts = load_procedures()
    if args.concept_ids:
        missing = set(args.concept_ids) - set(concepts.concept_id)
        if missing:
            raise SystemExit(f"Not standard SNOMED procedure-class concepts: {sorted(missing)}")
        concepts = concepts[concepts.concept_id.isin(args.concept_ids)]
    if args.sample:
        concepts = concepts.sample(n=args.sample, random_state=args.seed)
    if args.limit:
        concepts = concepts.head(args.limit)
    return concepts.reset_index(drop=True)


def plan_hash(concepts: pd.DataFrame, deployment: str, batch_size: int) -> str:
    payload = "\n".join(concepts.concept_id + "\t" + concepts.concept_name) + f"\n{deployment}\n{batch_size}"
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--model", choices=EMBEDDING_DEPLOYMENTS, required=True)
    parser.add_argument("--sample", type=int, help="embed a seeded random sample of N concepts")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--limit", type=int, help="embed only the first N concepts")
    parser.add_argument("--concept-ids", nargs="+", help="embed only these concept_ids")
    parser.add_argument("--batch-size", type=int, default=500)
    parser.add_argument("--stop-after", type=int, help="stop after N new batches (to test resuming)")
    args = parser.parse_args()
    args.out = work_path(args.out)

    deployment = EMBEDDING_DEPLOYMENTS[args.model]
    concepts = select_concepts(args)
    run_id = plan_hash(concepts, deployment, args.batch_size)
    parts = args.out / "parts"
    progress_file = args.out / "progress.json"

    if (args.out / "manifest.json").exists():
        raise SystemExit(f"{args.out} already holds a completed index; choose another --out")
    if progress_file.exists():
        progress = json.loads(progress_file.read_text())
        if progress["run_id"] != run_id:
            raise SystemExit(f"{args.out} holds a different unfinished run; remove it or choose another --out")
        print(f"resuming run {run_id}: {len(progress['batches'])} batches already done")
    else:
        parts.mkdir(parents=True, exist_ok=True)
        progress = {"run_id": run_id, "started_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                    "batches": {}}

    client = azure_client()
    texts = concepts.concept_name.tolist()
    n_batches = -(-len(texts) // args.batch_size)
    new_batches = 0
    for b in range(n_batches):
        if str(b) in progress["batches"]:
            continue
        if args.stop_after is not None and new_batches >= args.stop_after:
            print(f"stopping early after {new_batches} new batches (--stop-after)")
            return
        batch = texts[b * args.batch_size:(b + 1) * args.batch_size]
        t0 = time.time()
        response = client.embeddings.create(input=batch, model=deployment)
        if len(response.data) != len(batch):
            raise RuntimeError(f"batch {b}: got {len(response.data)} embeddings for {len(batch)} inputs")
        # The API may return items out of order; sort by index to keep rows aligned.
        matrix = np.asarray([d.embedding for d in sorted(response.data, key=lambda d: d.index)], dtype=np.float32)
        np.save(parts / f"part_{b:05d}.npy", matrix)
        progress["batches"][str(b)] = {"rows": len(batch), "prompt_tokens": response.usage.prompt_tokens,
                                       "model": response.model, "seconds": round(time.time() - t0, 2)}
        progress_file.write_text(json.dumps(progress, indent=1))  # written after the part, so a done batch is always on disk
        new_batches += 1
        print(f"batch {b + 1}/{n_batches} ({len(progress['batches']) * 100 // n_batches}%)", flush=True)

    matrix = np.concatenate([np.load(parts / f"part_{b:05d}.npy") for b in range(n_batches)])
    if matrix.shape[0] != len(concepts):
        raise RuntimeError(f"merged {matrix.shape[0]} rows for {len(concepts)} concepts")
    np.save(args.out / "embeddings.npy", matrix)
    concepts.to_csv(args.out / "concepts.csv", index=False)
    batches = progress["batches"].values()
    manifest = {
        "run_id": run_id,
        "started_at": progress["started_at"],
        "finished_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "api_version": AZURE_API_VERSION,
        "deployment": deployment,
        "returned_models": sorted({b["model"] for b in batches}),
        "dimensions": int(matrix.shape[1]),
        "n_concepts": int(matrix.shape[0]),
        "embedded_text": "concept_name",
        "concept_filter": "SNOMED, concept_class_id Procedure, standard_concept S",
        "selection": {"sample": args.sample, "seed": args.seed, "limit": args.limit, "concept_ids": args.concept_ids},
        "batch_size": args.batch_size,
        "n_batches": n_batches,
        "vocabulary_version": vocabulary_version(),
        "prompt_tokens": sum(b["prompt_tokens"] for b in batches),
        "api_seconds": round(sum(b["seconds"] for b in batches), 1),
    }
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    shutil.rmtree(parts)
    progress_file.unlink()
    print(f"saved {matrix.shape} to {args.out}")


if __name__ == "__main__":
    main()
