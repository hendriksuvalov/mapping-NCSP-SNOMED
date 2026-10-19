"""Summarise reranker runs: accuracy, comparison with the embedding top-1, choice quality, token use.

  python analyze_rerank.py runs/rerank/et_expand__large-3072__k30__low__rerank.v1_f1.csv [...]
"""
import argparse
import json
from pathlib import Path

import pandas as pd

from evaluate import candidates_to_predictions, per_code_scores
from gold import load_gold_codes
from rerank import candidates_file
from config import work_path


def analyse(path: Path, gold: pd.DataFrame) -> dict:
    manifest = json.loads(path.with_suffix(".json").read_text())
    run = pd.read_csv(path, dtype=str, keep_default_na=False)
    g = gold[gold.source_code.isin(run.source_code)].reset_index(drop=True)
    predictions = {c: [i] for c, i in zip(run.source_code, run.concept_id) if i}
    scores = per_code_scores(predictions, g).set_index("source_code")

    candidates = pd.read_csv(candidates_file(manifest["route"], manifest["variant"]),
                             dtype={"source_code": str, "concept_id": str})
    candidates = candidates[candidates.source_code.isin(run.source_code)]
    embed = per_code_scores(candidates_to_predictions(candidates, 1), g).set_index("source_code")

    in_cands = run.set_index("source_code").gold_in_candidates == "True"
    choice = pd.to_numeric(run.set_index("source_code").choice, errors="coerce")
    n = len(run)
    return {
        "run": path.stem,
        "n": n,
        "rerank hit %": round(100 * scores.hit.mean(), 1),
        "embed top1 %": round(100 * embed.hit.mean(), 1),
        "gold in cands %": round(100 * in_cands.mean(), 1),
        "hit | gold in cands %": round(100 * scores.hit[in_cands].mean(), 1),
        "rescued": int(((scores.hit == 1) & (embed.hit == 0)).sum()),
        "damaged": int(((scores.hit == 0) & (embed.hit == 1)).sum()),
        "picked #1 %": round(100 * (choice == 1).mean(), 1),
        "recall %": round(100 * scores.recall.mean(), 1),
        "failed": manifest["failed_codes"],
        "tokens in/code": round(manifest["prompt_tokens"] / n),
        "tokens out/code": round(manifest["completion_tokens"] / n),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("runs", nargs="+", type=Path)
    args = parser.parse_args()
    args.runs = [work_path(p) for p in args.runs]
    gold = load_gold_codes()
    table = pd.DataFrame([analyse(p, gold) for p in args.runs]).set_index("run")
    print(table.T.to_string())


if __name__ == "__main__":
    main()
