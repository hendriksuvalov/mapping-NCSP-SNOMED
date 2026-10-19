"""Zero-shot baseline: Luna maps each NCSP code to SNOMED CT from its own knowledge (no retrieval).

Luna sees the same input as the reranker (original NCSP name in the route's language + NCSP hierarchy path) and
returns a SNOMED CT concept id (SCTID) and preferred term. The SCTID is converted to an OMOP concept_id through
Athena's SNOMED concept_code, and scoring is by concept id (primary). As a diagnostic, the returned term is also
matched to a SNOMED concept of the same name (case-insensitive).

  python zero_shot.py --language et
Output: runs/zero_shot/<language>__<effort>.csv / .jsonl / .json
"""
import argparse
import csv
import json
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from config import (ATHENA_DIR, AZURE_API_VERSION, CHAT_DEPLOYMENTS, EXIT_NETWORK, EXIT_QUOTA, WORK_DIR,
                    azure_client, chat_options, describe_error, read_log, reasoning_tokens, stop_reason)
from gold import load_gold_codes
from llm_queries import hierarchy_paths

ZERO_SHOT_V1 = """You are a clinical terminologist mapping codes from the NOMESCO Classification of Surgical \
Procedures (NCSP), as used in Estonia, to SNOMED CT. You receive one NCSP procedure name (in Estonian or English) \
with its place in the NCSP hierarchy. Give the single SNOMED CT procedure concept that best represents it: its \
SNOMED CT concept identifier (SCTID) and its preferred term. Do not choose a concept that adds information the NCSP \
name does not state; if there is no exact match, give the closest more general concept.

Return JSON: {"sctid": "<SNOMED CT concept id>", "term": "<preferred term>"}"""


def snomed_lookup():
    concepts = pd.read_csv(ATHENA_DIR / "CONCEPT.csv", sep="\t", dtype=str, keep_default_na=False,
                           quoting=csv.QUOTE_NONE,
                           usecols=["concept_id", "concept_name", "vocabulary_id", "concept_code", "standard_concept"])
    snomed = concepts[concepts.vocabulary_id == "SNOMED"]
    by_code = snomed.drop_duplicates("concept_code").set_index("concept_code")
    standard = snomed[snomed.standard_concept == "S"]
    by_name = standard.assign(key=standard.concept_name.str.lower()).drop_duplicates("key").set_index("key").concept_id
    return by_code, by_name


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--language", choices=["et", "en"], required=True)
    parser.add_argument("--llm", choices=CHAT_DEPLOYMENTS, default="luna")
    parser.add_argument("--reasoning-effort", default="low", choices=["none", "low", "medium", "high"],
                        help="only used by reasoning models")
    parser.add_argument("--folds", nargs="*")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    codes = load_gold_codes().source_code
    if args.folds:
        folds = pd.read_csv(WORK_DIR / "splits/folds.csv", dtype=str)
        codes = codes[codes.isin(folds[folds.fold.isin(args.folds)].source_code)]
    terms = pd.read_csv(WORK_DIR / "queries" / f"{args.language}.csv", dtype=str, keep_default_na=False).set_index("source_code").text
    paths = hierarchy_paths(args.language)
    by_code, by_name = snomed_lookup()

    out_dir = WORK_DIR / "runs/zero_shot"
    out_dir.mkdir(parents=True, exist_ok=True)
    name = f"{args.language}__{args.reasoning_effort if args.llm == 'luna' else args.llm}" + (f"_f{''.join(args.folds)}" if args.folds else "")
    log_path = out_dir / f"{name}.jsonl"
    done = {}
    for record in read_log(log_path):
        if record.get("final") and record.get("status") == "ok":  # failed codes are retried on rerun
            done[record["source_code"]] = record
    todo = [c for c in codes if c not in done]
    print(f"{name}: {len(codes)} codes, {len(todo)} to go", flush=True)

    client = azure_client()
    lock = threading.Lock()
    stop = {}  # set to {"reason": "quota" | "network"} by the first worker that hits such an error
    label = "NCSP procedure (Estonian)" if args.language == "et" else "NCSP procedure (English)"

    def call(code):
        message = f"{label}: {terms[code]}\nNCSP hierarchy: {paths[code]}"
        messages = [{"role": "system", "content": ZERO_SHOT_V1}, {"role": "user", "content": message}]
        for attempt in range(1, 4):
            if stop:
                return None
            t0 = time.time()
            record = {"source_code": code, "attempt": attempt}
            try:
                response = client.chat.completions.create(
                    messages=messages, max_completion_tokens=8000, **chat_options(args.llm, args.reasoning_effort),
                )
                content = response.choices[0].message.content or ""
                record.update(raw=content, model=response.model, seconds=round(time.time() - t0, 2),
                              prompt_tokens=response.usage.prompt_tokens,
                              completion_tokens=response.usage.completion_tokens,
                              reasoning_tokens=reasoning_tokens(response.usage))
                answer = json.loads(content)
                record.update(status="ok", sctid=str(answer.get("sctid", "")).strip(), term=str(answer.get("term", "")).strip())
            except Exception as error:
                reason = stop_reason(error)
                if reason:  # stop the whole run; this code is neither done nor failed
                    stop.setdefault("reason", reason)
                    record.update(status=reason, error=describe_error(error), final=False)
                    with lock, log_path.open("a") as log:
                        log.write(json.dumps(record, ensure_ascii=False) + "\n")
                    return None
                record.update(status="failed", error=describe_error(error))
            record["final"] = record["status"] == "ok" or attempt == 3
            with lock, log_path.open("a") as log:
                log.write(json.dumps(record, ensure_ascii=False) + "\n")
            if record["final"]:
                return record

    with ThreadPoolExecutor(args.workers) as pool:
        for future in as_completed([pool.submit(call, c) for c in todo]):
            record = future.result()
            if record is not None:
                done[record["source_code"]] = record
    if stop:
        exit_code = EXIT_QUOTA if stop["reason"] == "quota" else EXIT_NETWORK
        print(f"stopped ({stop['reason']}): {len(done)}/{len(codes)} codes done; rerun to resume", flush=True)
        sys.exit(exit_code)

    rows = []
    for code in codes:
        r = done[code]
        sctid, term = r.get("sctid", ""), r.get("term", "")
        known = sctid in by_code.index
        id_concept = by_code.concept_id.get(sctid, "")
        rows.append({
            "source_code": code, "sctid": sctid, "term": term, "status": r["status"],
            "sctid_exists": known,
            "concept_id": id_concept,                                  # primary: by identifier
            "id_name_agrees": known and by_code.concept_name[sctid].lower() == term.lower(),
            "concept_id_by_name": by_name.get(term.lower(), ""),       # diagnostic: by returned term
        })
    pd.DataFrame(rows).to_csv(out_dir / f"{name}.csv", index=False)
    attempts = read_log(log_path)
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "language": args.language,
        "system_prompt": ZERO_SHOT_V1, "reasoning_effort": args.reasoning_effort, "folds": args.folds,
        "api_version": AZURE_API_VERSION, "llm": args.llm, "deployment": CHAT_DEPLOYMENTS[args.llm],
        "returned_models": sorted({a["model"] for a in attempts if "model" in a}),
        "n_codes": len(rows), "failed_codes": sum(r["status"] != "ok" for r in rows),
        "prompt_tokens": sum(a.get("prompt_tokens", 0) for a in attempts),
        "completion_tokens": sum(a.get("completion_tokens", 0) for a in attempts),
    }
    (out_dir / f"{name}.json").write_text(json.dumps(manifest, indent=2))
    print(f"wrote {out_dir / name}.csv (failed codes: {manifest['failed_codes']})")


if __name__ == "__main__":
    main()
