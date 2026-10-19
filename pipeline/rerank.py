"""LLM reranking: Luna picks one SNOMED CT concept from the retrieved candidates for each NCSP code.

What Luna sees (design B): the original NCSP name in the route's source language (Estonian for et* routes,
English for en* routes) plus the NCSP hierarchy path, and the top-k candidates as a numbered list.
Generated translations/expansions are used only for retrieval, never shown here.

Luna answers with the list number of its choice. An answer that is not valid JSON or not a number in the
list is retried; if all attempts fail the code gets no prediction (a miss) and is flagged. There is no
silent fallback to the first candidate.

  python rerank.py --route et_expand --variant large-3072 --k 30 --folds 1
Output: runs/rerank/<name>.csv (source_code, concept_id, choice, gold_in_candidates, status)
        plus <name>.jsonl (every attempt) and <name>.json (manifest).
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

RERANK_V1 = """You are a clinical terminologist mapping codes from the NOMESCO Classification of Surgical Procedures \
(NCSP), as used in Estonia, to SNOMED CT. You receive one NCSP procedure name (in Estonian or English) with its place \
in the NCSP hierarchy, and a numbered list of candidate SNOMED CT concepts.

Choose the single candidate that best represents the NCSP procedure. Judge by:
- the core meaning: the same kind of procedure (action) on the same anatomical site;
- the clinical context given by the NCSP hierarchy;
- approach and qualifiers (e.g. laparoscopic, percutaneous) only where a candidate matches them.
The chosen concept must not be more specific than the NCSP procedure: if no candidate matches exactly, choose the \
closest more general concept rather than a narrower one. Ignore classification wording such as "other".

Return JSON: {"choice": <number of the chosen candidate>}"""

# v2: makes "no added detail" explicit (fold-1 errors were mostly candidates adding guidance/technique/sub-site).
RERANK_V2 = """You are a clinical terminologist mapping codes from the NOMESCO Classification of Surgical Procedures \
(NCSP), as used in Estonia, to SNOMED CT. You receive one NCSP procedure name (in Estonian or English) with its place \
in the NCSP hierarchy, and a numbered list of candidate SNOMED CT concepts.

Choose the single candidate that best represents the NCSP procedure. Judge by:
- the core meaning: the same kind of procedure (action) on the same anatomical site;
- the clinical context given by the NCSP hierarchy;
- approach and qualifiers (e.g. laparoscopic, percutaneous) only where a candidate matches them.

Do not choose a candidate that adds information the NCSP name does not state, such as imaging guidance \
(e.g. "ultrasound guided"), a specific technique (e.g. "fine needle", "core needle"), a narrower anatomical site, \
a specific device or a specific drug. Prefer the candidate that says exactly what the NCSP name says, even if it is \
more general; if no candidate matches exactly, choose the closest more general concept rather than a narrower one. \
Ignore classification wording such as "other".

Return JSON: {"choice": <number of the chosen candidate>}"""

# Appendix variant (E7): may return two concepts when the NCSP code combines two distinct procedures.
RERANK_MULTI_V1 = RERANK_V2.replace(
    'Return JSON: {"choice": <number of the chosen candidate>}',
    """Most NCSP procedures correspond to a single concept: then return exactly one candidate. Return two candidates \
only if the NCSP name combines two distinct procedures (for example a partial excision of an organ followed by an \
anastomosis) and no single candidate covers both; never return a second candidate that is merely a broader, narrower or \
alternative version of the first.

Return JSON: {"choices": [<number of the chosen candidate>]} or {"choices": [<number>, <number>]}""")

PROMPTS = {"rerank.v1": RERANK_V1, "rerank.v2": RERANK_V2, "rerank_multi.v1": RERANK_MULTI_V1}
MULTI_PROMPTS = {"rerank_multi.v1"}
def route_source(route: str) -> str:
    """Source language of a query route: et, et_translate, et_expand, et_expand_gpt4o -> et; en* -> en."""
    return route.split("_")[0]


def concept_names() -> pd.Series:
    concepts = pd.read_csv(ATHENA_DIR / "CONCEPT.csv", sep="\t", dtype=str, keep_default_na=False,
                           quoting=csv.QUOTE_NONE, usecols=["concept_id", "concept_name"])
    return concepts.set_index("concept_id").concept_name


def candidates_file(route: str, variant: str) -> Path:
    suffix = "__mean" if "_expand" in route else ""
    return WORK_DIR / "runs/retrieval" / f"{route}__{variant}{suffix}.csv"


def build_message(term: str, path: str, candidates: list[str], language: str) -> str:
    label = "NCSP procedure (Estonian)" if language == "et" else "NCSP procedure (English)"
    listing = "\n".join(f"{i}. {name}" for i, name in enumerate(candidates, 1))
    return f"{label}: {term}\nNCSP hierarchy: {path}\n\nCandidates:\n{listing}"


def parse_choice(content: str, k: int) -> int:
    choice = json.loads(content)["choice"]
    if isinstance(choice, str) and choice.strip().isdigit():
        choice = int(choice.strip())
    if not isinstance(choice, int) or isinstance(choice, bool) or not 1 <= choice <= k:
        raise ValueError(f"choice {choice!r} not in 1..{k}")
    return choice


def parse_choices(content: str, k: int) -> list[int]:
    """One or two distinct list numbers from {"choices": [...]}."""
    choices = json.loads(content)["choices"]
    if not isinstance(choices, list) or not 1 <= len(choices) <= 2:
        raise ValueError(f"choices {choices!r} must be a list of one or two numbers")
    parsed = [parse_choice(json.dumps({"choice": c}), k) for c in choices]
    if len(set(parsed)) != len(parsed):
        raise ValueError(f"duplicate choices {parsed}")
    return parsed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--route", required=True, help="query route, e.g. et_expand or et_expand_gpt4o")
    parser.add_argument("--variant", default="large-3072")
    parser.add_argument("--k", type=int, required=True)
    parser.add_argument("--prompt", choices=PROMPTS, default="rerank.v2")
    parser.add_argument("--llm", choices=CHAT_DEPLOYMENTS, default="luna")
    parser.add_argument("--reasoning-effort", default="low", choices=["none", "low", "medium", "high"],
                        help="only used by reasoning models")
    parser.add_argument("--folds", nargs="*")
    parser.add_argument("--out-dir", type=Path, default=WORK_DIR / "runs/rerank")
    parser.add_argument("--tag", default="", help="suffix for the output name (e.g. repeat number)")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--limit", type=int, help="only the first N codes (cost probes)")
    args = parser.parse_args()

    language = route_source(args.route)
    if language not in ("et", "en"):
        raise SystemExit(f"cannot tell the source language of route {args.route}")
    gold = load_gold_codes()
    codes = gold.source_code
    if args.folds:
        folds = pd.read_csv(WORK_DIR / "splits/folds.csv", dtype=str)
        codes = codes[codes.isin(folds[folds.fold.isin(args.folds)].source_code)]
    if args.limit:
        codes = codes.head(args.limit)
    terms = pd.read_csv(WORK_DIR / "queries" / f"{language}.csv", dtype=str, keep_default_na=False).set_index("source_code").text
    paths = hierarchy_paths(language)
    names = concept_names()
    candidates = pd.read_csv(candidates_file(args.route, args.variant), dtype={"source_code": str, "concept_id": str})
    candidates = candidates[(candidates["rank"] <= args.k) & candidates.source_code.isin(codes)]
    candidate_ids = candidates.sort_values(["source_code", "rank"]).groupby("source_code").concept_id.apply(list)
    targets = gold.set_index("source_code").index_targets

    fold_part = f"_f{''.join(args.folds)}" if args.folds else ""
    effort = args.reasoning_effort if args.llm == "luna" else args.llm   # keeps existing Luna run names unchanged
    name = f"{args.route}__{args.variant}__k{args.k}__{effort}__{args.prompt}{fold_part}{args.tag}"
    args.out_dir.mkdir(parents=True, exist_ok=True)
    log_path = args.out_dir / f"{name}.jsonl"
    done = {}
    for record in read_log(log_path):
        if record.get("final") and record.get("status") == "ok":  # failed codes are retried on rerun
            done[record["source_code"]] = record
    todo = [c for c in codes if c not in done]
    print(f"{name}: {len(codes)} codes, {len(done)} already done, {len(todo)} to go", flush=True)

    client = azure_client()
    lock = threading.Lock()
    stop = {}  # set to {"reason": "quota" | "network"} by the first worker that hits such an error

    def call(code):
        ids = candidate_ids[code]
        message = build_message(terms[code], paths[code], [names[i] for i in ids], language)
        messages = [{"role": "system", "content": PROMPTS[args.prompt]}, {"role": "user", "content": message}]
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
                if args.prompt in MULTI_PROMPTS:  # one or two answers, stored as "a;b"
                    choices = parse_choices(content, len(ids))
                    record.update(status="ok", choice=";".join(map(str, choices)),
                                  concept_id=";".join(ids[c - 1] for c in choices))
                else:
                    choice = parse_choice(content, len(ids))
                    record.update(status="ok", choice=choice, concept_id=ids[choice - 1])
            except Exception as error:  # API error after client retries, bad JSON, or out-of-range choice
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
        futures = [pool.submit(call, c) for c in todo]
        for i, future in enumerate(as_completed(futures), 1):
            record = future.result()
            if record is not None:
                done[record["source_code"]] = record
            if i % 50 == 0 or i == len(futures):
                print(f"  {i}/{len(futures)}", flush=True)
    if stop:
        exit_code = EXIT_QUOTA if stop["reason"] == "quota" else EXIT_NETWORK
        print(f"stopped ({stop['reason']}): {len(done)}/{len(codes)} codes done; rerun to resume", flush=True)
        sys.exit(exit_code)

    rows = []
    for code in codes:
        record = done[code]
        rows.append({"source_code": code, "concept_id": record.get("concept_id", ""), "choice": record.get("choice", ""),
                     "gold_in_candidates": bool(set(candidate_ids[code]) & targets[code]), "status": record["status"]})
    pd.DataFrame(rows).to_csv(args.out_dir / f"{name}.csv", index=False)

    attempts = read_log(log_path)
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "route": args.route, "variant": args.variant, "k": args.k, "prompt": args.prompt,
        "system_prompt": PROMPTS[args.prompt], "shown_to_llm": f"original NCSP name ({language}) + NCSP hierarchy path",
        "reasoning_effort": args.reasoning_effort, "temperature": "default (1); not configurable for this model",
        "folds": args.folds, "api_version": AZURE_API_VERSION, "llm": args.llm, "deployment": CHAT_DEPLOYMENTS[args.llm],
        "returned_models": sorted({a["model"] for a in attempts if "model" in a}),
        "n_codes": len(rows), "failed_codes": sum(r["status"] != "ok" for r in rows),
        "attempts_total": len(attempts), "failed_attempts": sum(a["status"] == "failed" for a in attempts),
        "interrupted_attempts": sum(a["status"] in ("quota", "network") for a in attempts),
        "prompt_tokens": sum(a.get("prompt_tokens", 0) for a in attempts),
        "completion_tokens": sum(a.get("completion_tokens", 0) for a in attempts),
        "reasoning_tokens": sum(a.get("reasoning_tokens", 0) for a in attempts),
    }
    (args.out_dir / f"{name}.json").write_text(json.dumps(manifest, indent=2))
    print(f"wrote {args.out_dir / name}.csv  (failed codes: {manifest['failed_codes']})")


if __name__ == "__main__":
    main()
