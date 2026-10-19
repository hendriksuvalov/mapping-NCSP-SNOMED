"""Suggest SNOMED CT mappings for NCSP codes in Excel files, with the article's best pipeline.

Pipeline (frozen, as in the article): Luna expands the official English NCSP name (prompt en_expand.v2, with the NCSP
hierarchy path) → text-embedding-3-large search over the 71,181 standard SNOMED CT procedure-class concepts (mean
similarity over the expansion phrasings) → top 30 → Luna chooses one (prompt rerank.v2).

Input: Excel files with an NCSP_CODE column (all sheets). Output: a copy of each file named "<name> - suggestions.xlsx"
with added columns; the input files are not changed. LLM answers are cached in <work dir>/log.jsonl, so an
interrupted run resumes; exit codes 2/3 mean quota/network (rerun later).

  python map_codes.py path/to/requests/*.xlsx --work runs/mapping_requests
"""
import argparse
import json
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from config import (ATHENA_DIR, CHAT_DEPLOYMENTS, EMBEDDING_DEPLOYMENTS, EXIT_NETWORK, EXIT_QUOTA, GOLD_DIR,
                    WORK_DIR, azure_client, chat_options, describe_error, read_log, stop_reason)
from index_io import embed_texts, load_index, normalise
from llm_queries import PROMPTS as QUERY_PROMPTS, hierarchy_paths, parse
from rerank import PROMPTS as RERANK_PROMPTS, build_message, parse_choice

K = 30
QUALIFIER_NOTE = ("NCSP qualifier code (side, urgency, duration, …), not a procedure: SNOMED CT represents it as a "
                  "qualifier value, which is outside the procedure concepts searched here. Suggestions are unreliable.")


def llm_step(kind, codes, make_message, spec_system, parse_answer, log_path, workers=8):
    """Run one LLM call per code (cached in log_path). Returns {code: parsed answer}."""
    done = {r["code"]: r["answer"] for r in read_log(log_path) if r.get("kind") == kind and "answer" in r}
    todo = [c for c in codes if c not in done]
    client, lock, stop = azure_client(), threading.Lock(), {}

    def call(code):
        messages = [{"role": "system", "content": spec_system}, {"role": "user", "content": make_message(code)}]
        for attempt in range(1, 4):
            if stop:
                return
            try:
                response = client.chat.completions.create(messages=messages, max_completion_tokens=8000,
                                                          **chat_options("luna", "low"))
                answer = parse_answer(response.choices[0].message.content or "")
                record = {"kind": kind, "code": code, "answer": answer, "model": response.model,
                          "prompt_tokens": response.usage.prompt_tokens,
                          "completion_tokens": response.usage.completion_tokens}
            except Exception as error:
                reason = stop_reason(error)
                if reason:
                    stop.setdefault("reason", reason)
                    return
                record = {"kind": kind, "code": code, "attempt": attempt, "error": describe_error(error)}
            with lock, log_path.open("a") as log:
                log.write(json.dumps(record, ensure_ascii=False) + "\n")
            if "answer" in record:
                done[code] = record["answer"]
                return

    with ThreadPoolExecutor(workers) as pool:
        for future in as_completed([pool.submit(call, c) for c in todo]):
            future.result()
    if stop:
        print(f"stopped ({stop['reason']}) during {kind}; rerun to resume", flush=True)
        sys.exit(EXIT_QUOTA if stop["reason"] == "quota" else EXIT_NETWORK)
    missing = [c for c in codes if c not in done]
    if missing:
        raise SystemExit(f"{kind}: {len(missing)} codes failed after 3 attempts (see {log_path}); rerun to retry")
    print(f"{kind}: {len(codes)} codes done ({len(todo)} new calls)", flush=True)
    return done


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("files", nargs="+", type=Path)
    parser.add_argument("--work", type=Path, default=WORK_DIR / "runs/mapping_requests")
    args = parser.parse_args()
    args.work.mkdir(parents=True, exist_ok=True)
    log_path = args.work / "log.jsonl"

    books = {f: pd.read_excel(f, sheet_name=None, dtype=str) for f in args.files}
    codes = sorted({c.strip() for sheets in books.values() for s in sheets.values() for c in s["NCSP_CODE"].dropna()})
    classifier = pd.read_csv(GOLD_DIR / "NCSP" / "ncsp_classifier.csv", dtype=str, keep_default_na=False).set_index("code")
    unknown = [c for c in codes if c not in classifier.index]
    if unknown:
        raise SystemExit(f"codes not in the NCSP classifier: {unknown}")
    english = classifier.english_name.str.strip()
    paths = hierarchy_paths("en")
    print(f"{len(codes)} unique codes from {len(args.files)} files", flush=True)

    # 1. Expansion (Luna, en_expand.v2)
    expand_spec = QUERY_PROMPTS["en_expand.v2"]
    expansions = llm_step("expand", codes, lambda c: f"Term: {english[c]}\nNCSP: {paths[c]}", expand_spec["system"],
                          lambda content: parse(content, expand_spec["key"]), log_path)

    # 2. Search: mean similarity over the phrasings, top 30
    concepts, matrix, _ = load_index(WORK_DIR / "index/large")
    client = azure_client()
    candidates = {}
    for code in codes:
        vectors = normalise(embed_texts(client, EMBEDDING_DEPLOYMENTS["large"], expansions[code]))
        scores = (vectors @ matrix.T).mean(axis=0)
        top = np.argpartition(-scores, K - 1)[:K]
        candidates[code] = list(concepts.concept_id.to_numpy()[top[np.argsort(-scores[top])]])
    print("search done", flush=True)

    # 3. Choice (Luna, rerank.v2)
    snomed = pd.read_csv(ATHENA_DIR / "CONCEPT.csv", sep="\t", dtype=str, keep_default_na=False, quoting=3,
                         usecols=["concept_id", "concept_name", "concept_code"]).set_index("concept_id")
    choices = llm_step("choose", codes,
                       lambda c: build_message(english[c], paths[c], [snomed.concept_name[i] for i in candidates[c]], "en"),
                       RERANK_PROMPTS["rerank.v2"], lambda content: parse_choice(content, K), log_path)

    # 4. Output files
    def columns_for(code):
        code = str(code).strip()
        if code not in choices:
            return {}
        chosen = candidates[code][choices[code] - 1]
        listing = "\n".join(f"{snomed.concept_code[i]} | {snomed.concept_name[i]}" for i in candidates[code])
        return {
            "Exact guess (SNOMED CT ID)": snomed.concept_code[chosen],
            "Exact guess (name)": snomed.concept_name[chosen],
            "Exact guess (OMOP concept_id)": chosen,
            "Top 30 candidates by similarity (SNOMED CT ID | name)": listing,
            "Note": QUALIFIER_NOTE if code.startswith("ZX") else "",
        }

    records = [json.loads(l) for l in log_path.read_text().splitlines() if l.strip().startswith("{")]
    models = sorted({r["model"] for r in records if "model" in r})
    about = pd.DataFrame({"About these suggestions": [
        f"Automatic suggestions made on {date.today()}; not reviewed by an expert. Please check before use.",
        "Exact guess: the concept chosen by the LLM among the 30 candidates. Top 30: candidates ordered by embedding "
        "similarity (best first); the exact guess is usually, but not always, near the top.",
        "Method: the official English NCSP name is expanded by an LLM into several SNOMED CT-style phrasings (using "
        "the NCSP hierarchy as context), these are compared with 71,181 standard SNOMED CT procedure-class concepts "
        "(text-embedding-3-large, mean similarity), and the LLM chooses one of the 30 most similar concepts.",
        f"Models: {', '.join(models)} ({CHAT_DEPLOYMENTS['luna']}), {EMBEDDING_DEPLOYMENTS['large']}. Vocabulary: "
        "OMOP vocabularies v5.0 29-AUG-26 (SNOMED CT 2026-02 International Edition).",
        "Expected quality (tested on NCSP chapters L, M and T; other chapters may differ): about 62% exact agreement "
        "with expert mappings, about 78% exact or one step away in the SNOMED CT hierarchy, and the correct concept is "
        "among the 30 candidates for about 93% of codes. Wrong guesses are most often one step more specific than an "
        "expert would choose.",
        "Only procedure concepts were searched: codes whose correct mapping is a qualifier, finding, device or drug "
        "cannot be matched (see the Note column).",
    ]})
    for path, sheets in books.items():
        out = path.with_name(f"{path.stem} - suggestions.xlsx")
        with pd.ExcelWriter(out, engine="openpyxl") as writer:
            for name, frame in sheets.items():
                extra = pd.DataFrame([columns_for(c) for c in frame["NCSP_CODE"]], index=frame.index)
                pd.concat([frame, extra], axis=1).to_excel(writer, sheet_name=name, index=False)
            about.to_excel(writer, sheet_name="About", index=False)
            for name, sheet in writer.sheets.items():
                from openpyxl.styles import Alignment
                for column in sheet.columns:
                    header = str(column[0].value or "")
                    width = 70 if header.startswith("Top 30") or header == "About these suggestions" else \
                        45 if "name" in header.lower() or header == "Note" else 18
                    sheet.column_dimensions[column[0].column_letter].width = width
                    for cell in column:
                        cell.alignment = Alignment(wrap_text=True, vertical="top")
                sheet.freeze_panes = "B2"
        print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()
