"""Generate LLM-based query texts (translation / semantic expansion) for retrieval.

Each prompt version is a named entry in PROMPTS. Raw responses are appended to a JSONL log as they arrive,
so an interrupted run resumes; the query file (source_code, text) is written at the end.

  python llm_queries.py --prompt et_translate.v1 --folds 1 --out queries/dev/et_translate.v1.csv
  python llm_queries.py --prompt et_expand.v1 --out queries/et_expand.csv       # all codes
"""
import argparse
import json
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from config import (AZURE_API_VERSION, CHAT_DEPLOYMENTS, EXIT_NETWORK, EXIT_QUOTA, GOLD_DIR, WORK_DIR, azure_client,
                    chat_options, describe_error, read_log, reasoning_tokens, stop_reason, work_path)

TRANSLATE_V1 = """You are a clinical terminologist. You receive the Estonian name of a surgical procedure from the \
NOMESCO Classification of Surgical Procedures (NCSP). Translate it into English medical terminology, phrased the way a \
SNOMED CT procedure concept would be named. Keep every detail of the original (approach, anatomy, laterality, purpose) \
and add nothing that is not in the original.

Return JSON: {"translation": "<English name>"}"""

EXPAND_V1 = """You are a clinical terminologist. You receive the {language} name of a surgical procedure from the \
NOMESCO Classification of Surgical Procedures (NCSP). Your output is used to search SNOMED CT for the matching \
procedure concept, so write English phrasings that such a concept could plausibly be named:

1. the most likely SNOMED CT-style preferred term for exactly this procedure;
2. two or three synonyms or alternative phrasings of it (different word order, Latin/Greek vs plain terms);
3. one slightly more general procedure name, in case SNOMED CT has no concept this specific;
4. one short sentence describing what is done.

Keep the meaning of the original; do not invent details. Return JSON: {{"variants": ["...", "..."]}}"""

# v2: adds the NCSP hierarchy path as context, two generalisation levels, and terminology guidance.
TRANSLATE_V2 = """You are a clinical terminologist. You receive the Estonian name of a surgical procedure from the \
NOMESCO Classification of Surgical Procedures (NCSP), together with its place in the NCSP hierarchy (chapter > \
section > group), which tells you the clinical context. Translate the procedure name into English medical terminology, \
phrased the way a SNOMED CT procedure concept would be named. Use established English procedure terms rather than \
literal word-by-word translations. Keep every detail of the original (approach, anatomy, laterality, purpose), add \
nothing that is not in the original, and leave out classification wording such as "other" or "not elsewhere classified".

Return JSON: {"translation": "<English name>"}"""

EXPAND_V2 = """You are a clinical terminologist. You receive the {language} name of a surgical procedure from the \
NOMESCO Classification of Surgical Procedures (NCSP), together with its place in the NCSP hierarchy (chapter > \
section > group), which tells you the clinical context. Your output is used to search SNOMED CT for the matching \
procedure concept. Write English phrasings that such a concept could plausibly be named, using established \
SNOMED CT-style procedure terminology rather than literal translations:

1. the most likely SNOMED CT-style preferred term for exactly this procedure;
2. two or three synonyms or alternative phrasings of it (different word order, eponyms, Latin/Greek vs plain terms);
3. a more general name without the approach and qualifiers (e.g. without "laparoscopic", "percutaneous", "endoscopic");
4. a broad name at the level of the procedure type and organ (e.g. "Operation on <organ>", "Procedure on <body part>");
5. one short sentence describing what is done.

Leave out classification wording such as "other" or "not elsewhere classified". Keep the meaning of the original; \
do not invent details. Return JSON: {{"variants": ["...", "..."]}}"""

PROMPTS = {
    "et_translate.v1": {"system": TRANSLATE_V1, "source": "et", "key": "translation"},
    "et_expand.v1": {"system": EXPAND_V1.format(language="Estonian"), "source": "et", "key": "variants"},
    "en_expand.v1": {"system": EXPAND_V1.format(language="English"), "source": "en", "key": "variants"},
    "et_translate.v2": {"system": TRANSLATE_V2, "source": "et", "key": "translation", "context": True},
    "et_expand.v2": {"system": EXPAND_V2.format(language="Estonian"), "source": "et", "key": "variants", "context": True},
    "en_expand.v2": {"system": EXPAND_V2.format(language="English"), "source": "en", "key": "variants", "context": True},
}


def hierarchy_paths(language: str) -> pd.Series:
    """NCSP chapter > section > group path per code, in Estonian ('et') or English ('en')."""
    classifier = pd.read_csv(GOLD_DIR / "NCSP" / "ncsp_classifier.csv", dtype=str, keep_default_na=False)
    classifier = classifier.set_index("code")
    names = classifier.name if language == "et" else classifier.english_name
    names = names.str.replace("\xa0", " ").str.strip()
    parents = classifier[["parent0_code", "parent1_code", "parent2_code"]]
    return parents.apply(lambda r: " > ".join(names.get(c, "") for c in r if c), axis=1)


def parse(content: str, key: str) -> list[str]:
    value = json.loads(content)[key]
    texts = [value] if isinstance(value, str) else list(value)
    texts = [t.strip() for t in texts if isinstance(t, str) and t.strip()]
    if not texts:
        raise ValueError("no usable text in response")
    return list(dict.fromkeys(texts))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", choices=PROMPTS, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--folds", nargs="*", help="only codes in these folds (prompt development: fold 1)")
    parser.add_argument("--llm", choices=CHAT_DEPLOYMENTS, default="luna")
    parser.add_argument("--reasoning-effort", default="low", choices=["none", "low", "medium", "high"],
                        help="only used by reasoning models")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    args.out = work_path(args.out)

    spec = PROMPTS[args.prompt]
    source = pd.read_csv(WORK_DIR / "queries" / f"{spec['source']}.csv", dtype=str, keep_default_na=False)
    if args.folds:
        folds = pd.read_csv(WORK_DIR / "splits/folds.csv", dtype=str)
        source = source[source.source_code.isin(folds[folds.fold.isin(args.folds)].source_code)]
    if spec.get("context"):
        paths = hierarchy_paths(spec["source"])
        label = "Term" if spec["source"] == "en" else "Termin"
        source = source.assign(text=[f"{label}: {t}\nNCSP: {paths[c]}" for c, t in zip(source.source_code, source.text)])

    log_path = args.out.with_suffix(".jsonl")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    done = {}
    for record in read_log(log_path):
        if record.get("texts"):
            done[record["source_code"]] = record
    todo = source[~source.source_code.isin(done)]
    print(f"{args.prompt}: {len(source)} codes, {len(done)} already done, {len(todo)} to go", flush=True)

    client = azure_client()
    lock = threading.Lock()
    stop = {}  # set to {"reason": "quota" | "network"} by the first worker that hits such an error

    def call(code, text):
        messages = [{"role": "system", "content": spec["system"]}, {"role": "user", "content": text}]
        record = None
        for attempt in range(3):
            if stop:
                return None
            t0 = time.time()
            try:
                response = client.chat.completions.create(
                    messages=messages, max_completion_tokens=4000, **chat_options(args.llm, args.reasoning_effort),
                )
            except Exception as error:
                reason = stop_reason(error)
                if reason:  # stop the whole run; this code is neither done nor failed
                    stop.setdefault("reason", reason)
                    return None
                record = {"source_code": code, "input": text, "attempt": attempt + 1,
                          "error": describe_error(error)}
                with lock, log_path.open("a") as log:
                    log.write(json.dumps(record, ensure_ascii=False) + "\n")
                continue
            content = response.choices[0].message.content or ""
            record = {"source_code": code, "input": text, "attempt": attempt + 1, "raw": content,
                      "model": response.model, "seconds": round(time.time() - t0, 2),
                      "prompt_tokens": response.usage.prompt_tokens,
                      "completion_tokens": response.usage.completion_tokens,
                      "reasoning_tokens": reasoning_tokens(response.usage)}
            try:
                record["texts"] = parse(content, spec["key"])
            except (ValueError, KeyError, TypeError) as error:  # logged, then retried
                record["error"] = f"{type(error).__name__}: {error}"
            with lock, log_path.open("a") as log:
                log.write(json.dumps(record, ensure_ascii=False) + "\n")
            if "texts" in record:
                return record
        return record

    failures = 0
    with ThreadPoolExecutor(args.workers) as pool:
        futures = [pool.submit(call, c, t) for c, t in zip(todo.source_code, todo.text)]
        for i, future in enumerate(as_completed(futures), 1):
            record = future.result()
            if record is None:
                continue
            if "texts" in record:
                done[record["source_code"]] = record
            else:
                failures += 1
            if i % 25 == 0 or i == len(futures):
                print(f"  {i}/{len(futures)}", flush=True)

    if stop:
        exit_code = EXIT_QUOTA if stop["reason"] == "quota" else EXIT_NETWORK
        print(f"stopped ({stop['reason']}): {len(done)}/{len(source)} codes done; rerun to resume", flush=True)
        sys.exit(exit_code)
    missing = set(source.source_code) - set(done)
    if missing:
        raise SystemExit(f"{len(missing)} codes failed after retries (see {log_path}); rerun to retry them")
    rows = [(code, t) for code in source.source_code for t in done[code]["texts"]]
    pd.DataFrame(rows, columns=["source_code", "text"]).to_csv(args.out, index=False)

    records = [done[c] for c in source.source_code]
    all_attempts = read_log(log_path)
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "prompt": args.prompt, "system_prompt": spec["system"], "source_route": spec["source"],
        "folds": args.folds, "api_version": AZURE_API_VERSION,
        "llm": args.llm, "deployment": CHAT_DEPLOYMENTS[args.llm], "returned_models": sorted({r["model"] for r in records}),
        "reasoning_effort": args.reasoning_effort, "temperature": "default (1); not configurable for this model",
        "n_codes": len(records), "n_query_rows": len(rows),
        "attempts_total": len(all_attempts), "parse_failures": sum("error" in r for r in all_attempts),
        "prompt_tokens": sum(r.get("prompt_tokens", 0) for r in all_attempts),
        "completion_tokens": sum(r.get("completion_tokens", 0) for r in all_attempts),
        "reasoning_tokens": sum(r.get("reasoning_tokens") or 0 for r in all_attempts),
    }
    args.out.with_suffix(".json").write_text(json.dumps(manifest, indent=2))
    print(f"wrote {len(rows)} query rows for {len(records)} codes to {args.out}")


if __name__ == "__main__":
    main()
