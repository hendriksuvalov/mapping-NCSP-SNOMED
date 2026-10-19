"""Run the experiment plan step by step, surviving network outages.

Each step is a resumable script. Exit code 3 (network) -> wait until the endpoint is reachable, then rerun the same
step, which continues where it stopped. Exit code 2 (quota) -> stop; start again after the quota resets.
Any other failure stops the plan so it can be looked at.

  python run_plan.py day1                 # all day-1 steps
  python run_plan.py day1 --steps E2      # only steps whose name starts with E2
Run detached so it survives a closed terminal:
  nohup python run_plan.py day1 --steps E2 > ../outputs/logs/day1.log 2>&1 &
"""
import argparse
import socket
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from urllib.parse import urlparse

from config import AZURE_ENDPOINT, EXIT_NETWORK, EXIT_QUOTA, WORK_DIR

PY = sys.executable
TEST = ["--folds", "2", "3", "4", "5"]
MAIN = ["--variant", "large-3072", "--k", "30", "--prompt", "rerank.v2", "--reasoning-effort", "low"]

PLANS = {
    "day1": [
        *[(f"E2 RAG {route}", [PY, "rerank.py", "--route", route, *MAIN, *TEST])
          for route in ["et_expand", "en_expand", "et", "et_translate", "en"]],
        ("E1 zero-shot en", [PY, "zero_shot.py", "--language", "en"]),
        ("E6 RAG et_expand k10", [PY, "rerank.py", "--route", "et_expand", "--variant", "large-3072", "--k", "10",
                                  "--prompt", "rerank.v2", "--reasoning-effort", "low", *TEST]),
        ("E9 RAG et_expand small+luna", [PY, "rerank.py", "--route", "et_expand", "--variant", "small-1536", "--k", "30",
                                         "--prompt", "rerank.v2", "--reasoning-effort", "low", *TEST]),
    ],
    # Independent parts of day 2, started with the leftover quota of 28 Sep; continues after the reset.
    "day2a": [
        ("E6 RAG et_expand k50", [PY, "rerank.py", "--route", "et_expand", "--variant", "large-3072", "--k", "50",
                                  "--prompt", "rerank.v2", "--reasoning-effort", "low", *TEST]),
        ("E9 gpt4o expansion", [PY, "llm_queries.py", "--prompt", "et_expand.v2", "--llm", "gpt4o", *TEST,
                                "--out", str(WORK_DIR / "queries/et_expand_gpt4o.csv")]),
        ("E9 gpt4o retrieval", [PY, "retrieve.py", "--route", "et_expand_gpt4o", "--variants", "large-3072",
                                "small-1536", "--pooling", "mean"]),
        ("E9 RAG gpt4o large", [PY, "rerank.py", "--route", "et_expand_gpt4o", "--llm", "gpt4o", "--variant",
                                "large-3072", "--k", "30", "--prompt", "rerank.v2", *TEST]),
        ("E9 RAG gpt4o small", [PY, "rerank.py", "--route", "et_expand_gpt4o", "--llm", "gpt4o", "--variant",
                                "small-1536", "--k", "30", "--prompt", "rerank.v2", *TEST]),
    ],
    # E7: appendix variant that may return up to two concepts (prompt frozen after one fold-1 round).
    "day3": [
        (f"E7 up-to-2 {route}", [PY, "rerank.py", "--route", route, "--variant", "large-3072", "--k", "30",
                                 "--prompt", "rerank_multi.v1", "--reasoning-effort", "low", *TEST])
        for route in ["et_expand", "en_expand"]
    ],
    # E10: smaller model (gpt-4o-mini) for expansion and choice, expanded Estonian, like the GPT-4o rows (E9).
    "day4": [
        ("E10 gpt4omini expansion", [PY, "llm_queries.py", "--prompt", "et_expand.v2", "--llm", "gpt4omini", *TEST,
                                     "--out", str(WORK_DIR / "queries/et_expand_gpt4omini.csv")]),
        ("E10 gpt4omini retrieval", [PY, "retrieve.py", "--route", "et_expand_gpt4omini", "--variants", "large-3072",
                                     "small-1536", "--pooling", "mean"]),
        ("E10 RAG gpt4omini large", [PY, "rerank.py", "--route", "et_expand_gpt4omini", "--llm", "gpt4omini",
                                     "--variant", "large-3072", "--k", "30", "--prompt", "rerank.v2", *TEST]),
        ("E10 RAG gpt4omini small", [PY, "rerank.py", "--route", "et_expand_gpt4omini", "--llm", "gpt4omini",
                                     "--variant", "small-1536", "--k", "30", "--prompt", "rerank.v2", *TEST]),
    ],
    # E3: two end-to-end repeats of the main configurations (new expansions, search, choice), ~2,044 calls each.
    "day2b": [
        step
        for rep in ["r2", "r3"]
        for step in [
            (f"E3 {rep} expansion et", [PY, "llm_queries.py", "--prompt", "et_expand.v2", *TEST,
                                        "--out", str(WORK_DIR / f"queries/et_expand_{rep}.csv")]),
            (f"E3 {rep} expansion en", [PY, "llm_queries.py", "--prompt", "en_expand.v2", *TEST,
                                        "--out", str(WORK_DIR / f"queries/en_expand_{rep}.csv")]),
            (f"E3 {rep} retrieval", [PY, "retrieve.py", "--route", f"et_expand_{rep}", f"en_expand_{rep}",
                                     "--variants", "large-3072", "--pooling", "mean"]),
            (f"E3 {rep} RAG et", [PY, "rerank.py", "--route", f"et_expand_{rep}", *MAIN, *TEST]),
            (f"E3 {rep} RAG en", [PY, "rerank.py", "--route", f"en_expand_{rep}", *MAIN, *TEST]),
        ]
    ],
}


def log(message: str):
    print(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {message}", flush=True)


def endpoint_reachable() -> bool:
    host = urlparse(AZURE_ENDPOINT).hostname
    try:
        socket.create_connection((host, 443), timeout=10).close()
        return True
    except OSError:
        return False


def run_step(name: str, command: list[str]) -> int:
    wait = 60
    while True:
        log(f"start: {name}")
        code = subprocess.call(command, cwd=Path(__file__).resolve().parent)
        if code != EXIT_NETWORK:
            return code
        log(f"network problem during '{name}'; waiting for the connection")
        while True:
            time.sleep(wait)
            if endpoint_reachable():
                break
            wait = min(wait * 2, 300)
            log(f"endpoint still unreachable; next check in {wait}s")
        wait = 60
        log("connection back; resuming")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("plan", choices=PLANS)
    parser.add_argument("--steps", nargs="*", help="only steps whose name starts with one of these")
    args = parser.parse_args()

    steps = [(n, c) for n, c in PLANS[args.plan] if not args.steps or any(n.startswith(s) for s in args.steps)]
    log(f"plan {args.plan}: {len(steps)} steps")
    for name, command in steps:
        code = run_step(name, command)
        if code == EXIT_QUOTA:
            log(f"quota exhausted during '{name}'; run the same command again after the quota resets")
            sys.exit(EXIT_QUOTA)
        if code != 0:
            log(f"'{name}' failed with exit code {code}; stopping")
            sys.exit(code)
        log(f"done: {name}")
    log(f"plan {args.plan} complete")


if __name__ == "__main__":
    main()
