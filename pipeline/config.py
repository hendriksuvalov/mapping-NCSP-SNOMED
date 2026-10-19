"""Shared paths and the Azure OpenAI client for the experiments."""
import os
from pathlib import Path

from openai import AzureOpenAI

ROOT = Path(__file__).resolve().parent.parent
ATHENA_DIR = ROOT / "data" / "vocabulary"      # OMOP vocabulary from athena.ohdsi.org (licensed; not in the repository)
GOLD_DIR = ROOT / "data" / "reference"         # reference mappings (NCSP_AUG26/) and NCSP names/hierarchy (NCSP/)
WORK_DIR = ROOT / "outputs"                    # everything the pipeline writes: splits, queries, index, runs, usagi
KEY_FILE = ROOT / ".azure_key"                 # or set AZURE_OPENAI_API_KEY


def work_path(path: Path) -> Path:
    """Relative paths given on the command line (e.g. index/large, runs/rerank/x.csv) are inside WORK_DIR."""
    return path if path.is_absolute() else WORK_DIR / path

# Azure OpenAI endpoint: set AZURE_OPENAI_ENDPOINT (it is not recorded in any output file).
AZURE_ENDPOINT = os.environ.get("AZURE_OPENAI_ENDPOINT", "")
AZURE_API_VERSION = "2024-12-01-preview"

EMBEDDING_DEPLOYMENTS = {"small": "text-embedding-3-small", "large": "text-embedding-3-large"}
EMBEDDING_DEPLOYMENT = EMBEDDING_DEPLOYMENTS["small"]
CHAT_DEPLOYMENTS = {"luna": "gpt-6-luna", "gpt4o": "gpt-4o", "gpt4omini": "gpt-4o-mini"}
REASONING_LLMS = {"luna"}          # models that accept reasoning_effort
CHAT_DEPLOYMENT = CHAT_DEPLOYMENTS["luna"]


def _read_key() -> str:
    if os.environ.get("AZURE_OPENAI_API_KEY"):
        return os.environ["AZURE_OPENAI_API_KEY"]
    text = KEY_FILE.read_text().strip()
    return text.split("=", 1)[1].strip().strip("\"'") if "=" in text else text


def azure_client() -> AzureOpenAI:
    if not AZURE_ENDPOINT:
        raise SystemExit("Set AZURE_OPENAI_ENDPOINT to your Azure OpenAI endpoint (and AZURE_OPENAI_API_KEY or .azure_key).")
    return AzureOpenAI(
        api_key=_read_key(),
        azure_endpoint=AZURE_ENDPOINT,
        api_version=AZURE_API_VERSION,
        max_retries=5,
    )


EXIT_QUOTA = 2    # run stopped: API call quota exhausted; rerun after it resets
EXIT_NETWORK = 3  # run stopped: no connection or endpoint unavailable; rerun when the network is back


def is_quota_error(error: Exception) -> bool:
    """The endpoint's call-volume quota is exhausted (HTTP 403); retrying is pointless until it resets."""
    return getattr(error, "status_code", None) == 403 and "quota" in str(error).lower()


def is_network_error(error: Exception) -> bool:
    """No connection, timeout, or server failure that persisted through the client's own retries."""
    import openai

    return isinstance(error, (openai.APIConnectionError, openai.InternalServerError))


def describe_error(error: Exception) -> str:
    """Short, provider-neutral description of an API error for the logs (no endpoint, region or resource names)."""
    name, text, status = type(error).__name__, str(error).lower(), getattr(error, "status_code", None)
    if is_quota_error(error):
        return f"{name}: quota exceeded"
    if status == 429 or "rate limit" in text:
        return f"{name}: rate limit exceeded"
    if "content management policy" in text or "content_filter" in text:
        return f"{name}: content filter"
    if is_network_error(error):
        return f"{name}: connection or server error"
    if status is not None:
        return f"{name}: HTTP {status}"
    return f"{name}: {str(error)[:200]}"  # local errors, e.g. an answer that is not valid JSON


def stop_reason(error: Exception) -> str | None:
    """'quota' or 'network' if the whole run should stop (and resume later) instead of marking the code failed."""
    if is_quota_error(error):
        return "quota"
    if is_network_error(error):
        return "network"
    return None


def read_log(path: Path) -> list[dict]:
    """Read a JSONL log, skipping a half-written last line left by an interrupted process."""
    import json

    records = []
    if path.exists():
        for line in path.read_text().splitlines():
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return records


def chat_options(llm: str, reasoning_effort: str) -> dict:
    """Model-specific request options: reasoning models take reasoning_effort, others do not."""
    options = {"model": CHAT_DEPLOYMENTS[llm], "response_format": {"type": "json_object"}}
    if llm in REASONING_LLMS:
        options["reasoning_effort"] = reasoning_effort
    return options


def reasoning_tokens(usage) -> int:
    details = getattr(usage, "completion_tokens_details", None)
    return (getattr(details, "reasoning_tokens", 0) or 0) if details else 0
