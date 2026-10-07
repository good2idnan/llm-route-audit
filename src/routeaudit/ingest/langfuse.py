"""Import Langfuse generations.

Accepts observations from the Langfuse API (GET /api/public/v2/observations, or the older
/api/public/observations), a JSON export from the Langfuse UI, or blob-storage exports:
a .json/.jsonl file or a folder of them. Only GENERATION observations become records.

Task types come from trace tags such as "task:classify_ticket", or from each generation's
name when `task_from_name` is set.
"""

from pathlib import Path
from typing import Any

from pydantic import ValidationError

from routeaudit.ingest.common import (
    AGENT_TURN,
    DEFAULT_TASK_TAG_PREFIX,
    NOT_JSON,
    ImportResult,
    Skip,
    chat_messages,
    collect,
    is_unreadable,
    read_items,
    tagged_task,
    text_of,
)
from routeaudit.records import LogRecord

# Keys Langfuse integrations use for cached input tokens in usageDetails.
CACHE_READ_KEYS = ("cache_read_input_tokens", "input_cached_tokens", "input_cache_read")


def _input_messages(raw: Any) -> list[dict[str, str]]:
    if isinstance(raw, dict) and isinstance(raw.get("messages"), list):
        raw = raw["messages"]
    if isinstance(raw, str) and raw:
        return [{"role": "user", "content": raw}]
    return chat_messages(raw, "no input logged for this generation")


def _output_text(raw: Any) -> str:
    if isinstance(raw, dict):
        if raw.get("tool_calls"):
            raise Skip(AGENT_TURN)
        if raw.get("choices"):
            raw = (raw["choices"][0] or {}).get("message") or {}
            if raw.get("tool_calls"):
                raise Skip(AGENT_TURN)
        raw = raw.get("content")
    if isinstance(raw, list):
        if any(isinstance(p, dict) and p.get("type") in ("tool_use", "tool_call") for p in raw):
            raise Skip(AGENT_TURN)
    text = text_of(raw)
    if not text:
        raise Skip("no output logged for this generation")
    return text


def _usage(observation: dict[str, Any]) -> tuple[int | None, int | None, int]:
    details = observation.get("usageDetails") or {}
    usage = observation.get("usage") or {}
    cached = next((details[k] for k in CACHE_READ_KEYS if details.get(k)), 0)
    return (
        details.get("input", usage.get("input")),
        details.get("output", usage.get("output")),
        int(cached),
    )


def convert(
    observation: Any,
    task_tag_prefix: str = DEFAULT_TASK_TAG_PREFIX,
    task_from_name: bool = False,
) -> LogRecord:
    """One Langfuse observation as a LogRecord, or Skip with the reason."""
    if not isinstance(observation, dict) or is_unreadable(observation):
        raise Skip(NOT_JSON)
    if observation.get("type") != "GENERATION":
        raise Skip("not a model call (span or event)")
    if observation.get("level") == "ERROR":
        raise Skip("failed request")
    model = str(observation.get("model") or "").removeprefix("anthropic/")
    if not model:
        raise Skip("no model name")

    messages = _input_messages(observation.get("input"))
    text = _output_text(observation.get("output"))
    input_tokens, output_tokens, cached = _usage(observation)
    task = (
        observation.get("name")
        if task_from_name
        else tagged_task(observation.get("tags"), task_tag_prefix)
    )
    try:
        record = LogRecord.model_validate(
            {
                "id": str(observation.get("id") or ""),
                "timestamp": observation.get("startTime"),
                "model": model,
                "task_type": task or None,
                "messages": messages,
                "response": text,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "cache_read_tokens": cached or None,
                "metadata": {
                    "source": "langfuse",
                    "trace_id": observation.get("traceId"),
                    "name": observation.get("name"),
                    "logged_cost": observation.get("totalCost")
                    or observation.get("calculatedTotalCost"),
                },
            }
        )
    except (ValidationError, TypeError, ValueError) as e:
        raise Skip(f"missing or invalid fields ({type(e).__name__})") from None
    end = observation.get("endTime")
    if end:
        try:
            ended = LogRecord.model_validate(
                {"id": "x", "timestamp": end, "model": "x", "prompt": "x", "response": ""}
            ).timestamp
            record.latency_ms = max(0.0, (ended - record.timestamp).total_seconds() * 1000)
        except ValidationError:
            pass
    return record


def import_langfuse(
    source: str | Path,
    task_tag_prefix: str = DEFAULT_TASK_TAG_PREFIX,
    task_from_name: bool = False,
) -> ImportResult:
    return collect(
        read_items(Path(source)),
        lambda item: convert(item, task_tag_prefix, task_from_name),
    )
