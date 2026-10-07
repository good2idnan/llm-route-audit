"""Import Langfuse generations.

Accepts observations from the Langfuse API (GET /api/public/v2/observations, or the older
/api/public/observations), a JSON export from the Langfuse UI, or blob-storage exports:
a .json/.jsonl file or a folder of them. Only GENERATION observations become records.

Task types come from trace tags such as "task:classify_ticket", or from each generation's
name when `task_from_name` is set. Agent steps keep their tool calls, tool results and tool
definitions, and the steps of one trace (or session) form one agent session.
"""

from pathlib import Path
from typing import Any

from pydantic import ValidationError

from llm_route_audit.ingest.common import (
    DEFAULT_TASK_TAG_PREFIX,
    NO_RESPONSE,
    NOT_JSON,
    ImportResult,
    Skip,
    chat_messages,
    collect,
    is_unreadable,
    read_items,
    reply_of,
    tagged_task,
    tool_defs,
)
from llm_route_audit.records import LogRecord

# Keys Langfuse integrations use for cached input tokens in usageDetails.
CACHE_READ_KEYS = ("cache_read_input_tokens", "input_cached_tokens", "input_cache_read")


def _input(raw: Any) -> tuple[list[dict[str, Any]], list[dict[str, Any]] | None]:
    """(messages, tool definitions). Integrations log tools next to the messages."""
    tools = None
    if isinstance(raw, dict) and isinstance(raw.get("messages"), list):
        tools = tool_defs(raw.get("tools"))
        system = raw.get("system")  # Anthropic keeps the system prompt apart
        raw = ([{"role": "system", "content": system}] if system else []) + raw["messages"]
    if isinstance(raw, str) and raw:
        return [{"role": "user", "content": raw}], tools
    return chat_messages(raw, "no input logged for this generation"), tools


def _output(raw: Any) -> tuple[str, list[dict[str, Any]] | None]:
    if isinstance(raw, dict) and raw.get("choices"):
        raw = (raw["choices"][0] or {}).get("message") or {}
    text, calls = reply_of(raw)
    if not text and not calls:
        raise Skip(NO_RESPONSE)
    return text, calls


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

    messages, tools = _input(observation.get("input"))
    text, calls = _output(observation.get("output"))
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
                "response_tool_calls": calls,
                "tools": tools,
                "session_id": observation.get("sessionId") or observation.get("traceId"),
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
