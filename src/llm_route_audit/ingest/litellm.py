"""Import LiteLLM logs: the StandardLoggingPayload that LiteLLM's logging callbacks write.

Accepts a .jsonl file (one payload per line), a .json file (one payload or a list), or a
folder of such files. Requests that can't be replayed faithfully are skipped and counted:
failures, LiteLLM cache hits and images. Agent steps keep their tool calls, tool results and
tool definitions; steps are grouped into sessions by `litellm_session_id` (or the trace id).

Task types come from request tags: a tag like "task:classify_ticket" sets the task type.
"""

from datetime import UTC, datetime
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
    write_records,
)
from llm_route_audit.records import LogRecord

__all__ = ["DEFAULT_TASK_TAG_PREFIX", "Skip", "convert", "import_litellm", "write_records"]


def _reply(raw: Any) -> tuple[str, list[dict[str, Any]] | None]:
    if isinstance(raw, str) and raw:
        return raw, None
    if isinstance(raw, dict):
        choices = raw.get("choices") or []
        if choices:
            text, calls = reply_of((choices[0] or {}).get("message") or {})
            if text or calls:
                return text, calls
    raise Skip(NO_RESPONSE)


def _session(payload: dict[str, Any]) -> str | None:
    metadata = payload.get("metadata") or {}
    for value in (
        payload.get("session_id"),
        metadata.get("session_id") if isinstance(metadata, dict) else None,
        payload.get("trace_id"),
    ):
        if value:
            return str(value)
    return None


def _cached_tokens(response: Any) -> int:
    usage = response.get("usage") if isinstance(response, dict) else None
    if not isinstance(usage, dict):
        return 0
    details = usage.get("prompt_tokens_details") or {}
    return int(details.get("cached_tokens") or usage.get("cache_read_input_tokens") or 0)


def convert(payload: Any, task_tag_prefix: str = DEFAULT_TASK_TAG_PREFIX) -> LogRecord:
    """One LiteLLM payload as a LogRecord, or Skip with the reason."""
    if not isinstance(payload, dict) or is_unreadable(payload):
        raise Skip(NOT_JSON)
    if payload.get("status") not in (None, "success"):
        raise Skip("failed request")
    if payload.get("cache_hit"):
        raise Skip("answered from LiteLLM's cache, not by a model")

    messages = chat_messages(
        payload.get("messages"), "no chat messages logged (turn on prompt logging in LiteLLM)"
    )
    response = payload.get("response")
    text, calls = _reply(response)
    parameters = payload.get("model_parameters") or {}
    tools = tool_defs(payload.get("tools") or parameters.get("tools"))
    cached = _cached_tokens(response)
    prompt_tokens = payload.get("prompt_tokens")
    start, end = payload.get("startTime"), payload.get("endTime")
    model = str(payload.get("model") or "").removeprefix("anthropic/")
    if not model:
        raise Skip("no model name")

    try:
        return LogRecord.model_validate(
            {
                "id": str(payload.get("id") or ""),
                "timestamp": datetime.fromtimestamp(float(start), UTC),
                "model": model,
                "task_type": tagged_task(payload.get("request_tags"), task_tag_prefix),
                "messages": messages,
                "response": text,
                "response_tool_calls": calls,
                "tools": tools,
                "session_id": _session(payload),
                "input_tokens": max(0, prompt_tokens - cached)
                if prompt_tokens is not None
                else None,
                "output_tokens": payload.get("completion_tokens"),
                "cache_read_tokens": cached or None,
                "latency_ms": (float(end) - float(start)) * 1000 if end is not None else None,
                "metadata": {
                    "source": "litellm",
                    "model_group": payload.get("model_group"),
                    "logged_cost": payload.get("response_cost"),
                },
            }
        )
    except (ValidationError, TypeError, ValueError) as e:
        raise Skip(f"missing or invalid fields ({type(e).__name__})") from None


def import_litellm(
    source: str | Path, task_tag_prefix: str = DEFAULT_TASK_TAG_PREFIX
) -> ImportResult:
    return collect(read_items(Path(source)), lambda item: convert(item, task_tag_prefix))
