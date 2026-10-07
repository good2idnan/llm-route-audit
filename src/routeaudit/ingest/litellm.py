"""Import LiteLLM logs: the StandardLoggingPayload that LiteLLM's logging callbacks write.

Accepts a .jsonl file (one payload per line), a .json file (one payload or a list), or a
folder of such files. Requests that can't be replayed faithfully yet are skipped and counted:
failures, LiteLLM cache hits, images and tool calls (agent turns come later).

Task types come from request tags: a tag like "task:classify_ticket" sets the task type.
"""

from datetime import UTC, datetime
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
    write_records,
)
from routeaudit.records import LogRecord

__all__ = ["DEFAULT_TASK_TAG_PREFIX", "Skip", "convert", "import_litellm", "write_records"]


def _response_text(raw: Any) -> str:
    if isinstance(raw, str):
        return raw
    if isinstance(raw, dict):
        choices = raw.get("choices") or []
        if choices:
            message = choices[0].get("message") or {}
            if message.get("tool_calls"):
                raise Skip(AGENT_TURN)
            text = message.get("content")
            if isinstance(text, str) and text:
                return text
    raise Skip("no response text logged")


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
    text = _response_text(response)
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
