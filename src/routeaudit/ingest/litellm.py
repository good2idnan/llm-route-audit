"""Import LiteLLM logs: the StandardLoggingPayload that LiteLLM's logging callbacks write.

Accepts a .jsonl file (one payload per line), a .json file (one payload or a list), or a
folder of such files. Requests that can't be replayed faithfully yet are skipped and counted:
failures, LiteLLM cache hits, images and tool calls (agent turns come later).

Task types come from request tags: a tag like "task:classify_ticket" sets the task type.
"""

import json
from collections import Counter
from collections.abc import Iterator
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from routeaudit.records import LogRecord

DEFAULT_TASK_TAG_PREFIX = "task:"
CHAT_ROLES = {"system", "user", "assistant"}


class Skip(Exception):
    """A payload that can't become a replayable record; the message says why."""


@dataclass
class ImportResult:
    records: list[LogRecord] = field(default_factory=list)
    skipped: Counter = field(default_factory=Counter)
    unreadable: list[str] = field(default_factory=list)


def read_payloads(source: Path) -> Iterator[tuple[str, Any]]:
    """Yield (where, payload) pairs from a file or a folder of files."""
    files = sorted(
        p
        for p in ([source] if source.is_file() else source.rglob("*"))
        if p.suffix in {".json", ".jsonl"}
    )
    for path in files:
        if path.suffix == ".jsonl":
            with path.open(encoding="utf-8") as f:
                for lineno, line in enumerate(f, start=1):
                    if line.strip():
                        yield f"{path.name}:{lineno}", _parse(line, f"{path.name}:{lineno}")
        else:
            data = _parse(path.read_text(encoding="utf-8"), path.name)
            for i, item in enumerate(data if isinstance(data, list) else [data], start=1):
                yield f"{path.name}#{i}", item


def _parse(text: str, where: str) -> Any:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {"__unreadable__": where}


def _text(content: Any) -> str:
    """Message content as plain text. Raises Skip for images, audio and other parts."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, dict) and part.get("type") == "text":
                parts.append(part.get("text", ""))
            else:
                raise Skip("contains images or other non-text content")
        return "".join(parts)
    if content is None:
        return ""
    raise Skip("unrecognised message content")


def _messages(raw: Any) -> list[dict[str, str]]:
    if not isinstance(raw, list) or not raw:
        raise Skip("no chat messages logged (turn on prompt logging in LiteLLM)")
    messages = []
    for m in raw:
        if not isinstance(m, dict) or m.get("role") not in CHAT_ROLES:
            raise Skip("tool calls or unsupported roles (agent turns are not supported yet)")
        if m.get("tool_calls"):
            raise Skip("tool calls or unsupported roles (agent turns are not supported yet)")
        messages.append({"role": m["role"], "content": _text(m.get("content"))})
    return messages


def _response_text(raw: Any) -> str:
    if isinstance(raw, str):
        return raw
    if isinstance(raw, dict):
        choices = raw.get("choices") or []
        if choices:
            message = choices[0].get("message") or {}
            if message.get("tool_calls"):
                raise Skip("tool calls or unsupported roles (agent turns are not supported yet)")
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


def _task(tags: Any, prefix: str) -> str | None:
    for tag in tags or []:
        if isinstance(tag, str) and tag.startswith(prefix) and len(tag) > len(prefix):
            return tag[len(prefix) :]
    return None


def convert(payload: Any, task_tag_prefix: str = DEFAULT_TASK_TAG_PREFIX) -> LogRecord:
    """One LiteLLM payload as a LogRecord, or Skip with the reason."""
    if not isinstance(payload, dict) or "__unreadable__" in payload:
        raise Skip("not valid JSON")
    if payload.get("status") not in (None, "success"):
        raise Skip("failed request")
    if payload.get("cache_hit"):
        raise Skip("answered from LiteLLM's cache, not by a model")

    messages = _messages(payload.get("messages"))
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
                "task_type": _task(payload.get("request_tags"), task_tag_prefix),
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
    result = ImportResult()
    seen: set[str] = set()
    for where, payload in read_payloads(Path(source)):
        try:
            record = convert(payload, task_tag_prefix)
        except Skip as reason:
            result.skipped[str(reason)] += 1
            if str(reason) == "not valid JSON":
                result.unreadable.append(where)
            continue
        if record.id in seen:
            result.skipped["duplicate id"] += 1
            continue
        seen.add(record.id)
        result.records.append(record)
    result.records.sort(key=lambda r: r.timestamp)
    return result


def write_records(path: str | Path, records: list[LogRecord]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for record in records:
            f.write(record.model_dump_json(exclude_none=True) + "\n")
