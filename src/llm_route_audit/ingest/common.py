"""Shared pieces for log importers: reading files, turning message content into text, and
collecting results with skip reasons."""

import json
from collections import Counter
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from llm_route_audit.providers.tooling import parse_arguments
from llm_route_audit.records import LogRecord

DEFAULT_TASK_TAG_PREFIX = "task:"
CHAT_ROLES = {"system", "user", "assistant", "tool"}
ROLE_ALIASES = {"developer": "system"}  # OpenAI's newer name for the system role
NOT_JSON = "not valid JSON"
NOT_TEXT = "contains images or other non-text content"
UNSUPPORTED_ROLE = "unsupported message role (only system, user, assistant and tool)"
NO_RESPONSE = "no response logged (neither text nor tool calls)"


class Skip(Exception):
    """An item that can't become a replayable record; the message says why."""


@dataclass
class ImportResult:
    records: list[LogRecord] = field(default_factory=list)
    skipped: Counter = field(default_factory=Counter)
    unreadable: list[str] = field(default_factory=list)


def read_items(source: Path) -> Iterator[tuple[str, Any]]:
    """Yield (where, item) pairs from a .json/.jsonl file or a folder of them.

    A .json file may hold one item, a list, or an API response that wraps its list in "data".
    """
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
                        where = f"{path.name}:{lineno}"
                        yield from _items(_parse(line, where), where)
        else:
            yield from _items(_parse(path.read_text(encoding="utf-8"), path.name), path.name)


def _items(data: Any, where: str) -> Iterator[tuple[str, Any]]:
    if isinstance(data, dict) and isinstance(data.get("data"), list):
        data = data["data"]
    if isinstance(data, list):
        for i, item in enumerate(data, start=1):
            yield f"{where}#{i}", item
    else:
        yield where, data


def _parse(text: str, where: str) -> Any:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {"__unreadable__": where}


def is_unreadable(item: Any) -> bool:
    return isinstance(item, dict) and "__unreadable__" in item


def text_of(content: Any) -> str:
    """Message content as plain text.

    Handles plain strings and lists of text parts, whether the part keeps its text under
    "text" (OpenAI, Anthropic) or "content" (OpenTelemetry). Raises Skip for images, audio
    and other non-text parts.
    """
    if isinstance(content, str):
        return content
    if content is None:
        return ""
    if isinstance(content, list):
        pieces = []
        for part in content:
            if isinstance(part, str):
                pieces.append(part)
            elif isinstance(part, dict) and part.get("type") == "text":
                pieces.append(str(part.get("text") or part.get("content") or ""))
            else:
                raise Skip(NOT_TEXT)
        return "".join(pieces)
    raise Skip("unrecognised message content")


def _block_type(part: Any) -> Any:
    return part.get("type") if isinstance(part, dict) else None


def tool_calls_of(raw: Any) -> list[dict[str, Any]] | None:
    """Tool calls as record dicts, from OpenAI ({id, function: {name, arguments}}), Anthropic
    tool_use blocks ({id, name, input}) or OpenTelemetry tool_call parts ({id, name,
    arguments}). Arguments may be a JSON string or already parsed."""
    if not raw:
        return None
    if not isinstance(raw, list):
        raise Skip("unrecognised tool calls")
    calls = []
    for item in raw:
        spec = item.get("function") if isinstance(item, dict) else None
        spec = spec if isinstance(spec, dict) else item
        if not isinstance(spec, dict) or not spec.get("name"):
            raise Skip("tool call without a tool name")
        arguments = spec.get("arguments", spec.get("input"))
        calls.append(
            {
                "id": item.get("id"),
                "name": str(spec["name"]),
                "arguments": parse_arguments(arguments),
            }
        )
    return calls


def tool_defs(raw: Any) -> list[dict[str, Any]] | None:
    """Tool definitions as record dicts, from OpenAI ({type: function, function: {...}}),
    Anthropic ({name, description, input_schema}) or flat ({name, description, parameters})
    form. Entries without a name are left out."""
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except json.JSONDecodeError:
            raise Skip("tool definitions are not valid JSON") from None
    if not isinstance(raw, list):
        return None
    tools = []
    for item in raw:
        spec = item.get("function") if isinstance(item, dict) else None
        spec = spec if isinstance(spec, dict) else item
        if not isinstance(spec, dict) or not spec.get("name"):
            continue
        parameters = spec.get("parameters") or spec.get("input_schema")
        if isinstance(parameters, str):
            parameters = json.loads(parameters) if parameters.strip() else None
        tool = {"name": str(spec["name"]), "description": str(spec.get("description") or "")}
        if isinstance(parameters, dict):
            tool["parameters"] = parameters
        tools.append(tool)
    return tools or None


def tool_result(call_id: Any, content: Any, name: Any = None) -> dict[str, Any]:
    """A "tool" message carrying one tool's result."""
    if not isinstance(content, str | list | type(None)):
        content = json.dumps(content, ensure_ascii=False)
    message: dict[str, Any] = {"role": "tool", "content": text_of(content)}
    if call_id:
        message["tool_call_id"] = str(call_id)
    if name:
        message["name"] = str(name)
    return message


def reply_of(message: Any) -> tuple[str, list[dict[str, Any]] | None]:
    """An assistant reply's text and tool calls. Takes an OpenAI message ({content,
    tool_calls}), an Anthropic content list (text and tool_use blocks) or plain text."""
    calls = None
    content = message
    if isinstance(message, dict):
        calls = tool_calls_of(message.get("tool_calls"))
        content = message.get("content")
    if isinstance(content, list):
        uses = [p for p in content if _block_type(p) == "tool_use"]
        if uses:
            calls = (calls or []) + (tool_calls_of(uses) or [])
            content = [p for p in content if _block_type(p) != "tool_use"]
    return text_of(content), calls


def chat_messages(raw: Any, missing: str) -> list[dict[str, Any]]:
    """Chat messages as record dicts, tool calls and tool results included.

    Takes OpenAI-style messages (tool calls on the assistant message, results in "tool"
    messages) and Anthropic-style content lists (tool_use blocks from the assistant,
    tool_result blocks in the next user turn). Raises Skip with `missing` when there are
    none, and for images or unsupported roles.
    """
    if not isinstance(raw, list) or not raw:
        raise Skip(missing)
    messages: list[dict[str, Any]] = []
    for m in raw:
        if not isinstance(m, dict):
            raise Skip("unrecognised message")
        role = ROLE_ALIASES.get(m.get("role"), m.get("role"))
        if role not in CHAT_ROLES:
            raise Skip(UNSUPPORTED_ROLE)
        content = m.get("content")
        if role == "tool":
            messages.append(tool_result(m.get("tool_call_id"), content, m.get("name")))
            continue
        if isinstance(content, list) and any(_block_type(p) == "tool_result" for p in content):
            for part in content:
                if _block_type(part) == "tool_result":
                    messages.append(tool_result(part.get("tool_use_id"), part.get("content")))
            rest = [p for p in content if _block_type(p) != "tool_result"]
            if rest:
                messages.append({"role": role, "content": text_of(rest)})
            continue
        text, calls = reply_of(m)
        message: dict[str, Any] = {"role": role, "content": text}
        if calls:
            if role != "assistant":
                raise Skip("tool calls outside an assistant message")
            message["tool_calls"] = calls
        messages.append(message)
    return messages


def tagged_task(tags: Any, prefix: str) -> str | None:
    """The task type from the first tag like "task:classify_ticket"."""
    for tag in tags or []:
        if isinstance(tag, str) and tag.startswith(prefix) and len(tag) > len(prefix):
            return tag[len(prefix) :]
    return None


def collect(items: Iterator[tuple[str, Any]], convert: Callable[[Any], LogRecord]) -> ImportResult:
    """Convert every item, counting skips by reason and dropping duplicate ids."""
    result = ImportResult()
    seen: set[str] = set()
    for where, item in items:
        try:
            record = convert(item)
        except Skip as reason:
            result.skipped[str(reason)] += 1
            if str(reason) == NOT_JSON:
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
