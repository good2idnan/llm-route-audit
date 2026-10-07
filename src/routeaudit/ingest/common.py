"""Shared pieces for log importers: reading files, turning message content into text, and
collecting results with skip reasons."""

import json
from collections import Counter
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from routeaudit.records import LogRecord

DEFAULT_TASK_TAG_PREFIX = "task:"
CHAT_ROLES = {"system", "user", "assistant"}
NOT_JSON = "not valid JSON"
NOT_TEXT = "contains images or other non-text content"
AGENT_TURN = "tool calls or unsupported roles (agent turns are not supported yet)"


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


def chat_messages(raw: Any, missing: str) -> list[dict[str, str]]:
    """OpenAI-style [{role, content}] messages as plain text. Raises Skip with `missing` when
    there are none, and for tool calls or other roles."""
    if not isinstance(raw, list) or not raw:
        raise Skip(missing)
    messages = []
    for m in raw:
        if not isinstance(m, dict) or m.get("role") not in CHAT_ROLES or m.get("tool_calls"):
            raise Skip(AGENT_TURN)
        messages.append({"role": m["role"], "content": text_of(m.get("content"))})
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
