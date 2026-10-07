"""Read routeaudit's native JSONL log format (one LogRecord per line)."""

import json
from dataclasses import dataclass, field
from pathlib import Path

from pydantic import ValidationError

from routeaudit.records import LogRecord


@dataclass
class LineError:
    line: int
    message: str


@dataclass
class LoadResult:
    records: list[LogRecord] = field(default_factory=list)
    errors: list[LineError] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors


def load_jsonl(path: str | Path) -> LoadResult:
    """Parse every line, collecting errors with line numbers instead of stopping at the first."""
    result = LoadResult()
    seen_ids: dict[str, int] = {}
    with Path(path).open(encoding="utf-8") as f:
        for lineno, raw in enumerate(f, start=1):
            if not raw.strip():
                continue
            try:
                data = json.loads(raw)
            except json.JSONDecodeError as e:
                result.errors.append(LineError(lineno, f"invalid JSON: {e.msg}"))
                continue
            try:
                record = LogRecord.model_validate(data)
            except ValidationError as e:
                result.errors.append(LineError(lineno, _describe(e)))
                continue
            if record.id in seen_ids:
                result.errors.append(
                    LineError(
                        lineno,
                        f"duplicate id '{record.id}' (first seen on line {seen_ids[record.id]})",
                    )
                )
                continue
            seen_ids[record.id] = lineno
            result.records.append(record)
    return result


def _describe(err: ValidationError) -> str:
    parts = []
    for e in err.errors():
        loc = ".".join(str(p) for p in e["loc"]) or "record"
        parts.append(f"{loc}: {e['msg']}")
    return "; ".join(parts)
