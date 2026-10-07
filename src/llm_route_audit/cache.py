"""SQLite cache of replayed completions, so an identical request is never paid for twice."""

import hashlib
import json
import sqlite3
from datetime import UTC, datetime
from pathlib import Path

from llm_route_audit.candidates import Candidate
from llm_route_audit.providers.base import Completion
from llm_route_audit.records import Message, ToolCall, ToolDef

SCHEMA = """
CREATE TABLE IF NOT EXISTS completions (
    key TEXT PRIMARY KEY,
    model TEXT NOT NULL,
    effort TEXT,
    text TEXT NOT NULL,
    input_tokens INTEGER NOT NULL,
    output_tokens INTEGER NOT NULL,
    cache_read_tokens INTEGER NOT NULL,
    cache_write_tokens INTEGER NOT NULL,
    status TEXT NOT NULL,
    latency_ms REAL,
    created_at TEXT NOT NULL,
    cost REAL,
    tool_calls TEXT
)
"""


def _message_key(message: Message) -> list:
    """Plain messages keep their original short form, so existing cache entries still match."""
    extra = message.model_dump(exclude_none=True, exclude={"role", "content"})
    return [message.role, message.content, extra] if extra else [message.role, message.content]


def request_key(
    candidate: Candidate, messages: list[Message], tools: list[ToolDef] | None = None
) -> str:
    """Everything that changes the answer goes into the key; nothing else does."""
    payload: dict = {
        "provider": candidate.provider,
        "model": candidate.api_model,
        "effort": candidate.effort,
        "max_tokens": candidate.max_tokens,
        "messages": [_message_key(m) for m in messages],
    }
    if candidate.base_url:  # a different server can answer differently
        payload["base_url"] = candidate.base_url
    if tools:
        payload["tools"] = [t.model_dump() for t in tools]
    blob = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


class ResultCache:
    """Use from one thread only. The replay writes results from its main thread."""

    def __init__(self, path: str | Path) -> None:
        path = Path(path)
        if str(path) != ":memory:":
            path.parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(path)
        self._db.execute(SCHEMA)
        columns = {row[1] for row in self._db.execute("PRAGMA table_info(completions)")}
        if "cost" not in columns:  # caches made before provider-reported costs existed
            self._db.execute("ALTER TABLE completions ADD COLUMN cost REAL")
        if "tool_calls" not in columns:  # caches made before agent steps were supported
            self._db.execute("ALTER TABLE completions ADD COLUMN tool_calls TEXT")

    def get(self, key: str) -> tuple[Completion, float | None] | None:
        row = self._db.execute(
            "SELECT text, input_tokens, output_tokens, cache_read_tokens, cache_write_tokens, "
            "status, latency_ms, cost, tool_calls FROM completions WHERE key = ?",
            (key,),
        ).fetchone()
        if row is None:
            return None
        text, inp, out, cread, cwrite, status, latency, cost, calls = row
        tool_calls = [ToolCall.model_validate(c) for c in json.loads(calls)] if calls else None
        return Completion(text, inp, out, cread, cwrite, status, cost, tool_calls), latency

    def put(
        self, key: str, candidate: Candidate, completion: Completion, latency_ms: float | None
    ) -> None:
        self._db.execute(
            "INSERT OR REPLACE INTO completions (key, model, effort, text, input_tokens, "
            "output_tokens, cache_read_tokens, cache_write_tokens, status, latency_ms, "
            "created_at, cost, tool_calls) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                key,
                candidate.model,
                candidate.effort,
                completion.text,
                completion.input_tokens,
                completion.output_tokens,
                completion.cache_read_tokens,
                completion.cache_write_tokens,
                completion.status,
                latency_ms,
                datetime.now(UTC).isoformat(),
                completion.cost,
                json.dumps([c.model_dump() for c in completion.tool_calls])
                if completion.tool_calls
                else None,
            ),
        )
        self._db.commit()

    def close(self) -> None:
        self._db.close()
