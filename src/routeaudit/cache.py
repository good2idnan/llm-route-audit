"""SQLite cache of replayed completions, so an identical request is never paid for twice."""

import hashlib
import json
import sqlite3
from datetime import UTC, datetime
from pathlib import Path

from routeaudit.candidates import Candidate
from routeaudit.providers.base import Completion
from routeaudit.records import Message

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
    created_at TEXT NOT NULL
)
"""


def request_key(candidate: Candidate, messages: list[Message]) -> str:
    """Everything that changes the answer goes into the key; nothing else does."""
    payload = {
        "provider": candidate.provider,
        "model": candidate.api_model,
        "effort": candidate.effort,
        "max_tokens": candidate.max_tokens,
        "messages": [[m.role, m.content] for m in messages],
    }
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

    def get(self, key: str) -> tuple[Completion, float | None] | None:
        row = self._db.execute(
            "SELECT text, input_tokens, output_tokens, cache_read_tokens, cache_write_tokens, "
            "status, latency_ms FROM completions WHERE key = ?",
            (key,),
        ).fetchone()
        if row is None:
            return None
        text, inp, out, cread, cwrite, status, latency = row
        return Completion(text, inp, out, cread, cwrite, status), latency

    def put(
        self, key: str, candidate: Candidate, completion: Completion, latency_ms: float | None
    ) -> None:
        self._db.execute(
            "INSERT OR REPLACE INTO completions VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
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
            ),
        )
        self._db.commit()

    def close(self) -> None:
        self._db.close()
