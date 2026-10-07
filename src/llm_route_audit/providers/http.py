"""Small JSON-over-HTTP helper for providers that speak an HTTP API, with retries."""

import json
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from typing import Any

RETRY_STATUSES = frozenset({408, 409, 429, 500, 502, 503, 504})


class HTTPFailure(Exception):
    """A request that failed after any retries. `status` is None for network problems."""

    def __init__(self, message: str, status: int | None = None, body: Any = None) -> None:
        super().__init__(message)
        self.status = status
        self.body = body


def _error_body(e: urllib.error.HTTPError) -> tuple[str, Any]:
    raw = e.read().decode("utf-8", "replace")
    try:
        body = json.loads(raw)
    except ValueError:
        return raw[:200] or str(e), None
    error = body.get("error") if isinstance(body, dict) else None
    message = error.get("message") if isinstance(error, dict) else None
    return str(message or raw[:200] or e), body


def post_json(
    url: str,
    payload: dict[str, Any] | None,
    headers: dict[str, str],
    *,
    timeout: float = 600,
    max_retries: int = 3,
    retry_statuses: frozenset[int] = RETRY_STATUSES,
    should_retry: Callable[[int, Any], bool] | None = None,
    sleep: Callable[[float], None] = time.sleep,
) -> Any:
    """POST JSON (or GET, when `payload` is None) and return the decoded reply. Retries busy
    and server errors with backoff."""
    request = urllib.request.Request(
        url,
        data=None if payload is None else json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", **headers},
        method="GET" if payload is None else "POST",
    )
    for attempt in range(max_retries + 1):
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            message, body = _error_body(e)
            retry = e.code in retry_statuses and (
                should_retry is None or should_retry(e.code, body)
            )
            if not retry or attempt == max_retries:
                raise HTTPFailure(message, e.code, body) from e
        except TimeoutError as e:
            if attempt == max_retries:
                raise HTTPFailure(f"timed out after {timeout:.0f}s") from e
        except (urllib.error.URLError, ConnectionError) as e:
            raise HTTPFailure(f"could not connect ({e})") from e
        sleep(2**attempt)
    raise AssertionError("unreachable")
