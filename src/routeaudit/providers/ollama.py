"""Replay requests on local models served by Ollama (free to run, useful for trying things out)."""

import json
import os
import urllib.error
import urllib.request
from typing import Any

from routeaudit.candidates import Candidate
from routeaudit.providers.base import Completion, ProviderError
from routeaudit.records import Message

DEFAULT_HOST = "http://localhost:11434"
TIMEOUT_SECONDS = 600


def ollama_url() -> str:
    host = os.environ.get("OLLAMA_HOST") or DEFAULT_HOST
    if "://" not in host:
        host = f"http://{host}"
    return host.rstrip("/")


def build_request(candidate: Candidate, messages: list[Message]) -> dict[str, Any]:
    return {
        "model": candidate.api_model,
        "messages": [{"role": m.role, "content": m.content} for m in messages],
        "stream": False,
        "options": {"num_predict": candidate.max_tokens},
    }


def parse_response(data: dict[str, Any]) -> Completion:
    return Completion(
        text=data.get("message", {}).get("content", ""),
        input_tokens=data.get("prompt_eval_count", 0),
        output_tokens=data.get("eval_count", 0),
        status="truncated" if data.get("done_reason") == "length" else "ok",
    )


class OllamaProvider:
    def __init__(self, base_url: str | None = None) -> None:
        self.base_url = base_url or ollama_url()

    def complete(self, candidate: Candidate, messages: list[Message]) -> Completion:
        body = json.dumps(build_request(candidate, messages)).encode("utf-8")
        request = urllib.request.Request(
            f"{self.base_url}/api/chat",
            data=body,
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(request, timeout=TIMEOUT_SECONDS) as response:
                data = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            detail = e.read().decode("utf-8", "replace")[:200]
            if e.code == 404:
                raise ProviderError(
                    f"{candidate.label}: model not found. Pull it with `ollama pull "
                    f"{candidate.api_model}`.",
                    disable=True,
                ) from e
            raise ProviderError(f"Ollama error {e.code}: {detail}") from e
        except TimeoutError as e:
            raise ProviderError(f"Ollama took longer than {TIMEOUT_SECONDS}s to answer") from e
        except (urllib.error.URLError, ConnectionError) as e:
            raise ProviderError(
                f"Could not reach Ollama at {self.base_url}. Is it running?", fatal=True
            ) from e
        return parse_response(data)
