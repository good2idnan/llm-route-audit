"""Replay requests on local models served by Ollama (free to run, useful for trying things out)."""

import json
import os
import urllib.error
import urllib.request
from typing import Any

from llm_route_audit.candidates import Candidate
from llm_route_audit.providers.base import Completion, ProviderError
from llm_route_audit.providers.tooling import (
    ollama_messages,
    openai_tools,
    parse_openai_tool_calls,
)
from llm_route_audit.records import Message, ToolDef

DEFAULT_HOST = "http://localhost:11434"
TIMEOUT_SECONDS = 600


def ollama_url() -> str:
    host = os.environ.get("OLLAMA_HOST") or DEFAULT_HOST
    if "://" not in host:
        host = f"http://{host}"
    return host.rstrip("/")


def build_request(
    candidate: Candidate, messages: list[Message], tools: list[ToolDef] | None = None
) -> dict[str, Any]:
    request: dict[str, Any] = {
        "model": candidate.api_model,
        "messages": ollama_messages(messages),
        "stream": False,
        "options": {"num_predict": candidate.max_tokens},
    }
    if tools:
        request["tools"] = openai_tools(tools)
    return request


def parse_response(data: dict[str, Any]) -> Completion:
    return Completion(
        text=data.get("message", {}).get("content", ""),
        input_tokens=data.get("prompt_eval_count", 0),
        output_tokens=data.get("eval_count", 0),
        status="truncated" if data.get("done_reason") == "length" else "ok",
        tool_calls=parse_openai_tool_calls(data.get("message", {}).get("tool_calls")),
    )


class OllamaProvider:
    def __init__(self, base_url: str | None = None) -> None:
        self.base_url = base_url or ollama_url()

    def complete(
        self, candidate: Candidate, messages: list[Message], tools: list[ToolDef] | None = None
    ) -> Completion:
        body = json.dumps(build_request(candidate, messages, tools)).encode("utf-8")
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
