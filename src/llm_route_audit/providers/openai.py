"""Replay requests on OpenAI models, or on any server that speaks OpenAI's Chat Completions API.

Candidates are named "openai/<model>", e.g. "openai/gpt-6-luna". For another
OpenAI-compatible server (Groq, Together, vLLM, LM Studio, ...), also set `base_url`, and
`api_key_env` if its key lives in a different environment variable.
"""

import os
from typing import Any

from llm_route_audit.candidates import Candidate
from llm_route_audit.providers.base import Completion, ProviderError
from llm_route_audit.providers.http import HTTPFailure, post_json
from llm_route_audit.providers.tooling import (
    openai_messages,
    openai_tools,
    parse_openai_tool_calls,
)
from llm_route_audit.records import Message, ToolDef

OPENAI_URL = "https://api.openai.com/v1"
DEFAULT_KEY_ENV = "OPENAI_API_KEY"
STATUS_BY_FINISH = {"length": "truncated", "content_filter": "refusal"}


def build_request(
    candidate: Candidate, messages: list[Message], tools: list[ToolDef] | None = None
) -> dict[str, Any]:
    request: dict[str, Any] = {
        "model": candidate.api_model,
        "messages": openai_messages(messages),
    }
    if tools:
        request["tools"] = openai_tools(tools)
    # OpenAI's own API wants max_completion_tokens; most compatible servers still use max_tokens.
    limit_field = "max_tokens" if candidate.base_url else "max_completion_tokens"
    request[limit_field] = candidate.max_tokens
    if candidate.effort:
        request["reasoning_effort"] = candidate.effort
    return request


def parse_response(data: dict[str, Any]) -> Completion:
    """OpenAI usage counts cached input inside prompt_tokens; split it back out."""
    choice = (data.get("choices") or [{}])[0]
    message = choice.get("message") or {}
    usage = data.get("usage") or {}
    cached = (usage.get("prompt_tokens_details") or {}).get("cached_tokens") or 0
    status = (
        "refusal"
        if message.get("refusal")
        else STATUS_BY_FINISH.get(choice.get("finish_reason"), "ok")
    )
    return Completion(
        text=message.get("content") or message.get("refusal") or "",
        input_tokens=max(0, (usage.get("prompt_tokens") or 0) - cached),
        output_tokens=usage.get("completion_tokens") or 0,
        cache_read_tokens=cached,
        status=status,
        tool_calls=parse_openai_tool_calls(message.get("tool_calls")),
        served_model=data.get("model"),
    )


def _out_of_credit(status: int, body: Any) -> bool:
    error = body.get("error") if isinstance(body, dict) else None
    return status == 429 and isinstance(error, dict) and error.get("code") == "insufficient_quota"


class OpenAIProvider:
    def __init__(self) -> None:
        self._sleep = None  # tests can replace the backoff sleep

    def complete(
        self, candidate: Candidate, messages: list[Message], tools: list[ToolDef] | None = None
    ) -> Completion:
        base_url = (candidate.base_url or OPENAI_URL).rstrip("/")
        key_env = candidate.api_key_env or DEFAULT_KEY_ENV
        key = os.environ.get(key_env)
        if not key and not candidate.base_url:
            raise ProviderError(
                f"{key_env} is not set. Add it to the .env file in your project folder.",
                fatal=True,
            )
        headers = {"Authorization": f"Bearer {key}"} if key else {}
        extra = {"sleep": self._sleep} if self._sleep else {}
        try:
            data = post_json(
                f"{base_url}/chat/completions",
                build_request(candidate, messages, tools),
                headers,
                should_retry=lambda status, body: not _out_of_credit(status, body),
                **extra,
            )
        except HTTPFailure as e:
            if e.status == 401:
                raise ProviderError(f"The API rejected the key in {key_env}.", fatal=True) from e
            if e.status is not None and _out_of_credit(e.status, e.body):
                raise ProviderError(f"Out of API credit: {e}", fatal=True) from e
            if e.status in (400, 404):
                raise ProviderError(f"{candidate.label}: {e}", disable=True) from e
            if e.status is None and "could not connect" in str(e):
                raise ProviderError(f"Could not reach {base_url}: {e}", fatal=True) from e
            raise ProviderError(f"API error {e.status}: {e}") from e
        return parse_response(data)
