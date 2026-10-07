"""Replay requests through OpenRouter, one API key for many providers' models.

Candidates are named "openrouter/<model id>", e.g. "openrouter/anthropic/claude-haiku-4.5".
OpenRouter reports what each call actually cost, and llm-route-audit uses that figure.
"""

import json
import os
import time
import urllib.error
import urllib.request
from typing import Any

from llm_route_audit.candidates import Candidate
from llm_route_audit.costs import ModelPrice
from llm_route_audit.providers.base import Completion, ProviderError
from llm_route_audit.providers.tooling import (
    openai_messages,
    openai_tools,
    parse_openai_tool_calls,
)
from llm_route_audit.records import Message, ToolDef

BASE_URL = "https://openrouter.ai/api/v1"
TIMEOUT_SECONDS = 600
MAX_RETRIES = 3
RETRY_STATUSES = {408, 429, 500, 502, 503, 504}
STATUS_BY_FINISH = {"length": "truncated", "content_filter": "refusal"}
PER_MILLION = 1_000_000


def build_request(
    candidate: Candidate, messages: list[Message], tools: list[ToolDef] | None = None
) -> dict[str, Any]:
    request: dict[str, Any] = {
        "model": candidate.api_model,
        "messages": openai_messages(messages),
        "max_tokens": candidate.max_tokens,
    }
    if tools:
        request["tools"] = openai_tools(tools)
    if candidate.effort:
        request["reasoning"] = {"effort": candidate.effort}
    return request


def parse_response(data: dict[str, Any]) -> Completion:
    """OpenAI-style usage counts cached input inside prompt_tokens; split it back out."""
    if data.get("error"):
        raise ProviderError(f"OpenRouter error: {data['error'].get('message', data['error'])}")
    choice = data["choices"][0]
    if choice.get("error"):
        raise ProviderError(f"OpenRouter error: {choice['error'].get('message', choice['error'])}")
    usage = data.get("usage") or {}
    cached = (usage.get("prompt_tokens_details") or {}).get("cached_tokens") or 0
    finish = choice.get("finish_reason")
    status = "refusal" if choice.get("native_finish_reason") == "refusal" else None
    return Completion(
        text=(choice.get("message") or {}).get("content") or "",
        input_tokens=max(0, (usage.get("prompt_tokens") or 0) - cached),
        output_tokens=usage.get("completion_tokens") or 0,
        cache_read_tokens=cached,
        status=status or STATUS_BY_FINISH.get(finish, "ok"),
        cost=usage.get("cost"),
        tool_calls=parse_openai_tool_calls((choice.get("message") or {}).get("tool_calls")),
        served_model=data.get("model"),
    )


def _error_message(e: urllib.error.HTTPError) -> str:
    body = e.read().decode("utf-8", "replace")
    try:
        return json.loads(body)["error"]["message"]
    except (ValueError, KeyError, TypeError):
        return body[:200] or str(e)


class OpenRouterProvider:
    def __init__(self, api_key: str | None = None, base_url: str = BASE_URL) -> None:
        self._api_key = api_key
        self.base_url = base_url
        self._sleep = time.sleep

    def _key(self) -> str:
        key = self._api_key or os.environ.get("OPENROUTER_API_KEY")
        if not key:
            raise ProviderError(
                "OPENROUTER_API_KEY is not set. Add it to the .env file in your project folder.",
                fatal=True,
            )
        return key

    def complete(
        self, candidate: Candidate, messages: list[Message], tools: list[ToolDef] | None = None
    ) -> Completion:
        request = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(build_request(candidate, messages, tools)).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self._key()}",
                "Content-Type": "application/json",
                "X-OpenRouter-Title": "llm-route-audit",
            },
        )
        for attempt in range(MAX_RETRIES + 1):
            try:
                with urllib.request.urlopen(request, timeout=TIMEOUT_SECONDS) as response:
                    return parse_response(json.loads(response.read().decode("utf-8")))
            except urllib.error.HTTPError as e:
                message = _error_message(e)
                if e.code == 401:
                    raise ProviderError(
                        "OpenRouter rejected the API key. Check OPENROUTER_API_KEY.", fatal=True
                    ) from e
                if e.code == 402:
                    raise ProviderError(
                        f"OpenRouter: not enough credits ({message})", fatal=True
                    ) from e
                if e.code in (400, 404):
                    raise ProviderError(f"{candidate.label}: {message}", disable=True) from e
                if e.code not in RETRY_STATUSES or attempt == MAX_RETRIES:
                    raise ProviderError(f"OpenRouter error {e.code}: {message}") from e
            except TimeoutError as e:
                if attempt == MAX_RETRIES:
                    raise ProviderError("OpenRouter timed out") from e
            except (urllib.error.URLError, ConnectionError) as e:
                raise ProviderError("Could not reach OpenRouter (network error)", fatal=True) from e
            self._sleep(2**attempt)
        raise AssertionError("unreachable")


def fetch_prices(model_ids: list[str], base_url: str = BASE_URL) -> dict[str, ModelPrice]:
    """Current per-token prices from OpenRouter's public model list (no key needed)."""
    with urllib.request.urlopen(f"{base_url}/models", timeout=30) as response:
        listing = json.loads(response.read().decode("utf-8"))["data"]
    wanted = set(model_ids)
    prices = {}
    for model in listing:
        if model["id"] not in wanted:
            continue
        p = model.get("pricing") or {}

        def per_million(field: str, pricing: dict[str, Any] = p) -> float | None:
            value = pricing.get(field)
            return None if value in (None, "") else float(value) * PER_MILLION

        prompt, completion = per_million("prompt"), per_million("completion")
        if prompt is None or completion is None or prompt < 0 or completion < 0:
            continue  # free-routing pseudo models report -1
        prices[model["id"]] = ModelPrice(
            input=prompt,
            output=completion,
            cache_read=per_million("input_cache_read"),
            cache_write=per_million("input_cache_write"),
        )
    return prices
