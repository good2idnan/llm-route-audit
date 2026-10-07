"""OpenAI and OpenAI-compatible provider, tested without network access."""

import io
import json
import urllib.error

import pytest

from llm_route_audit.cache import request_key
from llm_route_audit.candidates import Candidate
from llm_route_audit.providers import openai as openai_provider
from llm_route_audit.providers.base import ProviderError
from llm_route_audit.records import Message
from llm_route_audit.replay import candidate_cost

MESSAGES = [Message(role="system", content="Be brief."), Message(role="user", content="Hi")]
LUNA = Candidate(model="openai/gpt-6-luna", effort="low", max_tokens=500)


class FakeResponse(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


def reply(content="Hello", finish="stop", refusal=None, cached=0):
    return {
        "choices": [
            {
                "message": {"role": "assistant", "content": content, "refusal": refusal},
                "finish_reason": finish,
            }
        ],
        "usage": {
            "prompt_tokens": 120,
            "completion_tokens": 30,
            "prompt_tokens_details": {"cached_tokens": cached},
        },
    }


def http_error(code, message="boom", error_code=None):
    body = {"error": {"message": message, "code": error_code}}
    return urllib.error.HTTPError("u", code, message, {}, io.BytesIO(json.dumps(body).encode()))


def test_candidate_naming():
    assert (LUNA.provider, LUNA.api_model) == ("openai", "gpt-6-luna")
    local = Candidate(model="qwen3-8b", base_url="http://localhost:1234/v1")
    assert local.provider == "openai" and local.runs_locally
    assert not LUNA.runs_locally


def test_base_url_implies_an_openai_compatible_server():
    assert Candidate(model="claude-haiku-4-5", base_url="http://x").provider == "openai"


def test_base_url_conflicts_with_other_providers():
    with pytest.raises(ValueError, match="need provider: openai"):
        Candidate(model="claude-haiku-4-5", provider="anthropic", base_url="http://x")


def test_request_uses_the_right_token_limit_field():
    official = openai_provider.build_request(LUNA, MESSAGES)
    assert official["max_completion_tokens"] == 500 and official["reasoning_effort"] == "low"
    compatible = openai_provider.build_request(
        Candidate(model="m", provider="openai", base_url="http://localhost:1/v1"), MESSAGES
    )
    assert "max_tokens" in compatible and "max_completion_tokens" not in compatible


def test_response_parsing():
    completion = openai_provider.parse_response(reply(cached=100))
    assert (completion.text, completion.input_tokens, completion.cache_read_tokens) == (
        "Hello",
        20,
        100,
    )
    assert openai_provider.parse_response(reply(finish="length")).status == "truncated"
    refused = openai_provider.parse_response(reply(content=None, refusal="I can't help with that."))
    assert refused.status == "refusal" and refused.text == "I can't help with that."


def test_round_trip(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    seen = {}

    def fake_urlopen(request, timeout):
        seen["url"] = request.full_url
        seen["auth"] = request.get_header("Authorization")
        return FakeResponse(json.dumps(reply()).encode())

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    assert openai_provider.OpenAIProvider().complete(LUNA, MESSAGES).text == "Hello"
    assert seen == {"url": "https://api.openai.com/v1/chat/completions", "auth": "Bearer sk-test"}


def test_local_server_needs_no_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    seen = {}

    def fake_urlopen(request, timeout):
        seen["auth"] = request.get_header("Authorization")
        return FakeResponse(json.dumps(reply()).encode())

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    local = Candidate(model="m", provider="openai", base_url="http://localhost:11434/v1")
    openai_provider.OpenAIProvider().complete(local, MESSAGES)
    assert seen["auth"] is None


def test_missing_key_for_openai_is_fatal(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(ProviderError) as err:
        openai_provider.OpenAIProvider().complete(LUNA, MESSAGES)
    assert err.value.fatal and "OPENAI_API_KEY" in str(err.value)


@pytest.mark.parametrize(
    ("error", "fatal", "disable"),
    [
        (http_error(401), True, False),
        (http_error(429, "quota", "insufficient_quota"), True, False),
        (http_error(404, "model not found"), False, True),
        (http_error(400, "unsupported value: reasoning_effort"), False, True),
    ],
)
def test_errors_are_classified(monkeypatch, error, fatal, disable):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")

    def fake_urlopen(request, timeout):
        raise error

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    with pytest.raises(ProviderError) as err:
        openai_provider.OpenAIProvider().complete(LUNA, MESSAGES)
    assert (err.value.fatal, err.value.disable) == (fatal, disable)


def test_rate_limits_are_retried_but_quota_is_not(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    attempts = []

    def fake_urlopen(request, timeout):
        attempts.append(1)
        if len(attempts) < 3:
            raise http_error(429, "slow down", "rate_limit_exceeded")
        return FakeResponse(json.dumps(reply()).encode())

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    provider = openai_provider.OpenAIProvider()
    provider._sleep = lambda seconds: None
    assert provider.complete(LUNA, MESSAGES).text == "Hello"
    assert len(attempts) == 3


def test_cache_key_includes_the_server():
    a = Candidate(model="m", provider="openai", base_url="http://localhost:1/v1")
    b = Candidate(model="m", provider="openai", base_url="http://localhost:2/v1")
    assert request_key(a, MESSAGES) != request_key(b, MESSAGES)


def test_local_servers_are_free_even_without_a_price():
    from datetime import date

    from llm_route_audit.costs import PriceTable

    table = PriceTable(updated=date(2026, 1, 1), models={})
    local = Candidate(model="m", provider="openai", base_url="http://127.0.0.1:8000/v1")
    remote = Candidate(model="m", provider="openai", base_url="https://api.groq.com/openai/v1")
    assert candidate_cost(table, local, input_tokens=10, output_tokens=10) == 0.0
    assert candidate_cost(table, remote, input_tokens=10, output_tokens=10) is None
