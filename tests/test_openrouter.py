"""OpenRouter provider, tested without network access."""

import io
import json
import urllib.error

import pytest

from llm_route_audit.candidates import Candidate
from llm_route_audit.providers import openrouter
from llm_route_audit.providers.base import ProviderError
from llm_route_audit.records import Message

MESSAGES = [Message(role="system", content="Be brief."), Message(role="user", content="Hi")]
HAIKU = Candidate(model="openrouter/anthropic/claude-haiku-4.5", max_tokens=500)


class FakeResponse(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


def reply(content="Hello", finish="stop", native="end_turn", cost=0.0012, cached=0):
    return {
        "choices": [
            {
                "message": {"role": "assistant", "content": content},
                "finish_reason": finish,
                "native_finish_reason": native,
            }
        ],
        "usage": {
            "prompt_tokens": 120,
            "completion_tokens": 30,
            "prompt_tokens_details": {"cached_tokens": cached},
            "cost": cost,
        },
    }


def http_error(code, message="boom"):
    body = json.dumps({"error": {"code": code, "message": message}}).encode()
    return urllib.error.HTTPError("u", code, message, {}, io.BytesIO(body))


def test_candidate_name_selects_openrouter():
    assert (HAIKU.provider, HAIKU.api_model) == ("openrouter", "anthropic/claude-haiku-4.5")


def test_request_shape_with_effort():
    request = openrouter.build_request(
        Candidate(model="openrouter/anthropic/claude-sonnet-5.5", effort="low"), MESSAGES
    )
    assert request["model"] == "anthropic/claude-sonnet-5.5"
    assert request["reasoning"] == {"effort": "low"}
    assert request["messages"][0] == {"role": "system", "content": "Be brief."}


def test_response_uses_reported_cost_and_splits_cached_input():
    completion = openrouter.parse_response(reply(cached=100))
    assert completion.text == "Hello"
    assert (completion.input_tokens, completion.cache_read_tokens) == (20, 100)
    assert completion.output_tokens == 30
    assert completion.cost == pytest.approx(0.0012)
    assert completion.status == "ok"


def test_finish_reasons_map_to_statuses():
    assert openrouter.parse_response(reply(finish="length")).status == "truncated"
    assert openrouter.parse_response(reply(native="refusal")).status == "refusal"


def test_error_inside_a_200_response_is_raised():
    with pytest.raises(ProviderError, match="upstream failed"):
        openrouter.parse_response({"error": {"message": "upstream failed"}})


def test_round_trip_sends_the_key(monkeypatch):
    seen = {}

    def fake_urlopen(request, timeout):
        seen["auth"] = request.get_header("Authorization")
        seen["body"] = json.loads(request.data)
        return FakeResponse(json.dumps(reply()).encode())

    monkeypatch.setattr(openrouter.urllib.request, "urlopen", fake_urlopen)
    completion = openrouter.OpenRouterProvider(api_key="sk-test").complete(HAIKU, MESSAGES)
    assert seen["auth"] == "Bearer sk-test"
    assert seen["body"]["max_tokens"] == 500
    assert completion.text == "Hello"


def test_missing_key_is_fatal(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    with pytest.raises(ProviderError) as err:
        openrouter.OpenRouterProvider().complete(HAIKU, MESSAGES)
    assert err.value.fatal and "OPENROUTER_API_KEY" in str(err.value)


@pytest.mark.parametrize(
    ("code", "fatal", "disable"),
    [(401, True, False), (402, True, False), (404, False, True), (400, False, True)],
)
def test_http_errors_are_classified(monkeypatch, code, fatal, disable):
    def fake_urlopen(request, timeout):
        raise http_error(code)

    monkeypatch.setattr(openrouter.urllib.request, "urlopen", fake_urlopen)
    with pytest.raises(ProviderError) as err:
        openrouter.OpenRouterProvider(api_key="k").complete(HAIKU, MESSAGES)
    assert (err.value.fatal, err.value.disable) == (fatal, disable)


def test_rate_limits_are_retried(monkeypatch):
    attempts = []

    def fake_urlopen(request, timeout):
        attempts.append(1)
        if len(attempts) < 3:
            raise http_error(429, "slow down")
        return FakeResponse(json.dumps(reply()).encode())

    monkeypatch.setattr(openrouter.urllib.request, "urlopen", fake_urlopen)
    provider = openrouter.OpenRouterProvider(api_key="k")
    provider._sleep = lambda seconds: None
    assert provider.complete(HAIKU, MESSAGES).text == "Hello"
    assert len(attempts) == 3


def test_fetch_prices_converts_to_per_million(monkeypatch):
    listing = {
        "data": [
            {
                "id": "anthropic/claude-haiku-4.5",
                "pricing": {
                    "prompt": "0.000001",
                    "completion": "0.000005",
                    "input_cache_read": "0.0000001",
                },
            },
            {"id": "typesafe/jev-router", "pricing": {"prompt": "-1", "completion": "-1"}},
            {"id": "other/model", "pricing": {"prompt": "0.000002", "completion": "0.000002"}},
        ]
    }
    monkeypatch.setattr(
        openrouter.urllib.request,
        "urlopen",
        lambda url, timeout: FakeResponse(json.dumps(listing).encode()),
    )
    prices = openrouter.fetch_prices(["anthropic/claude-haiku-4.5", "typesafe/jev-router"])
    assert set(prices) == {"anthropic/claude-haiku-4.5"}
    haiku = prices["anthropic/claude-haiku-4.5"]
    assert (haiku.input, haiku.output, haiku.cache_read) == pytest.approx((1.0, 5.0, 0.1))
