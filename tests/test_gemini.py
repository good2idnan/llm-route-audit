"""Gemini provider, tested without network access."""

import io
import json
import urllib.error

import pytest

from llm_route_audit.candidates import Candidate
from llm_route_audit.providers import gemini
from llm_route_audit.providers.base import ProviderError
from llm_route_audit.records import Message, ToolCall, ToolDef

FLASH = Candidate(model="gemini/gemini-3-flash", effort="low", max_tokens=800)
HISTORY = [
    Message(role="system", content="You plan trips."),
    Message(role="user", content="Weather in Oslo and Rome?"),
    Message(
        role="assistant",
        tool_calls=[
            ToolCall(id="c1", name="get_weather", arguments={"city": "Oslo"}),
            ToolCall(id="c2", name="get_weather", arguments={"city": "Rome"}),
        ],
    ),
    Message(role="tool", tool_call_id="c1", content='{"temp_c": 4}'),
    Message(role="tool", tool_call_id="c2", content="22 C, sun"),
]
WEATHER = ToolDef(name="get_weather", parameters={"type": "object", "properties": {}})


class FakeResponse(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


def reply(parts, finish="STOP", **usage):
    return {
        "candidates": [{"content": {"role": "model", "parts": parts}, "finishReason": finish}],
        "usageMetadata": {
            "promptTokenCount": 120,
            "candidatesTokenCount": 30,
            "thoughtsTokenCount": 50,
            **usage,
        },
    }


def test_candidate_naming():
    assert FLASH.provider == "gemini" and FLASH.api_model == "gemini-3-flash"


def test_request_shape_with_history_tools_and_effort():
    request = gemini.build_request(FLASH, HISTORY, [WEATHER])
    assert request["systemInstruction"] == {"parts": [{"text": "You plan trips."}]}
    assert [c["role"] for c in request["contents"]] == ["user", "model", "user"]
    call = request["contents"][1]["parts"][0]
    assert call["functionCall"] == {"name": "get_weather", "args": {"city": "Oslo"}}
    assert call["thoughtSignature"] == gemini.FOREIGN_SIGNATURE
    results = request["contents"][2]["parts"]
    assert results[0]["functionResponse"] == {"name": "get_weather", "response": {"temp_c": 4}}
    assert results[1]["functionResponse"]["response"] == {"result": "22 C, sun"}
    assert request["tools"][0]["functionDeclarations"][0]["name"] == "get_weather"
    assert request["generationConfig"] == {
        "maxOutputTokens": 800,
        "thinkingConfig": {"thinkingLevel": "low"},
    }


@pytest.mark.parametrize(
    ("effort", "thinking"),
    [
        ("none", {"thinkingBudget": 0}),
        ("max", {"thinkingLevel": "high"}),
        (None, None),
    ],
)
def test_effort_maps_to_thinking(effort, thinking):
    candidate = Candidate(model="gemini/gemini-3-pro", effort=effort)
    config = gemini.build_request(candidate, HISTORY[:2])["generationConfig"]
    assert config.get("thinkingConfig") == thinking


def test_response_parsing_skips_thoughts_and_reads_calls():
    completion = gemini.parse_response(
        reply(
            [
                {"text": "thinking...", "thought": True},
                {"text": "Checking."},
                {"functionCall": {"name": "get_weather", "args": {"city": "Lima"}}},
            ],
            cachedContentTokenCount=100,
        )
    )
    assert completion.text == "Checking."
    assert completion.tool_calls == [ToolCall(name="get_weather", arguments={"city": "Lima"})]
    # cached input split out; thinking counts as output
    assert (completion.input_tokens, completion.cache_read_tokens) == (20, 100)
    assert completion.output_tokens == 80


@pytest.mark.parametrize(("finish", "status"), [("MAX_TOKENS", "truncated"), ("SAFETY", "refusal")])
def test_finish_reasons(finish, status):
    assert gemini.parse_response(reply([{"text": "x"}], finish=finish)).status == status


def test_blocked_prompt_is_a_refusal():
    data = {"promptFeedback": {"blockReason": "SAFETY"}, "usageMetadata": {"promptTokenCount": 9}}
    assert gemini.parse_response(data).status == "refusal"


def test_round_trip_sends_the_key_in_a_header(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setenv("GOOGLE_API_KEY", "g-test")
    seen = {}

    def fake_urlopen(request, timeout):
        seen["url"] = request.full_url
        seen["key"] = request.get_header("X-goog-api-key")
        return FakeResponse(json.dumps(reply([{"text": "Hello"}])).encode())

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    assert gemini.GeminiProvider().complete(FLASH, HISTORY[:2]).text == "Hello"
    assert seen == {
        "url": f"{gemini.GEMINI_URL}/models/gemini-3-flash:generateContent",
        "key": "g-test",
    }


def http_error(code, message, reason=None):
    body = {"error": {"code": code, "message": message, "details": [{"reason": reason}]}}
    return urllib.error.HTTPError("u", code, message, {}, io.BytesIO(json.dumps(body).encode()))


@pytest.mark.parametrize(
    ("error", "fatal", "disable"),
    [
        (http_error(400, "API key not valid.", "API_KEY_INVALID"), True, False),
        (http_error(403, "Permission denied"), True, False),
        (http_error(404, "models/x is not found"), False, True),
        (http_error(400, "Invalid argument"), False, True),
    ],
)
def test_errors_are_classified(monkeypatch, error, fatal, disable):
    monkeypatch.setenv("GEMINI_API_KEY", "g-test")

    def fail(request, timeout):
        raise error

    monkeypatch.setattr("urllib.request.urlopen", fail)
    with pytest.raises(ProviderError) as caught:
        gemini.GeminiProvider().complete(FLASH, HISTORY[:2])
    assert (caught.value.fatal, caught.value.disable) == (fatal, disable)


def test_missing_key_is_fatal(monkeypatch):
    for name in gemini.KEY_ENVS:
        monkeypatch.delenv(name, raising=False)
    with pytest.raises(ProviderError, match="GEMINI_API_KEY") as caught:
        gemini.GeminiProvider().complete(FLASH, HISTORY[:2])
    assert caught.value.fatal


def test_prices_come_from_googles_listing():
    from llm_route_audit.cli import _price_id

    assert _price_id(FLASH) == "google/gemini-3-flash"
