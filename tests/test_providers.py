"""Provider adapters, tested without network access."""

import io
import json
import urllib.error
from types import SimpleNamespace

import pytest

from llm_route_audit.candidates import Candidate
from llm_route_audit.providers import ollama
from llm_route_audit.providers.anthropic import build_request, parse_response
from llm_route_audit.providers.base import ProviderError
from llm_route_audit.records import Message

MESSAGES = [
    Message(role="system", content="Be brief."),
    Message(role="user", content="Hi"),
]


def test_anthropic_request_moves_system_messages_and_sets_effort():
    request = build_request(Candidate(model="claude-opus-5-5", effort="low"), MESSAGES)
    assert request["system"] == "Be brief."
    assert request["messages"] == [{"role": "user", "content": "Hi"}]
    assert request["output_config"] == {"effort": "low"}


def test_anthropic_request_without_effort_leaves_the_model_default():
    request = build_request(Candidate(model="claude-haiku-4-5"), MESSAGES[1:])
    assert "output_config" not in request
    assert "system" not in request


def test_anthropic_response_keeps_text_and_drops_thinking():
    response = SimpleNamespace(
        content=[
            SimpleNamespace(type="thinking", thinking=""),
            SimpleNamespace(type="text", text="Hello"),
        ],
        usage=SimpleNamespace(
            input_tokens=10,
            output_tokens=5,
            cache_read_input_tokens=None,
            cache_creation_input_tokens=3,
        ),
        stop_reason="end_turn",
    )
    completion = parse_response(response)
    assert (completion.text, completion.input_tokens, completion.output_tokens) == ("Hello", 10, 5)
    assert (completion.cache_read_tokens, completion.cache_write_tokens) == (0, 3)
    assert completion.status == "ok"


@pytest.mark.parametrize(("stop", "status"), [("refusal", "refusal"), ("max_tokens", "truncated")])
def test_anthropic_stop_reasons_map_to_statuses(stop, status):
    response = SimpleNamespace(
        content=[],
        usage=SimpleNamespace(input_tokens=1, output_tokens=1),
        stop_reason=stop,
    )
    assert parse_response(response).status == status


class FakeHTTPResponse(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


def test_ollama_round_trip(monkeypatch):
    sent = {}

    def fake_urlopen(request, timeout):
        sent.update(json.loads(request.data))
        body = {
            "message": {"content": "Hello"},
            "prompt_eval_count": 12,
            "eval_count": 4,
            "done_reason": "stop",
        }
        return FakeHTTPResponse(json.dumps(body).encode())

    monkeypatch.setattr(ollama.urllib.request, "urlopen", fake_urlopen)
    completion = ollama.OllamaProvider("http://x").complete(
        Candidate(model="ollama/llama3.2"), MESSAGES
    )
    assert sent["model"] == "llama3.2"
    assert sent["messages"][0] == {"role": "system", "content": "Be brief."}
    assert (completion.text, completion.input_tokens, completion.output_tokens) == ("Hello", 12, 4)


def test_ollama_missing_model_disables_the_candidate(monkeypatch):
    def fake_urlopen(request, timeout):
        raise urllib.error.HTTPError("u", 404, "not found", {}, io.BytesIO(b"model not found"))

    monkeypatch.setattr(ollama.urllib.request, "urlopen", fake_urlopen)
    with pytest.raises(ProviderError) as err:
        ollama.OllamaProvider("http://x").complete(Candidate(model="ollama/nope"), MESSAGES)
    assert err.value.disable and not err.value.fatal


def test_ollama_unreachable_is_fatal(monkeypatch):
    def fake_urlopen(request, timeout):
        raise urllib.error.URLError("connection refused")

    monkeypatch.setattr(ollama.urllib.request, "urlopen", fake_urlopen)
    with pytest.raises(ProviderError) as err:
        ollama.OllamaProvider("http://x").complete(Candidate(model="ollama/llama3.2"), MESSAGES)
    assert err.value.fatal


def test_ollama_host_without_scheme(monkeypatch):
    monkeypatch.setenv("OLLAMA_HOST", "127.0.0.1:11434")
    assert ollama.ollama_url() == "http://127.0.0.1:11434"
