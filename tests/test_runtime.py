"""The runtime router: applies an exported policy inside an app."""

import json

import pytest

from llm_route_audit.ingest.jsonl import load_jsonl
from llm_route_audit.policy import Policy
from llm_route_audit.providers.base import Completion, ProviderError
from llm_route_audit.records import ToolCall
from llm_route_audit.runtime import Router

POLICY = Policy.model_validate(
    {
        "default": {"model": "claude-opus-5-5"},
        "routes": {
            "classify": {
                "model": "claude-haiku-4-5",
                "reference": "claude-opus-5-5",
                "expected_pass_rate": 0.98,
            },
            "draft": {"model": "claude-sonnet-5-5", "effort": "low"},
            "search": {"model": "openai/gpt-6-luna", "effort": "low"},
            "summarize": {"model": "openrouter/qwen/qwen3", "effort": "high"},
        },
    }
)


class FakeProvider:
    def __init__(self, fail_models=(), fatal=False):
        self.calls = []
        self.fail_models = set(fail_models)
        self.fatal = fatal

    def complete(self, candidate, messages, tools=None):
        self.calls.append((candidate.model, candidate.effort, tools))
        if candidate.model in self.fail_models:
            raise ProviderError("overloaded", fatal=self.fatal)
        calls = [ToolCall(name="lookup", arguments={"q": 1})] if tools else None
        return Completion(f"answer from {candidate.model}", 50, 10, tool_calls=calls)


@pytest.fixture
def provider(monkeypatch):
    fake = FakeProvider()
    monkeypatch.setattr("llm_route_audit.runtime.get_provider", lambda name: fake)
    return fake


def test_choose_follows_routes_and_falls_back_to_the_default():
    router = Router(POLICY)
    classify = router.choose("classify")
    assert (classify.model, classify.routed, classify.reference) == (
        "claude-haiku-4-5",
        True,
        "claude-opus-5-5",
    )
    unknown = router.choose("translate")
    assert (unknown.model, unknown.routed) == ("claude-opus-5-5", False)
    assert router.choose(None).model == "claude-opus-5-5"


def test_sdk_arguments():
    router = Router(POLICY)
    assert router.anthropic_args("draft") == {
        "model": "claude-sonnet-5-5",
        "output_config": {"effort": "low"},
    }
    assert router.anthropic_args("classify") == {"model": "claude-haiku-4-5"}
    assert router.openai_args("search") == {"model": "gpt-6-luna", "reasoning_effort": "low"}
    assert router.openai_args("summarize") == {
        "model": "qwen/qwen3",
        "extra_body": {"reasoning": {"effort": "high"}},
    }
    assert router.litellm_args("draft") == {
        "model": "anthropic/claude-sonnet-5-5",
        "reasoning_effort": "low",
    }
    with pytest.raises(ValueError, match="not an Anthropic model"):
        router.anthropic_args("search")
    with pytest.raises(ValueError, match="not an OpenAI-style API"):
        router.openai_args("draft")


def test_complete_logs_in_the_audit_format(tmp_path, provider):
    log = tmp_path / "requests.jsonl"
    router = Router(POLICY, log_path=log)
    reply = router.complete("classify", [{"role": "user", "content": "Charged twice"}])
    assert reply.text == "answer from claude-haiku-4-5" and not reply.fell_back
    tooled = router.complete(
        "draft",
        [{"role": "user", "content": "Look it up"}],
        tools=[{"name": "lookup", "description": "Search."}],
        session_id="s1",
    )
    assert provider.calls[1][1] == "low" and provider.calls[1][2][0].name == "lookup"

    records = load_jsonl(log).records  # the log reads back as a valid audit log
    assert [(r.id, r.task_type, r.model) for r in records] == [
        (reply.record_id, "classify", "claude-haiku-4-5"),
        (tooled.record_id, "draft", "claude-sonnet-5-5"),
    ]
    assert records[1].response_tool_calls[0].name == "lookup"
    assert records[1].session_id == "s1" and records[1].tools[0].name == "lookup"
    assert records[0].input_tokens == 50 and records[0].latency_ms is not None

    router.record_outcome(reply.record_id, "thumbs_down", note="wrong category")
    [outcome] = [json.loads(line) for line in (tmp_path / "requests.outcomes.jsonl").open()]
    assert outcome["record_id"] == reply.record_id and outcome["outcome"] == "thumbs_down"


def test_failed_route_falls_back_to_the_model_it_replaced(tmp_path, monkeypatch):
    fake = FakeProvider(fail_models={"claude-haiku-4-5"})
    monkeypatch.setattr("llm_route_audit.runtime.get_provider", lambda name: fake)
    log = tmp_path / "requests.jsonl"
    reply = Router(POLICY, log_path=log).complete("classify", [{"role": "user", "content": "x"}])
    assert reply.fell_back and reply.model == "claude-opus-5-5"
    [record] = load_jsonl(log).records
    assert record.model == "claude-opus-5-5" and record.metadata["fell_back"] is True

    without = Router(POLICY, fallback_to_reference=False)
    with pytest.raises(ProviderError):
        without.complete("classify", [{"role": "user", "content": "x"}])


def test_fatal_errors_never_fall_back(monkeypatch):
    fake = FakeProvider(fail_models={"claude-haiku-4-5"}, fatal=True)
    monkeypatch.setattr("llm_route_audit.runtime.get_provider", lambda name: fake)
    with pytest.raises(ProviderError):
        Router(POLICY).complete("classify", [{"role": "user", "content": "x"}])
    assert [c[0] for c in fake.calls] == ["claude-haiku-4-5"]


def test_outcomes_need_somewhere_to_go():
    with pytest.raises(ValueError, match="log_path"):
        Router(POLICY).record_outcome("abc", "good")


def test_router_reads_an_exported_policy(tmp_path):
    path = tmp_path / "policy.yaml"
    path.write_text(
        'version: 1\ndefault: {model: "claude-opus-5-5"}\nroutes:\n'
        '  easy: {model: "claude-haiku-4-5", provider: anthropic, reference: "claude-opus-5-5", '
        "expected_pass_rate: 1.000}  # evidence\n",
        "utf-8",
    )
    assert Router.from_file(path).choose("easy").model == "claude-haiku-4-5"
