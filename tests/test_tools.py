"""Agent steps end to end through the plumbing: records, provider formats, cache and runner."""

from types import SimpleNamespace

from llm_route_audit.cache import ResultCache, request_key
from llm_route_audit.candidates import Candidate
from llm_route_audit.costs import load_prices
from llm_route_audit.providers import anthropic as anthropic_provider
from llm_route_audit.providers import ollama, openrouter
from llm_route_audit.providers import openai as openai_provider
from llm_route_audit.providers.base import Completion
from llm_route_audit.providers.tooling import (
    anthropic_messages,
    openai_messages,
    parse_openai_tool_calls,
    render_tool_calls,
)
from llm_route_audit.records import LogRecord, Message, ToolCall, ToolDef
from llm_route_audit.replay import worst_case_cost
from llm_route_audit.runner import Job, execute

WEATHER = ToolDef(
    name="get_weather",
    description="Current weather for a city.",
    parameters={"type": "object", "properties": {"city": {"type": "string"}}},
)
HISTORY = [
    Message(role="system", content="You plan trips."),
    Message(role="user", content="Weather in Oslo and Rome?"),
    Message(
        role="assistant",
        content="Checking both.",
        tool_calls=[
            ToolCall(id="c1", name="get_weather", arguments={"city": "Oslo"}),
            ToolCall(id="c2", name="get_weather", arguments={"city": "Rome"}),
        ],
    ),
    Message(role="tool", tool_call_id="c1", name="get_weather", content="4 C, rain"),
    Message(role="tool", tool_call_id="c2", name="get_weather", content="22 C, sun"),
]
PLAIN = [Message(role="user", content="Hi")]


def test_agent_records_are_recognised():
    step = LogRecord(
        id="s1",
        timestamp="2026-10-01T09:00:00Z",
        model="claude-opus-5-5",
        messages=HISTORY,
        response="Oslo is wet, Rome is sunny.",
        session_id="trip-1",
    )
    assert step.is_agent_step
    plain = LogRecord(id="p", timestamp=step.timestamp, model="m", prompt="Hi", response="Hey")
    assert not plain.is_agent_step


def test_openai_format_keeps_calls_and_results():
    converted = openai_messages(HISTORY)
    assert converted[2]["tool_calls"][1] == {
        "id": "c2",
        "type": "function",
        "function": {"name": "get_weather", "arguments": '{"city": "Rome"}'},
    }
    assert converted[3] == {"role": "tool", "tool_call_id": "c1", "content": "4 C, rain"}


def test_anthropic_format_groups_results_into_one_user_turn():
    system, converted = anthropic_messages(HISTORY)
    assert system == "You plan trips."
    assert [m["role"] for m in converted] == ["user", "assistant", "user"]
    assert converted[1]["content"][0] == {"type": "text", "text": "Checking both."}
    assert converted[1]["content"][1]["type"] == "tool_use"
    assert converted[1]["content"][1]["input"] == {"city": "Oslo"}
    assert [b["tool_use_id"] for b in converted[2]["content"]] == ["c1", "c2"]


def test_requests_carry_tool_definitions():
    opus = Candidate(model="claude-opus-5-5")
    assert anthropic_provider.build_request(opus, HISTORY, [WEATHER])["tools"] == [
        {
            "name": "get_weather",
            "description": "Current weather for a city.",
            "input_schema": WEATHER.parameters,
        }
    ]
    for module, model in (
        (openai_provider, "openai/gpt-6-luna"),
        (openrouter, "openrouter/qwen/qwen3"),
        (ollama, "ollama/qwen3"),
    ):
        request = module.build_request(Candidate(model=model), HISTORY, [WEATHER])
        assert request["tools"][0]["function"]["name"] == "get_weather"
        assert "tools" not in module.build_request(Candidate(model=model), PLAIN)


def test_tool_call_answers_are_parsed():
    reply = {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "c3",
                            "type": "function",
                            "function": {"name": "get_weather", "arguments": '{"city": "Lima"}'},
                        }
                    ],
                },
                "finish_reason": "tool_calls",
            }
        ],
        "usage": {"prompt_tokens": 50, "completion_tokens": 12},
    }
    completion = openai_provider.parse_response(reply)
    assert completion.text == ""
    assert completion.tool_calls == [
        ToolCall(id="c3", name="get_weather", arguments={"city": "Lima"})
    ]

    response = SimpleNamespace(
        content=[
            SimpleNamespace(type="text", text="Let me check."),
            SimpleNamespace(type="tool_use", id="t1", name="get_weather", input={"city": "Lima"}),
        ],
        usage=SimpleNamespace(input_tokens=50, output_tokens=12),
        stop_reason="tool_use",
    )
    completion = anthropic_provider.parse_response(response)
    assert completion.text == "Let me check."
    assert completion.status == "ok"
    assert completion.tool_calls[0].arguments == {"city": "Lima"}


def test_broken_tool_arguments_are_kept_not_lost():
    calls = parse_openai_tool_calls([{"function": {"name": "f", "arguments": "{not json"}}])
    assert calls[0].arguments == {"_raw": "{not json"}


def test_tool_calls_render_as_text_for_the_judge():
    calls = [ToolCall(name="get_weather", arguments={"city": "Oslo", "units": "c"})]
    assert render_tool_calls(calls) == 'CALL get_weather({"city": "Oslo", "units": "c"})'
    assert render_tool_calls(None) == ""


def test_cache_key_covers_tools_but_plain_keys_are_unchanged():
    opus = Candidate(model="claude-opus-5-5")
    assert request_key(opus, PLAIN) == request_key(opus, PLAIN, None)
    assert request_key(opus, PLAIN) != request_key(opus, PLAIN, [WEATHER])
    changed = [*HISTORY[:-1], HISTORY[-1].model_copy(update={"content": "23 C, sun"})]
    assert request_key(opus, HISTORY) != request_key(opus, changed)


def test_cache_round_trips_tool_calls(tmp_path):
    cache = ResultCache(tmp_path / "cache.sqlite")
    opus = Candidate(model="claude-opus-5-5")
    calls = [ToolCall(id="c3", name="get_weather", arguments={"city": "Lima"})]
    cache.put("k", opus, Completion("", 50, 12, tool_calls=calls), 100.0)
    completion, latency = cache.get("k")
    assert completion.tool_calls == calls
    assert latency == 100.0


class RecordingProvider:
    def __init__(self):
        self.calls = []

    def complete(self, candidate, messages, **kwargs):
        self.calls.append(kwargs)
        return Completion("ok", 10, 2)


def test_runner_passes_tools_only_when_there_are_some():
    provider = RecordingProvider()
    opus = Candidate(model="claude-opus-5-5")
    run = execute(
        [Job(opus, HISTORY, [WEATHER]), Job(opus, PLAIN)],
        ResultCache(":memory:"),
        lambda name: provider,
        concurrency=1,
    )
    assert [o.status for o in run.outcomes] == ["ok", "ok"]
    assert sorted(c.get("tools") == [WEATHER] for c in provider.calls) == [False, True]
    assert {} in provider.calls  # plain requests call providers exactly as before


def test_worst_case_counts_tool_history_and_definitions():
    prices = load_prices()
    opus = Candidate(model="claude-opus-5-5", max_tokens=100)
    bare = [m.model_copy(update={"tool_calls": None, "tool_call_id": None}) for m in HISTORY]
    with_tools = worst_case_cost(prices, Job(opus, HISTORY, [WEATHER]))
    assert with_tools > worst_case_cost(prices, Job(opus, HISTORY))
    assert worst_case_cost(prices, Job(opus, HISTORY)) > worst_case_cost(prices, Job(opus, bare))
