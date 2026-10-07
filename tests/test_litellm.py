import json

import pytest
from typer.testing import CliRunner

from llm_route_audit.cli import app
from llm_route_audit.ingest.jsonl import load_jsonl
from llm_route_audit.ingest.litellm import Skip, convert, import_litellm
from llm_route_audit.report import litellm_model


def payload(**overrides):
    base = {
        "id": "chatcmpl-1",
        "call_type": "acompletion",
        "model": "claude-opus-5-5",
        "model_group": "support-bot",
        "status": "success",
        "cache_hit": False,
        "startTime": 1759309200.0,  # 2025-10-01T09:00:00Z
        "endTime": 1759309201.5,
        "prompt_tokens": 150,
        "completion_tokens": 40,
        "response_cost": 0.0014,
        "request_tags": ["team:support", "task:classify_ticket"],
        "messages": [
            {"role": "system", "content": "Classify the ticket."},
            {"role": "user", "content": [{"type": "text", "text": "My card was charged twice."}]},
        ],
        "response": {
            "choices": [{"message": {"role": "assistant", "content": '{"category": "billing"}'}}],
            "usage": {"prompt_tokens": 150, "prompt_tokens_details": {"cached_tokens": 100}},
        },
        "metadata": {"user_api_key_hash": "secret-hash"},
    }
    base.update(overrides)
    return base


def test_payload_becomes_a_log_record():
    record = convert(payload())
    assert record.id == "chatcmpl-1"
    assert record.timestamp.isoformat() == "2025-10-01T09:00:00+00:00"
    assert record.task_type == "classify_ticket"
    assert [m.content for m in record.conversation()] == [
        "Classify the ticket.",
        "My card was charged twice.",
    ]
    assert record.response == '{"category": "billing"}'
    assert (record.input_tokens, record.cache_read_tokens, record.output_tokens) == (50, 100, 40)
    assert record.latency_ms == pytest.approx(1500)
    assert record.metadata["logged_cost"] == 0.0014
    assert "user_api_key_hash" not in json.dumps(record.metadata)  # keys and hashes stay out


def test_provider_prefix_is_dropped_for_anthropic_models():
    assert convert(payload(model="anthropic/claude-haiku-4-5")).model == "claude-haiku-4-5"


@pytest.mark.parametrize(
    ("overrides", "reason"),
    [
        ({"status": "failure"}, "failed request"),
        ({"cache_hit": True}, "cache"),
        ({"messages": None}, "turn on prompt logging"),
        (
            {"messages": [{"role": "user", "content": [{"type": "image_url", "image_url": {}}]}]},
            "images",
        ),
        ({"messages": [{"role": "function", "content": "42"}]}, "unsupported message role"),
        ({"response": {"choices": [{"message": {"content": None}}]}}, "no response logged"),
        ({"startTime": None}, "missing or invalid"),
    ],
)
def test_payloads_that_cannot_be_replayed_are_skipped(overrides, reason):
    with pytest.raises(Skip, match=reason):
        convert(payload(**overrides))


WEATHER_TOOL = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Current weather for a city.",
        "parameters": {"type": "object", "properties": {"city": {"type": "string"}}},
    },
}


def test_agent_step_keeps_tool_calls_results_and_definitions():
    record = convert(
        payload(
            messages=[
                {"role": "developer", "content": "You plan trips."},
                {"role": "user", "content": "Weather in Oslo and Rome?"},
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {"name": "get_weather", "arguments": '{"city": "Oslo"}'},
                        }
                    ],
                },
                {"role": "tool", "tool_call_id": "call_1", "content": "4 C, rain"},
            ],
            response={
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": None,
                            "tool_calls": [
                                {
                                    "id": "call_2",
                                    "type": "function",
                                    "function": {
                                        "name": "get_weather",
                                        "arguments": '{"city": "Rome"}',
                                    },
                                }
                            ],
                        }
                    }
                ]
            },
            model_parameters={"tools": [WEATHER_TOOL], "temperature": 0},
            metadata={"session_id": "trip-42"},
        )
    )
    assert record.is_agent_step
    assert [m.role for m in record.conversation()] == ["system", "user", "assistant", "tool"]
    assert record.messages[2].tool_calls[0].arguments == {"city": "Oslo"}
    assert record.messages[3].tool_call_id == "call_1"
    assert record.response == ""
    assert [(c.name, c.arguments) for c in record.response_tool_calls] == [
        ("get_weather", {"city": "Rome"})
    ]
    assert record.tools[0].name == "get_weather"
    assert record.tools[0].parameters["properties"] == {"city": {"type": "string"}}
    assert record.session_id == "trip-42"


def test_session_falls_back_to_the_trace_id():
    assert convert(payload(trace_id="trace-9")).session_id == "trace-9"
    assert convert(payload()).session_id is None


def test_import_reads_files_and_folders(tmp_path):
    folder = tmp_path / "logs"
    folder.mkdir()
    (folder / "a.jsonl").write_text(
        json.dumps(payload(id="1")) + "\n" + json.dumps(payload(id="2", status="failure")) + "\n",
        encoding="utf-8",
    )
    (folder / "b.json").write_text(json.dumps([payload(id="3"), payload(id="1")]), "utf-8")
    (folder / "broken.json").write_text("{not json", "utf-8")

    result = import_litellm(folder)
    assert [r.id for r in result.records] == ["1", "3"]
    assert result.skipped == {"failed request": 1, "duplicate id": 1, "not valid JSON": 1}


def test_import_command_writes_a_valid_log(tmp_path):
    source = tmp_path / "litellm.jsonl"
    source.write_text(
        "\n".join(json.dumps(payload(id=str(i), request_tags=[])) for i in range(3)), "utf-8"
    )
    out = tmp_path / "logs.jsonl"
    result = CliRunner().invoke(app, ["import", str(source), "--out", str(out)])
    assert result.exit_code == 0, result.output
    assert "Imported 3 requests" in result.output
    assert "Tip: 3 requests have no task type" in result.output
    loaded = load_jsonl(out)
    assert loaded.ok and len(loaded.records) == 3


def test_litellm_model_names():
    assert litellm_model("claude-haiku-4-5", "anthropic") == (
        "anthropic/claude-haiku-4-5",
        "anthropic",
    )
    assert litellm_model("claude-opus-5-5", None)[0] == "anthropic/claude-opus-5-5"
    assert litellm_model("openrouter/anthropic/claude-haiku-4.5", "openrouter")[1] == "openrouter"
    assert litellm_model("ollama/llama3.2", "ollama") == ("ollama/llama3.2", "ollama")
