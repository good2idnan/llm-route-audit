import json

import pytest
from typer.testing import CliRunner

from routeaudit.cli import app
from routeaudit.ingest import langfuse, otel
from routeaudit.ingest.common import Skip, read_items, text_of
from routeaudit.ingest.jsonl import load_jsonl

# --- Langfuse -----------------------------------------------------------------------------


def generation(**overrides):
    base = {
        "id": "obs-1",
        "traceId": "trace-1",
        "type": "GENERATION",
        "name": "classify-ticket",
        "startTime": "2026-10-01T09:00:00.000Z",
        "endTime": "2026-10-01T09:00:01.250Z",
        "model": "claude-opus-5-5",
        "input": [
            {"role": "system", "content": "Classify the ticket."},
            {"role": "user", "content": "My card was charged twice."},
        ],
        "output": {"role": "assistant", "content": '{"category": "billing"}'},
        "usageDetails": {"input": 40, "output": 9, "cache_read_input_tokens": 100},
        "totalCost": 0.0012,
        "level": "DEFAULT",
        "tags": ["prod", "task:classify_ticket"],
    }
    base.update(overrides)
    return base


def test_langfuse_generation_becomes_a_record():
    record = langfuse.convert(generation())
    assert record.task_type == "classify_ticket"
    assert [m.role for m in record.conversation()] == ["system", "user"]
    assert record.response == '{"category": "billing"}'
    assert (record.input_tokens, record.output_tokens, record.cache_read_tokens) == (40, 9, 100)
    assert record.latency_ms == pytest.approx(1250)
    assert record.metadata["logged_cost"] == 0.0012


def test_langfuse_task_from_generation_name():
    assert (
        langfuse.convert(generation(tags=None), task_from_name=True).task_type == "classify-ticket"
    )


def test_langfuse_accepts_string_input_and_content_parts():
    record = langfuse.convert(
        generation(
            input="Summarise this call.",
            output={"role": "assistant", "content": [{"type": "text", "text": "Short summary."}]},
        )
    )
    assert record.conversation()[0].content == "Summarise this call."
    assert record.response == "Short summary."


@pytest.mark.parametrize(
    ("overrides", "reason"),
    [
        ({"type": "SPAN"}, "not a model call"),
        ({"level": "ERROR"}, "failed request"),
        ({"model": None}, "no model name"),
        ({"input": None}, "no input logged"),
        ({"output": {"role": "assistant", "content": None, "tool_calls": [{}]}}, "tool calls"),
        ({"output": None}, "no output logged"),
    ],
)
def test_langfuse_skips_what_cannot_be_replayed(overrides, reason):
    with pytest.raises(Skip, match=reason):
        langfuse.convert(generation(**overrides))


def test_langfuse_api_response_wrapper_is_unwrapped(tmp_path):
    path = tmp_path / "observations.json"
    path.write_text(
        json.dumps({"data": [generation(id="a"), generation(id="b", type="EVENT")], "meta": {}}),
        "utf-8",
    )
    result = langfuse.import_langfuse(path)
    assert [r.id for r in result.records] == ["a"]
    assert result.skipped == {"not a model call (span or event)": 1}


# --- OpenTelemetry ------------------------------------------------------------------------


def attr(key, value):
    if isinstance(value, bool):
        return {"key": key, "value": {"boolValue": value}}
    if isinstance(value, int):
        return {"key": key, "value": {"intValue": str(value)}}
    return {"key": key, "value": {"stringValue": value}}


def genai_span(span_id="span-1", status=None, **extra_attrs):
    attributes = {
        "gen_ai.operation.name": "chat",
        "gen_ai.request.model": "claude-opus-5-5",
        "gen_ai.usage.input_tokens": 140,
        "gen_ai.usage.output_tokens": 9,
        "gen_ai.usage.cache_read.input_tokens": 100,
        "gen_ai.system_instructions": json.dumps([{"type": "text", "content": "Classify."}]),
        "gen_ai.input.messages": json.dumps(
            [{"role": "user", "parts": [{"type": "text", "content": "Charged twice."}]}]
        ),
        "gen_ai.output.messages": json.dumps(
            [
                {
                    "role": "assistant",
                    "parts": [{"type": "text", "content": '{"category": "billing"}'}],
                    "finish_reason": "stop",
                }
            ]
        ),
        "task_type": "classify_ticket",
    }
    attributes.update(extra_attrs)
    span = {
        "traceId": "t1",
        "spanId": span_id,
        "name": "chat claude-opus-5-5",
        "startTimeUnixNano": "1759309200000000000",
        "endTimeUnixNano": "1759309201500000000",
        "attributes": [attr(k, v) for k, v in attributes.items() if v is not None],
    }
    if status:
        span["status"] = {"code": status}
    return span


def otlp(*spans):
    return {
        "resourceSpans": [{"resource": {}, "scopeSpans": [{"scope": {}, "spans": list(spans)}]}]
    }


def test_otel_span_becomes_a_record():
    record = otel.convert(genai_span())
    assert record.id == "span-1"
    assert record.task_type == "classify_ticket"
    assert [(m.role, m.content) for m in record.conversation()] == [
        ("system", "Classify."),
        ("user", "Charged twice."),
    ]
    assert record.response == '{"category": "billing"}'
    assert (record.input_tokens, record.cache_read_tokens) == (40, 100)  # cached split out
    assert record.latency_ms == pytest.approx(1500)
    assert record.timestamp.isoformat() == "2025-10-01T09:00:00+00:00"


@pytest.mark.parametrize(
    ("span", "reason"),
    [
        (genai_span(status=2), "failed request"),
        (genai_span(**{"gen_ai.input.messages": None}), "turn on GenAI message capture"),
        (
            genai_span(
                **{
                    "gen_ai.input.messages": json.dumps(
                        [{"role": "tool", "parts": [{"type": "tool_call_response"}]}]
                    )
                }
            ),
            "tool calls",
        ),
        ({"spanId": "x", "name": "GET /health", "attributes": []}, "not a model call"),
    ],
)
def test_otel_skips_what_cannot_be_replayed(span, reason):
    with pytest.raises(Skip, match=reason):
        otel.convert(span)


def test_otel_reads_collector_file_exports(tmp_path):
    path = tmp_path / "traces.jsonl"
    path.write_text(
        json.dumps(otlp(genai_span("a"), {"spanId": "http", "attributes": []}))
        + "\n"
        + json.dumps(otlp(genai_span("b", status=2)))
        + "\n",
        "utf-8",
    )
    result = otel.import_otel(path)
    assert [r.id for r in result.records] == ["a"]
    assert sum(result.skipped.values()) == 2


def test_otel_custom_task_attribute():
    span = genai_span(**{"task_type": None, "app.task": "triage"})
    assert otel.convert(span, task_attribute="app.task").task_type == "triage"


# --- shared helpers and the command -------------------------------------------------------


def test_text_of_handles_both_part_styles():
    assert text_of([{"type": "text", "text": "a"}, {"type": "text", "content": "b"}]) == "ab"
    with pytest.raises(Skip, match="images"):
        text_of([{"type": "image_url"}])


def test_read_items_handles_folders_and_bad_files(tmp_path):
    (tmp_path / "a.jsonl").write_text('{"x": 1}\n{"x": 2}\n', "utf-8")
    (tmp_path / "b.json").write_text("{broken", "utf-8")
    items = [item for _, item in read_items(tmp_path)]
    assert items[:2] == [{"x": 1}, {"x": 2}]
    assert "__unreadable__" in items[2]


@pytest.mark.parametrize(
    ("fmt", "content", "extra"),
    [
        ("langfuse", lambda: json.dumps([generation(id=str(i)) for i in range(2)]), []),
        ("otel", lambda: json.dumps(otlp(genai_span("a"), genai_span("b"))), []),
    ],
)
def test_import_command_for_each_format(tmp_path, fmt, content, extra):
    source = tmp_path / "source.json"
    source.write_text(content(), "utf-8")
    out = tmp_path / "logs.jsonl"
    result = CliRunner().invoke(
        app, ["import", str(source), "--format", fmt, "--out", str(out), *extra]
    )
    assert result.exit_code == 0, result.output
    assert "Imported 2 requests" in result.output
    assert "classify_ticket (2)" in result.output
    assert load_jsonl(out).ok
