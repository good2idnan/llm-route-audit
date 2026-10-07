"""Import OpenTelemetry GenAI spans from OTLP JSON files.

Works with files written by the OpenTelemetry Collector's file exporter (one OTLP export per
line) and with plain lists of spans. The prompts and answers only exist in a span when
message capture is turned on (the opt-in attributes gen_ai.input.messages,
gen_ai.output.messages and gen_ai.system_instructions).

Task types come from a span attribute, "task_type" by default. Set it in your application
when you start the span. Agent steps keep their tool calls and tool results (tool_call and
tool_call_response parts) and tool definitions (gen_ai.tool.definitions); the steps of one
conversation (gen_ai.conversation.id, or else the trace) form one agent session.
"""

import json
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from llm_route_audit.ingest.common import (
    CHAT_ROLES,
    NO_RESPONSE,
    NOT_JSON,
    UNSUPPORTED_ROLE,
    ImportResult,
    Skip,
    collect,
    is_unreadable,
    read_items,
    text_of,
    tool_calls_of,
    tool_defs,
    tool_result,
)
from llm_route_audit.records import LogRecord

DEFAULT_TASK_ATTRIBUTE = "task_type"
STATUS_ERROR = 2
SKIPPED_PARTS = {"reasoning"}  # the model's thinking: neither request nor answer


def _value(value: dict[str, Any]) -> Any:
    """Decode an OTLP AnyValue."""
    if "stringValue" in value:
        return value["stringValue"]
    if "intValue" in value:
        return int(value["intValue"])
    if "doubleValue" in value:
        return float(value["doubleValue"])
    if "boolValue" in value:
        return bool(value["boolValue"])
    if "arrayValue" in value:
        return [_value(v) for v in value["arrayValue"].get("values", [])]
    if "kvlistValue" in value:
        return {
            kv["key"]: _value(kv.get("value", {})) for kv in value["kvlistValue"].get("values", [])
        }
    return None


def attributes(span: dict[str, Any]) -> dict[str, Any]:
    raw = span.get("attributes") or []
    if isinstance(raw, dict):  # some exporters write plain key/value maps
        return raw
    return {a["key"]: _value(a.get("value") or {}) for a in raw if "key" in a}


def spans(item: Any) -> Iterator[Any]:
    """Every span in one OTLP export (or the item itself if it is already a span)."""
    if isinstance(item, dict) and "resourceSpans" in item:
        for resource in item["resourceSpans"] or []:
            scopes = resource.get("scopeSpans") or resource.get("instrumentationLibrarySpans") or []
            for scope in scopes:
                yield from scope.get("spans") or []
    else:
        yield item


def _decoded(value: Any) -> Any:
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            raise Skip("message attributes are not valid JSON") from None
    return value


def _part_type(part: Any) -> Any:
    return part.get("type") if isinstance(part, dict) else None


def _converted(message: Any) -> list[dict[str, Any]]:
    """One GenAI message as record messages: tool results become "tool" messages, tool
    calls go on the assistant message."""
    if not isinstance(message, dict) or message.get("role") not in CHAT_ROLES:
        raise Skip(UNSUPPORTED_ROLE)
    parts = message.get("parts")
    if not isinstance(parts, list):
        return [{"role": message["role"], "content": text_of(parts)}]
    out = [
        tool_result(p.get("id"), p.get("response"))
        for p in parts
        if _part_type(p) == "tool_call_response"
    ]
    calls = tool_calls_of([p for p in parts if _part_type(p) == "tool_call"])
    rest = [
        p for p in parts if _part_type(p) not in SKIPPED_PARTS | {"tool_call", "tool_call_response"}
    ]
    if rest or calls or not out:
        if calls and message["role"] != "assistant":
            raise Skip("tool calls outside an assistant message")
        converted: dict[str, Any] = {"role": message["role"], "content": text_of(rest)}
        if calls:
            converted["tool_calls"] = calls
        out.append(converted)
    return out


def _messages(attrs: dict[str, Any]) -> list[dict[str, Any]]:
    messages = []
    system = _decoded(attrs.get("gen_ai.system_instructions"))
    if system:
        messages.append({"role": "system", "content": text_of(system)})
    inputs = _decoded(attrs.get("gen_ai.input.messages"))
    if not isinstance(inputs, list) or not inputs:
        raise Skip("no input messages recorded (turn on GenAI message capture)")
    for message in inputs:
        messages += _converted(message)
    return messages


def _output(attrs: dict[str, Any]) -> tuple[str, list[dict[str, Any]] | None]:
    outputs = _decoded(attrs.get("gen_ai.output.messages"))
    if not isinstance(outputs, list) or not outputs or not isinstance(outputs[0], dict):
        raise Skip("no output messages recorded (turn on GenAI message capture)")
    reply = _converted({**outputs[0], "role": "assistant"})[-1]
    if not reply["content"] and not reply.get("tool_calls"):
        raise Skip(NO_RESPONSE)
    return reply["content"], reply.get("tool_calls")


def _time(nanos: Any) -> datetime:
    return datetime.fromtimestamp(int(nanos) / 1e9, UTC)


def convert(span: Any, task_attribute: str = DEFAULT_TASK_ATTRIBUTE) -> LogRecord:
    """One GenAI span as a LogRecord, or Skip with the reason."""
    if not isinstance(span, dict) or is_unreadable(span):
        raise Skip(NOT_JSON)
    attrs = attributes(span)
    model = attrs.get("gen_ai.response.model") or attrs.get("gen_ai.request.model")
    if not model:
        raise Skip("not a model call (no gen_ai model attribute)")
    if (span.get("status") or {}).get("code") in (STATUS_ERROR, "STATUS_CODE_ERROR"):
        raise Skip("failed request")

    messages = _messages(attrs)
    text, calls = _output(attrs)
    input_tokens = attrs.get("gen_ai.usage.input_tokens")
    cached = int(attrs.get("gen_ai.usage.cache_read.input_tokens") or 0)
    try:
        start = _time(span["startTimeUnixNano"])
        end = span.get("endTimeUnixNano")
        return LogRecord.model_validate(
            {
                "id": str(span.get("spanId") or ""),
                "timestamp": start,
                "model": str(model).removeprefix("anthropic/"),
                "task_type": attrs.get(task_attribute) or None,
                "messages": messages,
                "response": text,
                "response_tool_calls": calls,
                "tools": tool_defs(_decoded(attrs.get("gen_ai.tool.definitions"))),
                "session_id": attrs.get("gen_ai.conversation.id") or span.get("traceId"),
                "input_tokens": max(0, input_tokens - cached) if input_tokens is not None else None,
                "output_tokens": attrs.get("gen_ai.usage.output_tokens"),
                "cache_read_tokens": cached or None,
                "latency_ms": (_time(end) - start).total_seconds() * 1000 if end else None,
                "metadata": {"source": "otel", "trace_id": span.get("traceId")},
            }
        )
    except (KeyError, ValidationError, TypeError, ValueError) as e:
        raise Skip(f"missing or invalid fields ({type(e).__name__})") from None


def _all_spans(source: Path) -> Iterator[tuple[str, Any]]:
    for where, item in read_items(source):
        if is_unreadable(item):
            yield where, item
            continue
        for i, span in enumerate(spans(item), start=1):
            yield f"{where}/span{i}", span


def import_otel(source: str | Path, task_attribute: str = DEFAULT_TASK_ATTRIBUTE) -> ImportResult:
    return collect(_all_spans(Path(source)), lambda span: convert(span, task_attribute))
