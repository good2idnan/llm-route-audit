"""Import OpenTelemetry GenAI spans from OTLP JSON files.

Works with files written by the OpenTelemetry Collector's file exporter (one OTLP export per
line) and with plain lists of spans. The prompts and answers only exist in a span when
message capture is turned on (the opt-in attributes gen_ai.input.messages,
gen_ai.output.messages and gen_ai.system_instructions).

Task types come from a span attribute, "task_type" by default. Set it in your application
when you start the span.
"""

import json
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from llm_route_audit.ingest.common import (
    AGENT_TURN,
    CHAT_ROLES,
    NOT_JSON,
    ImportResult,
    Skip,
    collect,
    is_unreadable,
    read_items,
    text_of,
)
from llm_route_audit.records import LogRecord

DEFAULT_TASK_ATTRIBUTE = "task_type"
STATUS_ERROR = 2
TOOL_PARTS = {"tool_call", "tool_call_response"}


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


def _parts_text(parts: Any) -> str:
    if isinstance(parts, list) and any(
        isinstance(p, dict) and p.get("type") in TOOL_PARTS for p in parts
    ):
        raise Skip(AGENT_TURN)
    return text_of(parts)


def _messages(attrs: dict[str, Any]) -> list[dict[str, str]]:
    messages = []
    system = _decoded(attrs.get("gen_ai.system_instructions"))
    if system:
        messages.append({"role": "system", "content": _parts_text(system)})
    inputs = _decoded(attrs.get("gen_ai.input.messages"))
    if not isinstance(inputs, list) or not inputs:
        raise Skip("no input messages recorded (turn on GenAI message capture)")
    for message in inputs:
        if not isinstance(message, dict) or message.get("role") not in CHAT_ROLES:
            raise Skip(AGENT_TURN)
        messages.append({"role": message["role"], "content": _parts_text(message.get("parts"))})
    return messages


def _output(attrs: dict[str, Any]) -> str:
    outputs = _decoded(attrs.get("gen_ai.output.messages"))
    if isinstance(outputs, list) and outputs and isinstance(outputs[0], dict):
        text = _parts_text(outputs[0].get("parts"))
        if text:
            return text
    raise Skip("no output messages recorded (turn on GenAI message capture)")


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
    text = _output(attrs)
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
