"""Translate tool definitions, tool calls and tool results into each provider's format."""

import json
from typing import Any

from llm_route_audit.records import Message, ToolCall, ToolDef


def _call_id(call: ToolCall, index: int, prefix: str) -> str:
    return call.id or f"{prefix}_{index}"


def parse_arguments(raw: Any) -> dict[str, Any]:
    """Tool arguments as a dict, whether they arrive as a JSON string or already parsed."""
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        try:
            value = json.loads(raw) if raw.strip() else {}
        except ValueError:
            return {"_raw": raw}
        return value if isinstance(value, dict) else {"_value": value}
    return {} if raw is None else {"_value": raw}


# --- OpenAI Chat Completions (also OpenRouter and OpenAI-compatible servers) -------------------


def openai_messages(messages: list[Message]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for m in messages:
        if m.role == "tool":
            out.append({"role": "tool", "tool_call_id": m.tool_call_id or "", "content": m.content})
        elif m.role == "assistant" and m.tool_calls:
            out.append(
                {
                    "role": "assistant",
                    "content": m.content or None,
                    "tool_calls": [
                        {
                            "id": _call_id(c, i, "call"),
                            "type": "function",
                            "function": {"name": c.name, "arguments": json.dumps(c.arguments)},
                        }
                        for i, c in enumerate(m.tool_calls)
                    ],
                }
            )
        else:
            out.append({"role": m.role, "content": m.content})
    return out


def openai_tools(tools: list[ToolDef]) -> list[dict[str, Any]]:
    return [
        {
            "type": "function",
            "function": {"name": t.name, "description": t.description, "parameters": t.parameters},
        }
        for t in tools
    ]


def parse_openai_tool_calls(raw: Any) -> list[ToolCall] | None:
    calls = []
    for item in raw or []:
        function = (item or {}).get("function") or {}
        calls.append(
            ToolCall(
                id=item.get("id"),
                name=function.get("name") or "unknown",
                arguments=parse_arguments(function.get("arguments")),
            )
        )
    return calls or None


# --- Anthropic Messages ------------------------------------------------------------------------


def anthropic_messages(messages: list[Message]) -> tuple[str, list[dict[str, Any]]]:
    """(system text, messages). Tool results go into the next user turn as tool_result
    blocks; tool calls become tool_use blocks on the assistant turn."""
    system = "\n\n".join(m.content for m in messages if m.role == "system")
    out: list[dict[str, Any]] = []
    for m in messages:
        if m.role == "system":
            continue
        if m.role == "tool":
            block = {
                "type": "tool_result",
                "tool_use_id": m.tool_call_id or "",
                "content": m.content,
            }
            last = out[-1] if out else None
            if (
                last
                and last["role"] == "user"
                and isinstance(last["content"], list)
                and all(b.get("type") == "tool_result" for b in last["content"])
            ):
                last["content"].append(block)
            else:
                out.append({"role": "user", "content": [block]})
        elif m.role == "assistant" and m.tool_calls:
            blocks: list[dict[str, Any]] = (
                [{"type": "text", "text": m.content}] if m.content else []
            )
            blocks += [
                {
                    "type": "tool_use",
                    "id": _call_id(c, i, "toolu"),
                    "name": c.name,
                    "input": c.arguments,
                }
                for i, c in enumerate(m.tool_calls)
            ]
            out.append({"role": "assistant", "content": blocks})
        else:
            out.append({"role": m.role, "content": m.content})
    return system, out


def anthropic_tools(tools: list[ToolDef]) -> list[dict[str, Any]]:
    return [
        {"name": t.name, "description": t.description, "input_schema": t.parameters} for t in tools
    ]


def parse_anthropic_tool_calls(content: Any) -> list[ToolCall] | None:
    calls = [
        ToolCall(id=block.id, name=block.name, arguments=parse_arguments(block.input))
        for block in content or []
        if getattr(block, "type", None) == "tool_use"
    ]
    return calls or None


# --- Ollama ------------------------------------------------------------------------------------


def ollama_messages(messages: list[Message]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for m in messages:
        if m.role == "assistant" and m.tool_calls:
            out.append(
                {
                    "role": "assistant",
                    "content": m.content,
                    "tool_calls": [
                        {"function": {"name": c.name, "arguments": c.arguments}}
                        for c in m.tool_calls
                    ],
                }
            )
        elif m.role == "tool":
            entry = {"role": "tool", "content": m.content}
            if m.name:
                entry["tool_name"] = m.name
            out.append(entry)
        else:
            out.append({"role": m.role, "content": m.content})
    return out


def render_tool_calls(calls: list[ToolCall] | None) -> str:
    """Tool calls as readable text, for the judge."""
    return "\n".join(
        f"CALL {c.name}({json.dumps(c.arguments, ensure_ascii=False, sort_keys=True)})"
        for c in calls or []
    )
