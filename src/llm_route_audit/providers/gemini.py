"""Replay requests on Google's Gemini models through the Gemini API (generateContent).

Candidates are named "gemini/<model>", e.g. "gemini/gemini-3-flash". The key comes from
GEMINI_API_KEY (or GOOGLE_API_KEY). Effort maps to Gemini's thinking level: none turns
thinking off; minimal, low, medium and high pass through; xhigh and max use high.
"""

import json
import os
from typing import Any

from llm_route_audit.candidates import Candidate
from llm_route_audit.providers.base import Completion, ProviderError
from llm_route_audit.providers.http import HTTPFailure, post_json
from llm_route_audit.providers.tooling import parse_arguments
from llm_route_audit.records import Message, ToolCall, ToolDef

GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta"
KEY_ENVS = ("GEMINI_API_KEY", "GOOGLE_API_KEY")
# Logged tool calls came from another model, so they carry no Gemini thought signature.
# Google documents this value for exactly that case: it skips the signature check.
FOREIGN_SIGNATURE = "skip_thought_signature_validator"
REFUSALS = {"SAFETY", "PROHIBITED_CONTENT", "BLOCKLIST", "SPII", "RECITATION", "IMAGE_SAFETY"}
THINKING_LEVEL = {"xhigh": "high", "max": "high"}


def _result_object(content: str) -> dict[str, Any]:
    """Gemini wants a tool result as an object; plain text goes under "result"."""
    try:
        value = json.loads(content)
    except ValueError:
        return {"result": content}
    return value if isinstance(value, dict) else {"result": value}


def gemini_contents(messages: list[Message]) -> tuple[str, list[dict[str, Any]]]:
    """(system instructions, contents). Tool results join one user turn, as Gemini expects."""
    system = "\n\n".join(m.content for m in messages if m.role == "system")
    names: dict[str, str] = {}  # tool call id -> tool name, to label results
    contents: list[dict[str, Any]] = []
    for m in messages:
        if m.role == "system":
            continue
        if m.role == "assistant":
            parts: list[dict[str, Any]] = [{"text": m.content}] if m.content else []
            for call in m.tool_calls or []:
                if call.id:
                    names[call.id] = call.name
                parts.append(
                    {
                        "functionCall": {"name": call.name, "args": call.arguments},
                        "thoughtSignature": FOREIGN_SIGNATURE,
                    }
                )
            contents.append({"role": "model", "parts": parts or [{"text": ""}]})
        elif m.role == "tool":
            name = m.name or names.get(m.tool_call_id or "", "tool")
            part = {"functionResponse": {"name": name, "response": _result_object(m.content)}}
            last = contents[-1] if contents else None
            if (
                last
                and last["role"] == "user"
                and all("functionResponse" in p for p in last["parts"])
            ):
                last["parts"].append(part)
            else:
                contents.append({"role": "user", "parts": [part]})
        else:
            contents.append({"role": "user", "parts": [{"text": m.content}]})
    return system, contents


def build_request(
    candidate: Candidate, messages: list[Message], tools: list[ToolDef] | None = None
) -> dict[str, Any]:
    system, contents = gemini_contents(messages)
    config: dict[str, Any] = {"maxOutputTokens": candidate.max_tokens}
    if candidate.effort == "none":
        config["thinkingConfig"] = {"thinkingBudget": 0}
    elif candidate.effort:
        level = THINKING_LEVEL.get(candidate.effort, candidate.effort)
        config["thinkingConfig"] = {"thinkingLevel": level}
    request: dict[str, Any] = {"contents": contents, "generationConfig": config}
    if system:
        request["systemInstruction"] = {"parts": [{"text": system}]}
    if tools:
        request["tools"] = [
            {
                "functionDeclarations": [
                    {
                        "name": t.name,
                        "description": t.description,
                        "parametersJsonSchema": t.parameters,
                    }
                    for t in tools
                ]
            }
        ]
    return request


def parse_response(data: dict[str, Any]) -> Completion:
    usage = data.get("usageMetadata") or {}
    cached = usage.get("cachedContentTokenCount") or 0
    input_tokens = max(0, (usage.get("promptTokenCount") or 0) - cached)
    # Thinking is billed as output.
    output_tokens = (usage.get("candidatesTokenCount") or 0) + (
        usage.get("thoughtsTokenCount") or 0
    )
    candidates = data.get("candidates") or []
    if not candidates:
        reason = (data.get("promptFeedback") or {}).get("blockReason")
        if reason:
            return Completion(
                f"blocked: {reason}", input_tokens, output_tokens, cached, status="refusal"
            )
        raise ProviderError("Gemini returned no answer")
    first = candidates[0]
    texts, calls = [], []
    for part in (first.get("content") or {}).get("parts") or []:
        if part.get("thought"):
            continue  # the model's thinking, not the answer
        if "functionCall" in part:
            call = part["functionCall"]
            calls.append(
                ToolCall(
                    id=call.get("id"),
                    name=call.get("name") or "unknown",
                    arguments=parse_arguments(call.get("args")),
                )
            )
        elif "text" in part:
            texts.append(part["text"])
    finish = first.get("finishReason")
    status = "truncated" if finish == "MAX_TOKENS" else "refusal" if finish in REFUSALS else "ok"
    return Completion(
        text="".join(texts),
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cache_read_tokens=cached,
        status=status,
        tool_calls=calls or None,
        served_model=data.get("modelVersion"),
    )


def _bad_key(e: HTTPFailure) -> bool:
    """Gemini answers a bad key with 400 (API_KEY_INVALID) or 403, not 401."""
    if e.status in (401, 403):
        return True
    return e.status == 400 and ("API_KEY_INVALID" in json.dumps(e.body) or "API key" in str(e))


class GeminiProvider:
    def __init__(self, base_url: str | None = None) -> None:
        self.base_url = (base_url or GEMINI_URL).rstrip("/")
        self._sleep = None  # tests can replace the backoff sleep

    def complete(
        self, candidate: Candidate, messages: list[Message], tools: list[ToolDef] | None = None
    ) -> Completion:
        key = next((os.environ[k] for k in KEY_ENVS if os.environ.get(k)), None)
        if not key:
            raise ProviderError(
                "GEMINI_API_KEY is not set. Add it to the .env file in your project folder.",
                fatal=True,
            )
        extra = {"sleep": self._sleep} if self._sleep else {}
        try:
            data = post_json(
                f"{self.base_url}/models/{candidate.api_model}:generateContent",
                build_request(candidate, messages, tools),
                {"x-goog-api-key": key},
                **extra,
            )
        except HTTPFailure as e:
            if _bad_key(e):
                raise ProviderError("Gemini rejected the API key.", fatal=True) from e
            if e.status in (400, 404):
                raise ProviderError(f"{candidate.label}: {e}", disable=True) from e
            raise ProviderError(f"Gemini API error {e.status}: {e}") from e
        return parse_response(data)
