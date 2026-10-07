from dataclasses import dataclass
from typing import Protocol

from llm_route_audit.candidates import Candidate
from llm_route_audit.records import Message, ToolCall, ToolDef


@dataclass
class Completion:
    text: str
    input_tokens: int
    output_tokens: int
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    # "refusal" and "truncated" are kept apart from "ok" so grading can count them.
    status: str = "ok"
    # What the provider actually charged, when it reports it (OpenRouter does).
    cost: float | None = None
    # Tools the model asked to call instead of (or as well as) answering in text.
    tool_calls: list[ToolCall] | None = None
    # The model that actually answered, as the provider reports it. For a router this is
    # the model it picked.
    served_model: str | None = None


class ProviderError(Exception):
    """A failed request.

    `fatal` stops the whole replay (bad key, provider unreachable). `disable` stops the
    candidate that hit it (unknown model, unsupported setting), since every later request
    for that candidate would fail the same way.
    """

    def __init__(self, message: str, *, fatal: bool = False, disable: bool = False) -> None:
        super().__init__(message)
        self.fatal = fatal
        self.disable = disable


class Provider(Protocol):
    def complete(
        self, candidate: Candidate, messages: list[Message], tools: list[ToolDef] | None = None
    ) -> Completion: ...
