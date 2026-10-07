"""The log record format that every llm-route-audit command reads."""

from datetime import UTC, datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

Role = Literal["system", "user", "assistant", "tool"]


class ToolCall(BaseModel):
    """A model's request to run a tool."""

    model_config = ConfigDict(extra="forbid")

    id: str | None = None
    name: str = Field(min_length=1)
    arguments: dict[str, Any] = Field(default_factory=dict)


class ToolDef(BaseModel):
    """A tool the model may call. `parameters` is a JSON schema."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1)
    description: str = ""
    parameters: dict[str, Any] = Field(default_factory=lambda: {"type": "object", "properties": {}})


class Message(BaseModel):
    """One message. Assistant messages may call tools; "tool" messages carry a tool's result."""

    model_config = ConfigDict(extra="forbid")

    role: Role
    content: str = ""
    tool_calls: list[ToolCall] | None = None
    tool_call_id: str | None = None  # on "tool" messages: which call this result answers
    name: str | None = None  # on "tool" messages: the tool's name


class LogRecord(BaseModel):
    """One logged model call and the response it got in production.

    In an agent, every step is one call: `messages` hold the history so far (including tool
    calls and tool results), and the response is either text or new tool calls.

    Token fields follow the Anthropic usage convention: ``input_tokens`` counts
    uncached input only, and cache reads and writes are counted separately.
    Unknown fields are ignored so existing logs can be used with light mapping.
    """

    model_config = ConfigDict(extra="ignore")

    id: str = Field(min_length=1)
    timestamp: datetime
    model: str = Field(min_length=1)
    messages: list[Message] | None = None
    prompt: str | None = None
    response: str = ""
    response_tool_calls: list[ToolCall] | None = None
    tools: list[ToolDef] | None = None
    session_id: str | None = None  # groups the steps of one agent session
    input_tokens: int | None = Field(default=None, ge=0)
    output_tokens: int | None = Field(default=None, ge=0)
    cache_read_tokens: int | None = Field(default=None, ge=0)
    cache_write_tokens: int | None = Field(default=None, ge=0)
    latency_ms: float | None = Field(default=None, ge=0)
    task_type: str | None = None
    outcome: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("timestamp")
    @classmethod
    def _assume_utc(cls, value: datetime) -> datetime:
        """Timestamps without a timezone are treated as UTC, so all records compare cleanly."""
        return value.replace(tzinfo=UTC) if value.tzinfo is None else value

    @model_validator(mode="after")
    def _has_input(self) -> "LogRecord":
        if not self.messages and not self.prompt:
            raise ValueError("record needs either 'messages' or 'prompt'")
        return self

    def conversation(self) -> list[Message]:
        """The request as a message list, whichever form the log used."""
        if self.messages:
            return list(self.messages)
        return [Message(role="user", content=self.prompt or "")]

    @property
    def is_agent_step(self) -> bool:
        """True for calls that involve tools: defined tools, tool history, or a tool call."""
        return bool(
            self.tools
            or self.response_tool_calls
            or any(m.role == "tool" or m.tool_calls for m in self.conversation())
        )
