"""The log record format that every routeaudit command reads."""

from datetime import UTC, datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

Role = Literal["system", "user", "assistant"]


class Message(BaseModel):
    model_config = ConfigDict(extra="forbid")

    role: Role
    content: str


class LogRecord(BaseModel):
    """One logged request and the response it got in production.

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
    response: str
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
