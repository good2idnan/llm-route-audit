"""Candidates: the (model, effort) options a replay tests against the logged traffic."""

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

Provider = Literal["anthropic", "gemini", "openai", "ollama", "openrouter"]
# "none" and "minimal" exist on OpenAI models; providers reject levels a model doesn't support.
Effort = Literal["none", "minimal", "low", "medium", "high", "xhigh", "max"]

OLLAMA_PREFIX = "ollama/"
OPENROUTER_PREFIX = "openrouter/"
OPENAI_PREFIX = "openai/"
GEMINI_PREFIX = "gemini/"
LOCAL_HOSTS = ("localhost", "127.0.0.1", "[::1]", "0.0.0.0")


class Candidate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model: str = Field(min_length=1)
    effort: Effort | None = None
    provider: Provider | None = None
    max_tokens: int = Field(default=16_000, gt=0)
    # OpenAI-compatible servers other than OpenAI itself (provider: openai only).
    base_url: str | None = None
    api_key_env: str | None = None
    # A router (OpenRouter's Auto Router, Jev Router, a LiteLLM auto-router, ...) picks a
    # model per request. Its picks are recorded and the report audits them.
    router: bool = False
    # A model to price estimates and --max-spend with, for candidates without a fixed
    # price (routers): usually the most expensive model the router may pick.
    price_as: str | None = None

    @model_validator(mode="after")
    def _resolve_provider(self) -> "Candidate":
        if self.provider is None:
            if self.model.startswith(OLLAMA_PREFIX):
                self.provider = "ollama"
            elif self.model.startswith(OPENROUTER_PREFIX):
                self.provider = "openrouter"
            elif self.model.startswith(GEMINI_PREFIX):
                self.provider = "gemini"
            elif self.model.startswith(OPENAI_PREFIX) or self.base_url:
                self.provider = "openai"
            elif self.model.startswith("claude-"):
                self.provider = "anthropic"
            else:
                raise ValueError(
                    f"can't tell which provider serves '{self.model}'. "
                    "Set provider: anthropic, gemini, openai, openrouter or ollama."
                )
        if self.provider == "ollama" and self.effort is not None:
            raise ValueError(f"'{self.model}': effort is not supported for Ollama models")
        if (self.base_url or self.api_key_env) and self.provider != "openai":
            raise ValueError(f"'{self.model}': base_url and api_key_env need provider: openai")
        return self

    @property
    def api_model(self) -> str:
        """The model name the provider's API expects."""
        for prefix in (OLLAMA_PREFIX, OPENROUTER_PREFIX, OPENAI_PREFIX, GEMINI_PREFIX):
            if self.model.startswith(prefix):
                return self.model.removeprefix(prefix)
        return self.model

    @property
    def runs_locally(self) -> bool:
        """Ollama and OpenAI-compatible servers on this machine cost nothing per call."""
        if self.provider == "ollama":
            return True
        return bool(self.base_url) and any(host in self.base_url for host in LOCAL_HOSTS)

    @property
    def label(self) -> str:
        return f"{self.model} @ {self.effort}" if self.effort else self.model


class CandidateFile(BaseModel):
    model_config = ConfigDict(extra="forbid")

    candidates: list[Candidate] = Field(min_length=1)

    @model_validator(mode="after")
    def _unique(self) -> "CandidateFile":
        labels = [c.label for c in self.candidates]
        duplicates = sorted({label for label in labels if labels.count(label) > 1})
        if duplicates:
            raise ValueError(f"listed more than once: {', '.join(duplicates)}")
        return self


def load_candidates(path: str | Path) -> list[Candidate]:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    return CandidateFile.model_validate(data).candidates
