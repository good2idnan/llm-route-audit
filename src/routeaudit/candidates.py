"""Candidates: the (model, effort) options a replay tests against the logged traffic."""

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

Provider = Literal["anthropic", "ollama"]
Effort = Literal["low", "medium", "high", "xhigh", "max"]

OLLAMA_PREFIX = "ollama/"


class Candidate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model: str = Field(min_length=1)
    effort: Effort | None = None
    provider: Provider | None = None
    max_tokens: int = Field(default=16_000, gt=0)

    @model_validator(mode="after")
    def _resolve_provider(self) -> "Candidate":
        if self.provider is None:
            if self.model.startswith(OLLAMA_PREFIX):
                self.provider = "ollama"
            elif self.model.startswith("claude-"):
                self.provider = "anthropic"
            else:
                raise ValueError(
                    f"can't tell which provider serves '{self.model}'. "
                    "Set provider: anthropic or ollama."
                )
        if self.provider == "ollama" and self.effort is not None:
            raise ValueError(f"'{self.model}': effort is not supported for Ollama models")
        return self

    @property
    def api_model(self) -> str:
        """The model name the provider's API expects."""
        return self.model.removeprefix(OLLAMA_PREFIX)

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
