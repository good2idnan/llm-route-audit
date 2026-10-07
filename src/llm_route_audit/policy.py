"""Read the routing policy written by `llm-route-audit export`."""

from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field

from llm_route_audit.candidates import Effort
from llm_route_audit.candidates import Provider as ProviderName


class Route(BaseModel):
    model_config = ConfigDict(extra="ignore")

    model: str
    effort: Effort | None = None
    provider: ProviderName | None = None
    # Set on routes that switched to a cheaper model: the model it replaced, and the pass
    # rate the audit measured against it.
    reference: str | None = None
    expected_pass_rate: float | None = Field(default=None, ge=0, le=1)

    @property
    def label(self) -> str:
        return f"{self.model} @ {self.effort}" if self.effort else self.model

    @property
    def switched(self) -> bool:
        return self.reference is not None and self.expected_pass_rate is not None


class Policy(BaseModel):
    model_config = ConfigDict(extra="ignore")

    version: int = 1
    default: Route
    routes: dict[str, Route] = Field(default_factory=dict)


def load_policy(path: str | Path) -> Policy:
    return Policy.model_validate(yaml.safe_load(Path(path).read_text(encoding="utf-8")))
