"""Price tables and per-request cost, including prompt-cache pricing."""

from datetime import date
from importlib.resources import files
from pathlib import Path

import yaml
from pydantic import BaseModel, Field

PER_MILLION = 1_000_000


class UnknownModelError(KeyError):
    pass


class ModelPrice(BaseModel):
    """USD per 1M tokens. Missing cache prices fall back to the full input price."""

    input: float = Field(ge=0)
    output: float = Field(ge=0)
    cache_read: float | None = Field(default=None, ge=0)
    cache_write: float | None = Field(default=None, ge=0)


class PriceTable(BaseModel):
    currency: str = "USD"
    updated: date
    source: str | None = None
    models: dict[str, ModelPrice]

    def price(self, model: str) -> ModelPrice:
        try:
            return self.models[model]
        except KeyError:
            raise UnknownModelError(
                f"no price for model '{model}'. Add it to your prices file."
            ) from None

    def cost(
        self,
        model: str,
        *,
        input_tokens: int = 0,
        output_tokens: int = 0,
        cache_read_tokens: int = 0,
        cache_write_tokens: int = 0,
    ) -> float:
        p = self.price(model)
        cache_read = p.input if p.cache_read is None else p.cache_read
        cache_write = p.input if p.cache_write is None else p.cache_write
        total = (
            input_tokens * p.input
            + output_tokens * p.output
            + cache_read_tokens * cache_read
            + cache_write_tokens * cache_write
        )
        return total / PER_MILLION


def load_prices(path: str | Path | None = None) -> PriceTable:
    """Load a prices file, or the bundled default table when no path is given."""
    if path is None:
        text = files("llm_route_audit").joinpath("data/prices.yaml").read_text(encoding="utf-8")
    else:
        text = Path(path).read_text(encoding="utf-8")
    return PriceTable.model_validate(yaml.safe_load(text))
