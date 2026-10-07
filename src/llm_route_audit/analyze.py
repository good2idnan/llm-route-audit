"""Traffic profile: what the logged requests cost today, by task type and by model."""

import math
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any

from llm_route_audit.costs import PriceTable, UnknownModelError
from llm_route_audit.records import LogRecord

UNLABELLED = "(unlabelled)"
CHARS_PER_TOKEN = 4
DAYS_PER_MONTH = 30
SECONDS_PER_DAY = 86_400


def estimate_tokens(text: str) -> int:
    """Rough token count for logs without usage data (about 4 characters per token)."""
    return math.ceil(len(text) / CHARS_PER_TOKEN)


def percentile(values: list[float], pct: float) -> float | None:
    """Nearest-rank percentile, or None when there are no values."""
    if not values:
        return None
    ordered = sorted(values)
    rank = max(1, math.ceil(pct / 100 * len(ordered)))
    return ordered[rank - 1]


@dataclass
class Usage:
    input_tokens: int
    output_tokens: int
    cache_read_tokens: int
    cache_write_tokens: int
    estimated: bool


def usage_of(record: LogRecord) -> Usage:
    """Token usage from the log, estimating any counts the log left out."""
    estimated = False
    input_tokens = record.input_tokens
    if input_tokens is None:
        input_tokens = sum(estimate_tokens(m.content) for m in record.conversation())
        estimated = True
    output_tokens = record.output_tokens
    if output_tokens is None:
        output_tokens = estimate_tokens(record.response)
        estimated = True
    return Usage(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cache_read_tokens=record.cache_read_tokens or 0,
        cache_write_tokens=record.cache_write_tokens or 0,
        estimated=estimated,
    )


@dataclass
class GroupStats:
    """Totals for one slice of traffic: everything, one task type, or one model."""

    name: str
    requests: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    cost: float = 0.0
    priced_requests: int = 0
    latencies_ms: list[float] = field(default_factory=list)

    def add(self, usage: Usage, cost: float | None, latency_ms: float | None) -> None:
        self.requests += 1
        self.input_tokens += usage.input_tokens
        self.output_tokens += usage.output_tokens
        self.cache_read_tokens += usage.cache_read_tokens
        self.cache_write_tokens += usage.cache_write_tokens
        if cost is not None:
            self.cost += cost
            self.priced_requests += 1
        if latency_ms is not None:
            self.latencies_ms.append(latency_ms)

    @property
    def avg_input_tokens(self) -> float:
        return self.input_tokens / self.requests if self.requests else 0.0

    @property
    def avg_output_tokens(self) -> float:
        return self.output_tokens / self.requests if self.requests else 0.0

    @property
    def cost_per_request(self) -> float | None:
        return self.cost / self.priced_requests if self.priced_requests else None

    def to_dict(self, total: "GroupStats") -> dict[str, Any]:
        return {
            "name": self.name,
            "requests": self.requests,
            "share": self.requests / total.requests if total.requests else 0.0,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cache_read_tokens": self.cache_read_tokens,
            "cache_write_tokens": self.cache_write_tokens,
            "avg_input_tokens": self.avg_input_tokens,
            "avg_output_tokens": self.avg_output_tokens,
            "cost": self.cost,
            "cost_share": self.cost / total.cost if total.cost else 0.0,
            "cost_per_request": self.cost_per_request,
            "p50_latency_ms": percentile(self.latencies_ms, 50),
            "p95_latency_ms": percentile(self.latencies_ms, 95),
        }


@dataclass
class TrafficProfile:
    total: GroupStats
    by_task: list[GroupStats]
    by_model: list[GroupStats]
    first_seen: datetime
    last_seen: datetime
    prices_updated: date
    estimated_records: int
    unlabelled_records: int
    unpriced_models: dict[str, int]

    @property
    def span_days(self) -> float:
        return (self.last_seen - self.first_seen).total_seconds() / SECONDS_PER_DAY

    @property
    def monthly_cost(self) -> float | None:
        """Cost scaled to 30 days. Needs at least a day of logs to mean anything."""
        if self.span_days < 1:
            return None
        return self.total.cost / self.span_days * DAYS_PER_MONTH

    def to_dict(self) -> dict[str, Any]:
        return {
            "requests": self.total.requests,
            "first_seen": self.first_seen.isoformat(),
            "last_seen": self.last_seen.isoformat(),
            "span_days": self.span_days,
            "total": self.total.to_dict(self.total),
            "monthly_cost_estimate": self.monthly_cost,
            "prices_updated": self.prices_updated.isoformat(),
            "by_task": [g.to_dict(self.total) for g in self.by_task],
            "by_model": [g.to_dict(self.total) for g in self.by_model],
            "estimated_records": self.estimated_records,
            "unlabelled_records": self.unlabelled_records,
            "unpriced_models": self.unpriced_models,
        }


def _by_cost(groups: dict[str, GroupStats]) -> list[GroupStats]:
    return sorted(groups.values(), key=lambda g: (-g.cost, -g.requests, g.name))


def build_profile(records: list[LogRecord], prices: PriceTable) -> TrafficProfile:
    if not records:
        raise ValueError("no records to analyze")

    total = GroupStats("total")
    tasks: dict[str, GroupStats] = {}
    models: dict[str, GroupStats] = {}
    unpriced: dict[str, int] = {}
    estimated = unlabelled = 0

    for record in records:
        usage = usage_of(record)
        try:
            cost: float | None = prices.cost(
                record.model,
                input_tokens=usage.input_tokens,
                output_tokens=usage.output_tokens,
                cache_read_tokens=usage.cache_read_tokens,
                cache_write_tokens=usage.cache_write_tokens,
            )
        except UnknownModelError:
            cost = None
            unpriced[record.model] = unpriced.get(record.model, 0) + 1

        task = record.task_type or UNLABELLED
        estimated += usage.estimated
        unlabelled += record.task_type is None

        for group in (
            total,
            tasks.setdefault(task, GroupStats(task)),
            models.setdefault(record.model, GroupStats(record.model)),
        ):
            group.add(usage, cost, record.latency_ms)

    timestamps = [r.timestamp for r in records]
    return TrafficProfile(
        total=total,
        by_task=_by_cost(tasks),
        by_model=_by_cost(models),
        first_seen=min(timestamps),
        last_seen=max(timestamps),
        prices_updated=prices.updated,
        estimated_records=estimated,
        unlabelled_records=unlabelled,
        unpriced_models=unpriced,
    )
