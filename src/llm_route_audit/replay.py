"""Replay a sample of logged requests on candidate models and record what each one answers."""

import json
import math
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from llm_route_audit.analyze import estimate_tokens, usage_of
from llm_route_audit.cache import ResultCache, request_key
from llm_route_audit.candidates import Candidate
from llm_route_audit.costs import PriceTable, UnknownModelError
from llm_route_audit.providers.base import Completion, Provider
from llm_route_audit.records import LogRecord
from llm_route_audit.runner import Job, execute


def candidate_cost(
    prices: PriceTable,
    candidate: Candidate,
    *,
    input_tokens: int,
    output_tokens: int,
    cache_read_tokens: int = 0,
    cache_write_tokens: int = 0,
) -> float | None:
    """Cost of one request on a candidate. Unpriced local Ollama models count as free."""
    try:
        return prices.cost(
            candidate.model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cache_read_tokens=cache_read_tokens,
            cache_write_tokens=cache_write_tokens,
        )
    except UnknownModelError:
        return 0.0 if candidate.provider == "ollama" else None


def completion_cost(
    prices: PriceTable, candidate: Candidate, completion: Completion
) -> float | None:
    """What a call cost: the provider's own figure when it reports one, else from prices."""
    if completion.cost is not None:
        return completion.cost
    return candidate_cost(
        prices,
        candidate,
        input_tokens=completion.input_tokens,
        output_tokens=completion.output_tokens,
        cache_read_tokens=completion.cache_read_tokens,
        cache_write_tokens=completion.cache_write_tokens,
    )


# Token counts estimated from text length can run low, and every message adds a few tokens.
INPUT_SAFETY_MARGIN = 1.5
PER_MESSAGE_TOKENS = 20


def worst_case_cost(prices: PriceTable, job: Job) -> float | None:
    """The most a call could cost: generous input estimate plus every allowed output token."""
    input_tokens = sum(estimate_tokens(m.content) + PER_MESSAGE_TOKENS for m in job.messages)
    return candidate_cost(
        prices,
        job.candidate,
        input_tokens=math.ceil(input_tokens * INPUT_SAFETY_MARGIN),
        output_tokens=job.candidate.max_tokens,
    )


def logged_cost(prices: PriceTable, records: list[LogRecord]) -> float | None:
    """What the sampled requests cost as originally logged, for comparison."""
    total = 0.0
    for record in records:
        usage = usage_of(record)
        try:
            total += prices.cost(
                record.model,
                input_tokens=usage.input_tokens,
                output_tokens=usage.output_tokens,
                cache_read_tokens=usage.cache_read_tokens,
                cache_write_tokens=usage.cache_write_tokens,
            )
        except UnknownModelError:
            return None
    return total


@dataclass
class CandidateEstimate:
    candidate: Candidate
    requests: int
    cached: int
    cost: float | None  # for the requests not already cached; None if the model has no price


def estimate(
    sample: list[LogRecord], candidates: list[Candidate], prices: PriceTable, cache: ResultCache
) -> list[CandidateEstimate]:
    """Expected spend, assuming each candidate reads the same input and writes as much as the
    original answer did. Real cost varies, mostly with how much the model thinks."""
    estimates = []
    for candidate in candidates:
        cost: float | None = 0.0
        cached = 0
        for record in sample:
            if cache.get(request_key(candidate, record.conversation())) is not None:
                cached += 1
                continue
            usage = usage_of(record)
            one = candidate_cost(
                prices,
                candidate,
                input_tokens=usage.input_tokens
                + usage.cache_read_tokens
                + usage.cache_write_tokens,
                output_tokens=usage.output_tokens,
            )
            cost = None if one is None or cost is None else cost + one
        estimates.append(CandidateEstimate(candidate, len(sample), cached, cost))
    return estimates


@dataclass
class ReplayResult:
    record_id: str
    task_type: str | None
    model: str
    effort: str | None
    provider: str | None
    status: str  # ok | refusal | truncated | error | skipped
    response: str | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    cache_read_tokens: int | None = None
    cache_write_tokens: int | None = None
    cost: float | None = None
    latency_ms: float | None = None
    cached: bool = False
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ReplayRun:
    results: list[ReplayResult]
    stopped_reason: str | None = None
    disabled: dict[str, str] = field(default_factory=dict)
    held_back: int = 0

    @property
    def spent(self) -> float:
        """Money spent on new API calls in this run (cache hits cost nothing)."""
        return sum(r.cost or 0.0 for r in self.results if not r.cached)


def _base(record: LogRecord, candidate: Candidate, status: str, **extra: Any) -> ReplayResult:
    return ReplayResult(
        record_id=record.id,
        task_type=record.task_type,
        model=candidate.model,
        effort=candidate.effort,
        provider=candidate.provider,
        status=status,
        **extra,
    )


def _completed(
    record: LogRecord,
    candidate: Candidate,
    completion: Completion,
    latency_ms: float | None,
    prices: PriceTable,
    cached: bool,
) -> ReplayResult:
    return _base(
        record,
        candidate,
        completion.status,
        response=completion.text,
        input_tokens=completion.input_tokens,
        output_tokens=completion.output_tokens,
        cache_read_tokens=completion.cache_read_tokens,
        cache_write_tokens=completion.cache_write_tokens,
        cost=completion_cost(prices, candidate, completion),
        latency_ms=latency_ms,
        cached=cached,
    )


def run_replay(
    sample: list[LogRecord],
    candidates: list[Candidate],
    cache: ResultCache,
    prices: PriceTable,
    provider_for: Callable[[str], Provider],
    concurrency: int = 4,
    on_progress: Callable[[int, int], None] | None = None,
    max_spend: float | None = None,
) -> ReplayRun:
    """Run every (record, candidate) pair, reusing cached answers. Results keep job order.

    With `max_spend`, the run never spends more than that many USD on new calls.
    """
    pairs = [(record, candidate) for candidate in candidates for record in sample]
    execution = execute(
        [Job(candidate, record.conversation()) for record, candidate in pairs],
        cache,
        provider_for,
        concurrency=concurrency,
        on_progress=on_progress,
        max_spend=max_spend,
        worst_case=lambda job: worst_case_cost(prices, job),
        actual_cost=lambda job, completion: completion_cost(prices, job.candidate, completion),
    )
    results = []
    for (record, candidate), outcome in zip(pairs, execution.outcomes, strict=True):
        if outcome.completion is None:
            results.append(_base(record, candidate, outcome.status, error=outcome.error))
        else:
            results.append(
                _completed(
                    record,
                    candidate,
                    outcome.completion,
                    outcome.latency_ms,
                    prices,
                    outcome.cached,
                )
            )
    return ReplayRun(results, execution.stopped_reason, execution.disabled, execution.held_back)


def write_results(path: str | Path, results: list[ReplayResult]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for result in results:
            f.write(json.dumps(result.to_dict(), ensure_ascii=False) + "\n")


def load_results(path: str | Path) -> list[ReplayResult]:
    with Path(path).open(encoding="utf-8") as f:
        return [ReplayResult(**json.loads(line)) for line in f if line.strip()]
