"""Replay a sample of logged requests on candidate models and record what each one answers."""

import json
import time
from collections.abc import Callable
from concurrent.futures import CancelledError, ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from routeaudit.analyze import usage_of
from routeaudit.cache import ResultCache, request_key
from routeaudit.candidates import Candidate
from routeaudit.costs import PriceTable, UnknownModelError
from routeaudit.providers.base import Completion, Provider, ProviderError
from routeaudit.records import LogRecord


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
        cost=candidate_cost(
            prices,
            candidate,
            input_tokens=completion.input_tokens,
            output_tokens=completion.output_tokens,
            cache_read_tokens=completion.cache_read_tokens,
            cache_write_tokens=completion.cache_write_tokens,
        ),
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
) -> ReplayRun:
    """Run every (record, candidate) pair, reusing cached answers. Results keep job order."""
    jobs = [(record, candidate) for candidate in candidates for record in sample]
    results: dict[int, ReplayResult] = {}
    pending = []
    for i, (record, candidate) in enumerate(jobs):
        key = request_key(candidate, record.conversation())
        hit = cache.get(key)
        if hit is None:
            pending.append((i, record, candidate, key))
        else:
            completion, latency = hit
            results[i] = _completed(record, candidate, completion, latency, prices, cached=True)

    run = ReplayRun(results=[])
    done, total = len(results), len(jobs)
    if on_progress:
        on_progress(done, total)
    providers = {name: provider_for(name) for name in {c.provider for _, _, c, _ in pending}}

    def call(record: LogRecord, candidate: Candidate) -> tuple[Completion, float] | str:
        """Runs in a worker thread. Returns the answer, or the reason it was skipped.

        Workers record disabled candidates and fatal stops themselves, so a request already
        queued behind a failure is skipped instead of failing the same way.
        """
        if run.stopped_reason:
            return run.stopped_reason
        if candidate.label in run.disabled:
            return run.disabled[candidate.label]
        start = time.perf_counter()
        try:
            completion = providers[candidate.provider].complete(candidate, record.conversation())
        except ProviderError as e:
            if e.disable:
                run.disabled.setdefault(candidate.label, str(e))
            if e.fatal and run.stopped_reason is None:
                run.stopped_reason = str(e)
            raise
        return completion, (time.perf_counter() - start) * 1000

    with ThreadPoolExecutor(max_workers=max(1, concurrency)) as pool:
        futures = {pool.submit(call, r, c): (i, r, c, key) for i, r, c, key in pending}
        for future in as_completed(futures):
            i, record, candidate, key = futures[future]
            try:
                outcome = future.result()
            except CancelledError:
                results[i] = _base(record, candidate, "skipped", error=run.stopped_reason)
            except ProviderError as e:
                if e.fatal:
                    for other in futures:
                        other.cancel()
                results[i] = _base(record, candidate, "error", error=str(e))
            else:
                if isinstance(outcome, str):
                    results[i] = _base(record, candidate, "skipped", error=outcome)
                else:
                    completion, latency = outcome
                    cache.put(key, candidate, completion, latency)
                    results[i] = _completed(
                        record, candidate, completion, latency, prices, cached=False
                    )
            done += 1
            if on_progress:
                on_progress(done, total)

    run.results = [results[i] for i in range(len(jobs))]
    return run


def write_results(path: str | Path, results: list[ReplayResult]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for result in results:
            f.write(json.dumps(result.to_dict(), ensure_ascii=False) + "\n")
