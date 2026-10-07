"""Run many model calls in parallel with caching, skipping and safe stopping.

Used by replay (candidate answers) and grading (judge calls).
"""

import threading
import time
from collections.abc import Callable
from concurrent.futures import CancelledError, ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field

from routeaudit.cache import ResultCache, request_key
from routeaudit.candidates import Candidate
from routeaudit.providers.base import Completion, Provider, ProviderError
from routeaudit.records import Message


@dataclass
class Job:
    candidate: Candidate
    messages: list[Message]


@dataclass
class Outcome:
    """A finished job. `completion` is set when the model answered (status ok)."""

    status: str  # ok | error | skipped
    completion: Completion | None = None
    latency_ms: float | None = None
    cached: bool = False
    error: str | None = None


@dataclass
class Execution:
    outcomes: list[Outcome]
    stopped_reason: str | None = None
    disabled: dict[str, str] = field(default_factory=dict)
    held_back: int = 0  # calls not sent because they could have broken max_spend


class SpendGuard:
    """Keeps money spent plus money reserved for calls in flight under a hard limit.

    Each call reserves its worst-case cost before it is sent and settles to the real cost
    afterwards, so even parallel calls can never push the total past the limit.
    """

    def __init__(self, limit: float) -> None:
        self.limit = limit
        self.spent = 0.0
        self.reserved = 0.0
        self.in_flight = 0
        self._cond = threading.Condition()

    def reserve(self, amount: float) -> bool:
        """Reserve `amount`, waiting for calls in flight to settle if that frees enough room.
        Returns False when the call can't fit even with nothing else in flight."""
        with self._cond:
            while self.spent + self.reserved + amount > self.limit:
                if self.in_flight == 0:
                    return False
                self._cond.wait()
            self.reserved += amount
            self.in_flight += 1
            return True

    def settle(self, reserved: float, actual: float) -> None:
        with self._cond:
            self.reserved -= reserved
            self.spent += actual
            self.in_flight -= 1
            self._cond.notify_all()


def execute(
    jobs: list[Job],
    cache: ResultCache,
    provider_for: Callable[[str], Provider],
    concurrency: int = 4,
    on_progress: Callable[[int, int], None] | None = None,
    max_spend: float | None = None,
    worst_case: Callable[[Job], float | None] | None = None,
    actual_cost: Callable[[Job, Completion], float | None] | None = None,
) -> Execution:
    """Run every job, reusing cached answers. Outcomes keep job order.

    A candidate that hits a `disable` error is skipped for the rest of the run, and a `fatal`
    error stops everything. Errors are never cached, so running again retries them.

    With `max_spend`, `worst_case` gives each call's highest possible cost and `actual_cost`
    its real cost once answered; calls that could break the limit are held back, not sent.
    """
    if max_spend is not None and (worst_case is None or actual_cost is None):
        raise ValueError("max_spend needs worst_case and actual_cost")
    guard = SpendGuard(max_spend) if max_spend is not None else None
    outcomes: dict[int, Outcome] = {}
    pending: list[tuple[int, Job, str]] = []
    for i, job in enumerate(jobs):
        key = request_key(job.candidate, job.messages)
        hit = cache.get(key)
        if hit is None:
            pending.append((i, job, key))
        else:
            completion, latency = hit
            outcomes[i] = Outcome("ok", completion, latency, cached=True)

    run = Execution(outcomes=[])
    done, total = len(outcomes), len(jobs)
    if on_progress:
        on_progress(done, total)
    providers = {name: provider_for(name) for name in {j.candidate.provider for _, j, _ in pending}}

    def call(job: Job) -> tuple[Completion, float] | str:
        """Runs in a worker thread. Returns the answer, or the reason it was skipped.

        Workers record disabled candidates and fatal stops themselves, so a request already
        queued behind a failure is skipped instead of failing the same way.
        """
        if run.stopped_reason:
            return run.stopped_reason
        if job.candidate.label in run.disabled:
            return run.disabled[job.candidate.label]
        worst = 0.0
        if guard is not None:
            limit = worst_case(job)  # type: ignore[misc]
            if limit is None:
                return f"no price for {job.candidate.label}, so --max-spend can't be enforced"
            if not guard.reserve(limit):
                return f"held back to stay under --max-spend ${guard.limit:.2f}"
            worst = limit
        start = time.perf_counter()
        try:
            completion = providers[job.candidate.provider].complete(job.candidate, job.messages)
        except ProviderError as e:
            if guard is not None:
                # Rejected requests (bad key, bad model) aren't billed; anything else might be.
                guard.settle(worst, 0.0 if e.fatal or e.disable else worst)
            if e.disable:
                run.disabled.setdefault(job.candidate.label, str(e))
            if e.fatal and run.stopped_reason is None:
                run.stopped_reason = str(e)
            raise
        if guard is not None:
            actual = actual_cost(job, completion)  # type: ignore[misc]
            guard.settle(worst, worst if actual is None else actual)
        return completion, (time.perf_counter() - start) * 1000

    with ThreadPoolExecutor(max_workers=max(1, concurrency)) as pool:
        futures = {pool.submit(call, job): (i, job, key) for i, job, key in pending}
        for future in as_completed(futures):
            i, job, key = futures[future]
            try:
                result = future.result()
            except CancelledError:
                outcomes[i] = Outcome("skipped", error=run.stopped_reason)
            except ProviderError as e:
                if e.fatal:
                    for other in futures:
                        other.cancel()
                outcomes[i] = Outcome("error", error=str(e))
            else:
                if isinstance(result, str):
                    outcomes[i] = Outcome("skipped", error=result)
                    run.held_back += result.startswith("held back")
                else:
                    completion, latency = result
                    cache.put(key, job.candidate, completion, latency)
                    outcomes[i] = Outcome("ok", completion, latency)
            done += 1
            if on_progress:
                on_progress(done, total)

    run.outcomes = [outcomes[i] for i in range(len(jobs))]
    return run
