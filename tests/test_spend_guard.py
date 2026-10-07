"""The --max-spend guard: total spend must never pass the limit, even with parallel calls."""

import sqlite3
import threading
import time
from datetime import date

import pytest

from routeaudit.cache import ResultCache
from routeaudit.candidates import Candidate
from routeaudit.costs import ModelPrice, PriceTable
from routeaudit.providers.base import Completion, ProviderError
from routeaudit.records import LogRecord
from routeaudit.replay import run_replay, worst_case_cost
from routeaudit.runner import Job, SpendGuard

PRICES = PriceTable(updated=date(2026, 10, 1), models={"claude-x": ModelPrice(input=1, output=10)})
CANDIDATE = Candidate(model="claude-x", provider="anthropic", max_tokens=1000)
# worst case per call: ~ (small input) + 1000 output tokens at $10/M = just over $0.01


def rec(i: int) -> LogRecord:
    return LogRecord.model_validate(
        {
            "id": f"r{i:02d}",
            "timestamp": "2026-10-01T00:00:00Z",
            "model": "claude-x",
            "prompt": "hello",
            "response": "hi",
        }
    )


class CostlyProvider:
    """Answers with a known real cost, slowly, so calls overlap."""

    def __init__(self, cost: float) -> None:
        self.cost = cost
        self.calls = 0
        self._lock = threading.Lock()

    def complete(self, candidate, messages):
        with self._lock:
            self.calls += 1
        time.sleep(0.01)
        return Completion(text="ok", input_tokens=10, output_tokens=100, cost=self.cost)


def test_guard_waits_for_calls_in_flight_then_refuses_what_cannot_fit():
    guard = SpendGuard(limit=1.0)
    assert guard.reserve(0.6)
    released = []

    def settle_later():
        time.sleep(0.05)
        released.append(True)
        guard.settle(0.6, 0.1)

    threading.Thread(target=settle_later).start()
    assert guard.reserve(0.6)  # had to wait for the first call to settle
    assert released == [True]
    guard.settle(0.6, 0.6)
    assert guard.spent == pytest.approx(0.7)
    assert not guard.reserve(0.5)  # nothing in flight, and 0.7 + 0.5 > 1.0


def test_worst_case_covers_full_output_budget():
    worst = worst_case_cost(PRICES, Job(CANDIDATE, rec(1).conversation()))
    assert worst > 1000 * 10 / 1_000_000


@pytest.mark.parametrize("concurrency", [1, 4])
def test_replay_never_spends_more_than_the_limit(concurrency):
    cache = ResultCache(":memory:")
    provider = CostlyProvider(cost=0.009)  # each call really costs $0.009
    run = run_replay(
        [rec(i) for i in range(20)],
        [CANDIDATE],
        cache,
        PRICES,
        lambda _: provider,
        concurrency=concurrency,
        max_spend=0.05,
    )
    assert run.spent <= 0.05
    assert provider.calls == sum(r.status == "ok" for r in run.results)
    assert run.held_back == sum(r.status == "skipped" for r in run.results) > 0
    assert provider.calls >= 3  # it still uses the budget it has


def test_errors_count_as_worst_case_but_rejections_do_not():
    cache = ResultCache(":memory:")

    class Flaky:
        def complete(self, candidate, messages):
            raise ProviderError("server error")

    run = run_replay(
        [rec(i) for i in range(10)], [CANDIDATE], cache, PRICES, lambda _: Flaky(), max_spend=0.05
    )
    statuses = [r.status for r in run.results]
    # each failed call might have been billed, so the guard assumes the worst and stops early
    assert statuses.count("error") < 10 and statuses.count("skipped") > 0


def test_unpriced_model_is_not_sent_under_max_spend():
    cache = ResultCache(":memory:")
    unknown = Candidate(model="claude-unknown", provider="anthropic")
    provider = CostlyProvider(cost=0.001)
    run = run_replay([rec(1)], [unknown], cache, PRICES, lambda _: provider, max_spend=1.0)
    assert provider.calls == 0
    assert "can't be enforced" in run.results[0].error


def test_old_cache_files_gain_the_cost_column(tmp_path):
    path = tmp_path / "old.sqlite"
    db = sqlite3.connect(path)
    db.execute(
        "CREATE TABLE completions (key TEXT PRIMARY KEY, model TEXT NOT NULL, effort TEXT, "
        "text TEXT NOT NULL, input_tokens INTEGER NOT NULL, output_tokens INTEGER NOT NULL, "
        "cache_read_tokens INTEGER NOT NULL, cache_write_tokens INTEGER NOT NULL, "
        "status TEXT NOT NULL, latency_ms REAL, created_at TEXT NOT NULL)"
    )
    db.execute("INSERT INTO completions VALUES ('k','m',NULL,'hi',1,1,0,0,'ok',5.0,'2026-10-01')")
    db.commit()
    db.close()

    cache = ResultCache(path)
    completion, latency = cache.get("k")
    assert completion.text == "hi" and completion.cost is None
    cache.put("k2", CANDIDATE, Completion("new", 1, 1, cost=0.5), None)
    assert cache.get("k2")[0].cost == 0.5
    cache.close()
