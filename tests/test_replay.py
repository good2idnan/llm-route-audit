from datetime import date

import pytest

from routeaudit.cache import ResultCache, request_key
from routeaudit.candidates import Candidate
from routeaudit.costs import ModelPrice, PriceTable
from routeaudit.providers.base import Completion, ProviderError
from routeaudit.records import LogRecord, Message
from routeaudit.replay import estimate, logged_cost, run_replay

PRICES = PriceTable(
    updated=date(2026, 9, 25),
    models={
        "claude-big": ModelPrice(input=10, output=50),
        "claude-small": ModelPrice(input=1, output=5),
    },
)
SMALL = Candidate(model="claude-small", provider="anthropic")
BIG = Candidate(model="claude-big", provider="anthropic", effort="low")


def rec(i: int) -> LogRecord:
    return LogRecord.model_validate(
        {
            "id": f"r{i}",
            "timestamp": "2026-10-01T00:00:00Z",
            "model": "claude-big",
            "task_type": "chat",
            "prompt": f"question {i}",
            "response": "answer",
            "input_tokens": 1000,
            "output_tokens": 100,
        }
    )


class FakeProvider:
    def __init__(self, fail_for: dict[str, ProviderError] | None = None):
        self.calls: list[str] = []
        self.fail_for = fail_for or {}

    def complete(self, candidate: Candidate, messages: list[Message]) -> Completion:
        self.calls.append(candidate.label)
        if candidate.model in self.fail_for:
            raise self.fail_for[candidate.model]
        return Completion(text=f"{candidate.model} says hi", input_tokens=1000, output_tokens=100)


@pytest.fixture
def cache():
    c = ResultCache(":memory:")
    yield c
    c.close()


def test_replay_runs_every_pair_and_prices_answers(cache):
    provider = FakeProvider()
    run = run_replay([rec(1), rec(2)], [SMALL, BIG], cache, PRICES, lambda _: provider)

    assert len(run.results) == 4
    assert [r.model for r in run.results] == ["claude-small"] * 2 + ["claude-big"] * 2
    assert all(r.status == "ok" for r in run.results)
    assert run.results[0].cost == pytest.approx(0.0015)
    assert run.spent == pytest.approx(2 * 0.0015 + 2 * 0.015)


def test_second_run_is_served_from_cache(cache):
    first = FakeProvider()
    run_replay([rec(1)], [SMALL], cache, PRICES, lambda _: first)
    second = FakeProvider()
    run = run_replay([rec(1)], [SMALL], cache, PRICES, lambda _: second)

    assert second.calls == []
    assert run.results[0].cached and run.results[0].response == "claude-small says hi"
    assert run.spent == 0


def test_cache_key_depends_on_effort_and_prompt():
    messages = [Message(role="user", content="hi")]
    assert request_key(SMALL, messages) != request_key(
        SMALL.model_copy(update={"effort": "low"}), messages
    )
    assert request_key(SMALL, messages) != request_key(
        SMALL, [Message(role="user", content="hello")]
    )


def test_a_bad_candidate_is_disabled_without_stopping_others(cache):
    provider = FakeProvider({"claude-big": ProviderError("unknown model", disable=True)})
    run = run_replay(
        [rec(i) for i in range(5)], [SMALL, BIG], cache, PRICES, lambda _: provider, concurrency=1
    )
    big = [r for r in run.results if r.model == "claude-big"]
    assert all(r.status == "ok" for r in run.results if r.model == "claude-small")
    assert {r.status for r in big} <= {"error", "skipped"}
    assert "skipped" in {r.status for r in big}
    assert run.stopped_reason is None
    assert "claude-big @ low" in run.disabled


def test_errors_are_not_cached_so_they_retry(cache):
    failing = FakeProvider({"claude-small": ProviderError("timeout")})
    run_replay([rec(1)], [SMALL], cache, PRICES, lambda _: failing)
    recovered = FakeProvider()
    run = run_replay([rec(1)], [SMALL], cache, PRICES, lambda _: recovered)
    assert recovered.calls == ["claude-small"]
    assert run.results[0].status == "ok"


def test_fatal_error_stops_the_run(cache):
    provider = FakeProvider({"claude-small": ProviderError("bad key", fatal=True)})
    run = run_replay(
        [rec(i) for i in range(10)], [SMALL], cache, PRICES, lambda _: provider, concurrency=1
    )
    assert run.stopped_reason == "bad key"
    assert len(provider.calls) < 10
    assert len(run.results) == 10


def test_estimate_skips_cached_answers(cache):
    run_replay([rec(1)], [SMALL], cache, PRICES, lambda _: FakeProvider())
    [est] = estimate([rec(1), rec(2)], [SMALL], PRICES, cache)
    assert (est.requests, est.cached) == (2, 1)
    assert est.cost == pytest.approx(0.0015)


def test_unpriced_cloud_model_has_unknown_cost_but_local_is_free(cache):
    cloud = Candidate(model="claude-mystery", provider="anthropic")
    local = Candidate(model="ollama/llama3.2")
    by_label = {e.candidate.label: e for e in estimate([rec(1)], [cloud, local], PRICES, cache)}
    assert by_label["claude-mystery"].cost is None
    assert by_label["ollama/llama3.2"].cost == 0


def test_logged_cost_of_the_sample():
    assert logged_cost(PRICES, [rec(1), rec(2)]) == pytest.approx(2 * 0.015)
