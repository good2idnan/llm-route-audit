from datetime import date

import pytest

from llm_route_audit.analyze import (
    UNLABELLED,
    build_profile,
    estimate_tokens,
    percentile,
)
from llm_route_audit.costs import ModelPrice, PriceTable
from llm_route_audit.records import LogRecord

PRICES = PriceTable(
    updated=date(2026, 9, 25),
    models={
        "big": ModelPrice(input=10, output=50),
        "small": ModelPrice(input=1, output=5, cache_read=0.1),
    },
)


def rec(i, model="big", task="chat", day=1, **usage):
    fields = {"input_tokens": 1000, "output_tokens": 100, **usage}
    return LogRecord.model_validate(
        {
            "id": f"r{i}",
            "timestamp": f"2026-10-{day:02d}T12:00:00Z",
            "model": model,
            "task_type": task,
            "prompt": "x" * 40,
            "response": "y" * 20,
            **{k: v for k, v in fields.items() if v is not None},
        }
    )


def test_totals_and_groups_add_up():
    records = [rec(1), rec(2, task="extract"), rec(3, model="small", task="extract")]
    profile = build_profile(records, PRICES)

    # big: 1000*10 + 100*50 = 15,000 per million -> $0.015; small: 1000*1 + 100*5 -> $0.0015
    assert profile.total.cost == pytest.approx(0.015 * 2 + 0.0015)
    assert sum(g.cost for g in profile.by_task) == pytest.approx(profile.total.cost)
    assert [g.name for g in profile.by_model] == ["big", "small"]
    assert {g.name: g.requests for g in profile.by_task} == {"chat": 1, "extract": 2}


def test_groups_are_sorted_most_expensive_first():
    records = [rec(1, model="small", task="cheap"), rec(2, task="pricey")]
    assert [g.name for g in build_profile(records, PRICES).by_task] == ["pricey", "cheap"]


def test_cache_reads_are_priced_separately():
    profile = build_profile([rec(1, model="small", cache_read_tokens=1_000_000)], PRICES)
    assert profile.total.cost == pytest.approx(0.0015 + 0.1)


def test_missing_token_counts_are_estimated_and_flagged():
    record = rec(1, input_tokens=None, output_tokens=None)
    profile = build_profile([record], PRICES)
    assert profile.estimated_records == 1
    assert profile.total.input_tokens == estimate_tokens("x" * 40) == 10
    assert profile.total.output_tokens == estimate_tokens("y" * 20) == 5


def test_unpriced_models_are_reported_not_fatal():
    profile = build_profile([rec(1), rec(2, model="mystery")], PRICES)
    assert profile.unpriced_models == {"mystery": 1}
    assert profile.total.requests == 2
    assert profile.total.priced_requests == 1
    assert profile.total.cost == pytest.approx(0.015)


def test_missing_task_type_is_grouped_as_unlabelled():
    profile = build_profile([rec(1, task=None)], PRICES)
    assert profile.by_task[0].name == UNLABELLED
    assert profile.unlabelled_records == 1


def test_monthly_estimate_needs_at_least_a_day_of_logs():
    same_day = build_profile([rec(1), rec(2)], PRICES)
    assert same_day.monthly_cost is None

    ten_days = build_profile([rec(1, day=1), rec(2, day=11)], PRICES)
    assert ten_days.span_days == pytest.approx(10)
    assert ten_days.monthly_cost == pytest.approx(0.03 / 10 * 30)


def test_percentile_nearest_rank():
    assert percentile([], 50) is None
    assert percentile([5, 1, 3], 50) == 3
    assert percentile(list(range(1, 101)), 95) == 95


def test_empty_input_is_rejected():
    with pytest.raises(ValueError):
        build_profile([], PRICES)
