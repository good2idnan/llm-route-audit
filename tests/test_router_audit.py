"""Auditing a router: its picks are recorded, graded and compared with simpler strategies."""

from datetime import date

import pytest

from llm_route_audit.cache import ResultCache
from llm_route_audit.candidates import Candidate
from llm_route_audit.costs import ModelPrice, PriceTable
from llm_route_audit.grading.grade import ORIGINAL
from llm_route_audit.providers import openrouter
from llm_route_audit.providers.base import Completion
from llm_route_audit.records import LogRecord
from llm_route_audit.replay import ReplayResult, candidate_cost, completion_cost
from llm_route_audit.report import build_report
from llm_route_audit.report_view import render_html, render_text

PRICES = PriceTable(
    updated=date(2026, 10, 1),
    models={
        "big": ModelPrice(input=10, output=50),
        "cheap": ModelPrice(input=1, output=5),
    },
)
ROUTER = Candidate(model="openrouter/openrouter/auto", router=True, price_as="big")


def test_router_estimates_use_price_as_but_real_costs_use_the_pick():
    assert candidate_cost(PRICES, ROUTER, input_tokens=1000, output_tokens=0) == pytest.approx(0.01)
    picked_cheap = Completion("ok", 1000, 0, served_model="cheap")
    assert completion_cost(PRICES, ROUTER, picked_cheap) == pytest.approx(0.001)
    reported = Completion("ok", 1000, 0, cost=0.0042, served_model="cheap")
    assert completion_cost(PRICES, ROUTER, reported) == 0.0042  # the provider's own figure
    assert completion_cost(PRICES, ROUTER, Completion("ok", 1000, 0)) is None  # unknown pick


def test_the_picked_model_is_read_and_cached(tmp_path):
    completion = openrouter.parse_response(
        {
            "model": "anthropic/claude-haiku-4.5",
            "choices": [{"message": {"content": "hi"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 2, "cost": 0.0001},
        }
    )
    assert completion.served_model == "anthropic/claude-haiku-4.5"
    cache = ResultCache(tmp_path / "cache.sqlite")
    cache.put("k", ROUTER, completion, 5.0)
    assert cache.get("k")[0].served_model == "anthropic/claude-haiku-4.5"


def record(rid: str, task: str) -> LogRecord:
    return LogRecord(
        id=rid,
        timestamp="2026-10-01T00:00:00Z",
        model="big",
        task_type=task,
        prompt="x" * 400,
        response="answer",
        input_tokens=100,
        output_tokens=100,
    )


def scenario():
    """Ten easy and ten hard requests. The cheap model handles only easy ones. The router
    sends easy ones to the big model (wasteful) and half the hard ones to the cheap model
    (wrong), so a per-task policy beats it."""
    records = [record(f"e{i}", "easy") for i in range(10)]
    records += [record(f"h{i}", "hard") for i in range(10)]
    results, grades = [], []
    logged = 100 * 10 / 1e6 + 100 * 50 / 1e6  # what each request cost on "big"
    for r in records:
        grades.append({"record_id": r.id, "candidate": ORIGINAL, "outcome": "pass"})
        results.append(
            ReplayResult(r.id, r.task_type, "cheap", None, "anthropic", "ok", "a", cost=logged / 10)
        )
        grades.append(
            {
                "record_id": r.id,
                "candidate": "cheap",
                "outcome": "pass" if r.task_type == "easy" else "fail",
            }
        )
        cheap_pick = r.task_type == "hard" and r.id < "h5"
        results.append(
            ReplayResult(
                r.id,
                r.task_type,
                ROUTER.model,
                None,
                "openrouter",
                "ok",
                "a",
                cost=logged / 10 if cheap_pick else logged,
                served_model="cheap" if cheap_pick else "big",
                router=True,
            )
        )
        grades.append(
            {
                "record_id": r.id,
                "candidate": ROUTER.model,
                "outcome": "fail" if cheap_pick else "pass",
            }
        )
    return records, results, grades


def test_report_audits_the_router():
    report = build_report(*scenario(), PRICES)
    assert f"Router {ROUTER.model}" in [s.name for s in report.strategies]
    [audit] = report.routers
    assert audit.result.quality == pytest.approx(0.75)
    assert audit.result.cost_ratio == pytest.approx(0.775)
    assert [s.name for s in audit.beaten_by] == ["Per-task policy"]
    assert audit.verdict.startswith("Beaten by Per-task policy: 100% quality at 55%")
    picks = {(p.task, p.model): (p.answers, p.passed) for p in audit.picks}
    assert picks == {("easy", "big"): (10, 10), ("hard", "big"): (5, 5), ("hard", "cheap"): (5, 0)}

    text = render_text(report)
    assert "Router audit" in text and "Model it picked" in text
    assert "a little optimistic" in text  # the policy was chosen on the same answers
    assert "Router audit: openrouter/openrouter/auto" in render_html(report, "logs.jsonl")
    assert report.to_dict()["routers"][0]["beaten_by"] == ["Per-task policy"]


def test_unbeaten_router_says_so():
    records, results, grades = scenario()
    for g in grades:  # make the router perfect and every other option fail
        g["outcome"] = "pass" if g["candidate"] in (ORIGINAL, ROUTER.model) else "fail"
    report = build_report(records, results, grades, PRICES)
    assert report.routers[0].beaten_by == []
    assert report.routers[0].verdict.startswith("No other strategy")


def test_reports_without_routers_have_no_router_section():
    records, results, grades = scenario()
    plain = [r for r in results if not r.router]
    report = build_report(records, plain, grades, PRICES)
    assert report.routers == [] and "Router audit" not in render_text(report)
