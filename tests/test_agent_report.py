"""The report for agent logs: routing per session type, step agreement, session cost."""

from datetime import date

import pytest

from llm_route_audit.costs import ModelPrice, PriceTable
from llm_route_audit.grading.grade import ORIGINAL
from llm_route_audit.records import LogRecord, ToolCall
from llm_route_audit.replay import ReplayResult
from llm_route_audit.report import build_report, by_session_type, routed_cost
from llm_route_audit.report_view import render_html, render_text

PRICES = PriceTable(
    updated=date(2026, 9, 25),
    models={
        "big": ModelPrice(input=10, output=50, cache_read=1, cache_write=12.5),
        "small": ModelPrice(input=1, output=5, cache_read=0.1, cache_write=1.25),
    },
)


def step(session: str, n: int, task: str = "refunds", final: bool = False) -> LogRecord:
    return LogRecord(
        id=f"{session}-{n}",
        timestamp=f"2026-10-01T00:00:0{n}Z",
        model="big",
        task_type=task,
        session_id=session,
        prompt="history",
        response="Done." if final else "",
        response_tool_calls=None if final else [ToolCall(name="lookup", arguments={"n": n})],
        input_tokens=100,
        cache_read_tokens=900,
        output_tokens=20,
    )


def replayed(record: LogRecord) -> ReplayResult:
    return ReplayResult(
        record_id=record.id,
        task_type=record.task_type,
        model="small",
        effort=None,
        provider="anthropic",
        status="ok",
        response=record.response,
        input_tokens=1000,  # a replay never hits the cache
        output_tokens=20,
        cost=1000 * 1 / 1e6 + 20 * 5 / 1e6,
    )


def grade(record: LogRecord, candidate: str, outcome: str) -> dict:
    step_kind = "tool_call" if record.response_tool_calls else "answer"
    return {
        "record_id": record.id,
        "candidate": candidate,
        "outcome": outcome,
        "step": step_kind,
        "judge_votes": [],
    }


def test_cost_assumes_the_candidate_caches_like_the_original():
    record = step("a", 1)
    result = replayed(record)
    cost = routed_cost(PRICES, record, result)
    # 10% uncached, 90% cache reads, as in the original
    assert cost == pytest.approx((100 * 1 + 900 * 0.1 + 20 * 5) / 1e6)
    assert cost < result.cost
    uncached = record.model_copy(update={"cache_read_tokens": None})
    assert routed_cost(PRICES, uncached, result) == result.cost


def test_steps_take_their_session_type():
    records = [step("a", 1, "plan"), step("a", 2, "act"), step("a", 3, "act"), step("b", 1, "plan")]
    assert [r.task_type for r in by_session_type(records)] == ["act", "act", "act", "plan"]
    plain = [r.model_copy(update={"session_id": None}) for r in records]
    assert by_session_type(plain) == plain


def test_session_section_counts_agreement_splits_and_cost():
    sessions = {s: [step(s, 1), step(s, 2), step(s, 3, final=True)] for s in ("a", "b", "c")}
    records = [r for steps in sessions.values() for r in steps]
    results = [replayed(r) for r in records]
    outcomes = {
        "a": ["pass", "pass", "pass"],
        "b": ["pass", "fail", "pass"],
        "c": ["fail", "pass", "fail"],
    }
    grades = [grade(r, ORIGINAL, "pass") for r in records]
    grades += [
        grade(r, "small", outcome)
        for s, steps in sessions.items()
        for r, outcome in zip(steps, outcomes[s], strict=True)
    ]
    report = build_report(records, results, grades, PRICES, min_samples=1)

    [kind] = report.sessions
    assert (kind.task, kind.sessions, kind.steps) == ("refunds", 3, 9)
    [small] = kind.options
    assert (small.tool_same, small.tool_steps) == (4, 6)
    assert (small.answers_passed, small.answers) == (2, 3)
    assert (small.sessions, small.all_same) == (3, 1)
    assert small.first_splits == [2, 1] and small.typical_split == 1.5
    original_session = 3 * (100 * 10 + 900 * 1 + 20 * 50) / 1e6
    assert kind.original_cost == pytest.approx(original_session)
    assert small.cost_per_session == pytest.approx(3 * (100 * 1 + 900 * 0.1 + 20 * 5) / 1e6)
    assert report.tasks[0].options[0].cost_ratio == pytest.approx(
        small.cost_per_session / original_session
    )

    text = render_text(report)
    assert "Agent sessions, step by step" in text
    assert "4/6 (67%)" in text and "1/3 (33%)" in text and "step 1.5" in text
    assert "Agent sessions, step by step" in render_html(report, "agent.jsonl")


def test_plain_logs_have_no_session_section():
    records = [step("a", 1).model_copy(update={"session_id": None})]
    report = build_report(records, [replayed(records[0])], [], PRICES)
    assert report.sessions == []
    assert "Agent sessions" not in render_text(report)
