import json
from datetime import date

import pytest

from llm_route_audit.cache import ResultCache
from llm_route_audit.costs import ModelPrice, PriceTable
from llm_route_audit.grading.grade import ORIGINAL, GradingConfig, plan_grades, run_judges
from llm_route_audit.providers.base import Completion
from llm_route_audit.records import LogRecord
from llm_route_audit.replay import ReplayResult

PRICES = PriceTable(
    updated=date(2026, 9, 25), models={"claude-judge": ModelPrice(input=1, output=5)}
)

CONFIG = GradingConfig.model_validate(
    {
        "judge": {"model": "claude-judge"},
        "tasks": {
            "classify": {
                "checks": [
                    {"type": "json"},
                    {"type": "match_reference", "fields": ["category"]},
                ]
            },
            "reply": {"checks": [{"type": "contains", "values": ["Thanks"]}], "judge": True},
        },
    }
)


def record(rid: str, task: str, response: str) -> LogRecord:
    return LogRecord.model_validate(
        {
            "id": rid,
            "timestamp": "2026-10-01T00:00:00Z",
            "model": "big",
            "task_type": task,
            "prompt": f"request {rid}",
            "response": response,
        }
    )


def answer(rid: str, task: str, text: str | None, status: str = "ok") -> ReplayResult:
    return ReplayResult(
        record_id=rid,
        task_type=task,
        model="small",
        effort=None,
        provider="anthropic",
        status=status,
        response=text,
    )


RECORDS = [
    record("c1", "classify", json.dumps({"category": "billing"})),
    record("c2", "classify", json.dumps({"category": "technical"})),
    record("r1", "reply", "Thanks for writing in."),
]


class ScriptedJudge:
    """Prefers whichever answer contains 'better'; otherwise calls a tie."""

    def __init__(self) -> None:
        self.calls = 0

    def complete(self, candidate, messages):
        self.calls += 1
        prompt = messages[-1].content
        a = prompt.split("<answer_a>")[1].split("</answer_a>")[0]
        b = prompt.split("<answer_b>")[1].split("</answer_b>")[0]
        verdict = "A" if "better" in a else "B" if "better" in b else "TIE"
        return Completion(
            text=f"Reasoning.\nVERDICT: {verdict}", input_tokens=200, output_tokens=20
        )


def by(grades, candidate):
    return {g.record_id: g for g in grades if g.candidate == candidate}


def test_checks_settle_answers_without_the_judge():
    results = [
        answer("c1", "classify", json.dumps({"category": "Billing"})),
        answer("c2", "classify", json.dumps({"category": "account"})),
    ]
    plan = plan_grades(RECORDS, results, CONFIG)
    small = by(plan.grades, "small")

    assert small["c1"].outcome == "pass"
    assert small["c2"].outcome == "fail"
    assert plan.judge_pairs == []
    assert {g.outcome for g in by(plan.grades, ORIGINAL).values()} == {"pass"}


def test_failed_checks_skip_the_judge_and_passing_ones_use_it():
    results = [
        answer("r1", "reply", "No greeting here."),
    ]
    plan = plan_grades(RECORDS, results, CONFIG)
    assert by(plan.grades, "small")["r1"].outcome == "fail"
    assert plan.judge_pairs == []

    plan = plan_grades(RECORDS, [answer("r1", "reply", "Thanks! A better reply.")], CONFIG)
    assert len(plan.judge_pairs) == 1


def test_replay_failures_are_graded_without_calls():
    results = [
        answer("c1", "classify", None, status="error"),
        answer("c2", "classify", "partial", status="truncated"),
    ]
    small = by(plan_grades(RECORDS, results, CONFIG).grades, "small")
    assert small["c1"].outcome == "ungraded"
    assert small["c2"].outcome == "fail" and small["c2"].reason == "truncated"


@pytest.fixture
def cache():
    c = ResultCache(":memory:")
    yield c
    c.close()


def test_judge_runs_both_orders_and_combines(cache):
    plan = plan_grades(RECORDS, [answer("r1", "reply", "Thanks! A better reply.")], CONFIG)
    judge = ScriptedJudge()
    run = run_judges(plan, cache, PRICES, lambda _: judge, concurrency=1)

    grade = by(run.grades, "small")["r1"]
    assert judge.calls == 2
    assert grade.judge_votes == ["win", "win"]
    assert (grade.judge, grade.outcome) == ("win", "pass")
    assert (run.judged, run.agreed) == (1, 1)
    assert run.judge_spent == pytest.approx(2 * (200 * 1 + 20 * 5) / 1_000_000)


def test_judge_loss_fails_the_answer_and_cache_avoids_repeat_calls(cache):
    # The original is the better answer here.
    plan_records = [record("r1", "reply", "Thanks, and here is a better reply.")]
    plan = plan_grades(plan_records, [answer("r1", "reply", "Thanks.")], CONFIG)
    first = ScriptedJudge()
    grade = by(run_judges(plan, cache, PRICES, lambda _: first).grades, "small")["r1"]
    assert grade.outcome == "fail" and grade.judge == "loss"

    again = ScriptedJudge()
    run_judges(
        plan_grades(plan_records, [answer("r1", "reply", "Thanks.")], CONFIG),
        cache,
        PRICES,
        lambda _: again,
    )
    assert again.calls == 0


def test_unusable_verdict_leaves_the_answer_ungraded(cache):
    class Mumbler:
        def complete(self, candidate, messages):
            return Completion(text="Hard to say.", input_tokens=10, output_tokens=5)

    plan = plan_grades(RECORDS, [answer("r1", "reply", "Thanks!")], CONFIG)
    grade = by(run_judges(plan, cache, PRICES, lambda _: Mumbler()).grades, "small")["r1"]
    assert grade.outcome == "ungraded"


def test_estimate_counts_only_uncached_judge_calls(cache):
    plan = plan_grades(RECORDS, [answer("r1", "reply", "Thanks!")], CONFIG)
    cost, calls = plan.estimate_cost(PRICES, cache)
    assert calls == 2 and cost > 0
    run_judges(plan, cache, PRICES, lambda _: ScriptedJudge())
    assert plan.estimate_cost(PRICES, cache) == (0.0, 0)


def test_tasks_without_rules_go_to_the_judge_by_default():
    plan = plan_grades(
        [record("x1", "other", "hello")], [answer("x1", "other", "hi")], GradingConfig()
    )
    assert len(plan.judge_pairs) == 1
    assert plan.judge.model == "claude-opus-5-5"
