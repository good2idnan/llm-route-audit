"""Grading agent steps: tool calls compared with the original, judge for alternatives."""

from datetime import date

from llm_route_audit.cache import ResultCache
from llm_route_audit.costs import ModelPrice, PriceTable
from llm_route_audit.grading.grade import (
    ORIGINAL,
    GradingConfig,
    judge_upper_bound,
    plan_grades,
    run_judges,
)
from llm_route_audit.grading.judge import render_request
from llm_route_audit.grading.steps import compare_steps
from llm_route_audit.providers.base import Completion
from llm_route_audit.records import LogRecord, Message, ToolCall, ToolDef
from llm_route_audit.replay import ReplayResult

PRICES = PriceTable(
    updated=date(2026, 9, 25), models={"claude-judge": ModelPrice(input=1, output=5)}
)


def call(name: str, **arguments) -> ToolCall:
    return ToolCall(name=name, arguments=arguments)


# --- comparing one step ---------------------------------------------------------------------


def test_same_calls_in_any_order_match():
    original = [call("check_status", domain="a.io"), call("recent_deploys", domain="a.io")]
    swapped = [call("recent_deploys", domain="A.io "), call("check_status", domain="a.io")]
    result = compare_steps(swapped, original)
    assert result.passed and result.detail == "same calls: check_status, recent_deploys"


def test_differences_are_named():
    refund = [call("issue_refund", invoice_id="INV-1", amount=49, reason="charged twice")]
    assert compare_steps(
        [call("issue_refund", invoice_id="INV-1", amount=490, reason="x")], refund
    ).detail == ("issue_refund: different amount, reason")
    assert compare_steps([call("escalate_to_billing", invoice_id="INV-1")], refund).detail == (
        "called escalate_to_billing instead of issue_refund"
    )
    assert compare_steps([], refund).detail == "replied in text instead of calling issue_refund"
    assert compare_steps(refund, None).detail == "called issue_refund instead of replying"
    assert compare_steps(refund + [call("ping")], refund).detail == "also called ping"
    assert (
        compare_steps([call("ping")], refund + [call("ping")]).detail == "did not call issue_refund"
    )


def test_ignored_arguments_are_left_out():
    refund = [call("issue_refund", invoice_id="INV-1", amount=49, reason="charged twice")]
    reworded = [call("issue_refund", invoice_id="INV-1", amount="49.00", reason="duplicate charge")]
    assert not compare_steps(reworded, refund).passed
    assert compare_steps(reworded, refund, ignore=["reason"]).passed


# --- grading agent sessions -------------------------------------------------------------------

TOOLS = [ToolDef(name="find_order", description="Look up an order."), ToolDef(name="refund")]
HISTORY = [
    Message(role="user", content="Refund order 7."),
    Message(
        role="assistant", tool_calls=[ToolCall(id="c1", name="find_order", arguments={"id": 7})]
    ),
    Message(role="tool", tool_call_id="c1", name="find_order", content='{"paid": 40}'),
]


def step(rid: str, response: str = "", calls: list[ToolCall] | None = None) -> LogRecord:
    return LogRecord(
        id=rid,
        timestamp="2026-10-01T00:00:00Z",
        model="big",
        task_type="refunds",
        session_id="s1",
        messages=HISTORY,
        tools=TOOLS,
        response=response,
        response_tool_calls=calls,
    )


def replayed(rid: str, text: str = "", calls: list[ToolCall] | None = None) -> ReplayResult:
    return ReplayResult(
        record_id=rid,
        task_type="refunds",
        model="small",
        effort=None,
        provider="anthropic",
        status="ok",
        response=text,
        tool_calls=[c.model_dump() for c in calls] if calls else None,
    )


RECORDS = [
    step("tool", calls=[call("refund", order=7, amount=40)]),
    step("answer", response="Done: refunded $40."),
]
CONFIG = GradingConfig.model_validate({"judge": {"model": "claude-judge"}})


def by_key(plan):
    return {(g.record_id, g.candidate): g for g in plan.grades}


def test_tool_steps_are_graded_by_their_calls():
    plan = plan_grades(
        RECORDS,
        [
            replayed("tool", calls=[call("refund", order=7, amount=40)]),
            replayed("answer", calls=[call("find_order", id=7)]),
        ],
        CONFIG,
    )
    grades = by_key(plan)
    assert grades[("tool", ORIGINAL)].outcome == "pass"
    assert grades[("tool", ORIGINAL)].step == "tool_call"
    assert grades[("answer", ORIGINAL)].step == "answer"
    assert (grades[("tool", "small")].outcome, grades[("tool", "small")].step) == (
        "pass",
        "tool_call",
    )
    assert grades[("answer", "small")].outcome == "fail"
    assert grades[("answer", "small")].reason == "called find_order instead of replying"
    assert plan.judge_pairs == []  # exact comparison settles tool steps


def test_text_answers_still_go_to_the_judge_with_the_tool_history():
    plan = plan_grades(RECORDS, [replayed("answer", "Refunded $40.")], CONFIG)
    assert by_key(plan)[("answer", "small")].reason == "waiting for judge"
    [(_, first, _)] = plan.judge_pairs
    prompt = first.messages[1].content
    assert "CALL find_order" in prompt and '<tool_result name="find_order">' in prompt
    assert "- find_order: Look up an order." in prompt


class AlwaysTie:
    def complete(self, candidate, messages, **kwargs):
        return Completion("Both fine.\nVERDICT: TIE", 100, 10)


def test_judge_can_accept_a_different_but_reasonable_step():
    config = GradingConfig.model_validate(
        {
            "judge": {"model": "claude-judge"},
            "tasks": {"refunds": {"agent": {"judge_alternatives": True}}},
        }
    )
    plan = plan_grades(
        RECORDS, [replayed("tool", "Checking first.", [call("find_order", id=7)])], config
    )
    [(_, first, second)] = plan.judge_pairs
    assert "chose its next step" in first.messages[1].content
    assert 'Checking first.\nCALL find_order({"id": 7})' in first.messages[1].content
    run = run_judges(plan, ResultCache(":memory:"), PRICES, lambda name: AlwaysTie())
    grade = by_key(plan)[("tool", "small")]
    assert (grade.outcome, grade.reason, grade.judge) == ("pass", "judge: tie", "tie")
    assert run.judged == 1


def test_judge_estimate_follows_the_agent_rules():
    assert judge_upper_bound(PRICES, CONFIG.judge.candidate(), CONFIG, RECORDS[:1]) == 0
    assert judge_upper_bound(PRICES, CONFIG.judge.candidate(), CONFIG, RECORDS[1:]) > 0


def test_request_rendering_without_tools_is_unchanged():
    plain = [Message(role="system", content="Be brief."), Message(role="user", content="Hi")]
    assert render_request(plain) == (
        "<request>\n<system>\nBe brief.\n</system>\n<user>\nHi\n</user>\n</request>"
    )
