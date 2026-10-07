"""Grade replayed answers against the original answers from the log."""

import json
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field

from llm_route_audit.analyze import UNLABELLED, estimate_tokens
from llm_route_audit.cache import ResultCache, request_key
from llm_route_audit.candidates import Candidate, Effort
from llm_route_audit.candidates import Provider as ProviderName
from llm_route_audit.costs import PriceTable
from llm_route_audit.grading.checks import Check, CheckResult
from llm_route_audit.grading.judge import (
    NEXT_STEP_INSTRUCTIONS,
    Result,
    combine,
    from_candidate_side,
    judge_messages,
    parse_verdict,
    render_action,
    render_request,
)
from llm_route_audit.grading.steps import compare_steps
from llm_route_audit.providers.base import Provider
from llm_route_audit.records import LogRecord, ToolCall
from llm_route_audit.replay import ReplayResult, candidate_cost, completion_cost, worst_case_cost
from llm_route_audit.runner import Job, execute

ORIGINAL = "original (as logged)"
# Judges think before answering; this is a rough allowance for the estimate only.
JUDGE_OUTPUT_TOKENS = 500


class AgentRule(BaseModel):
    """How agent steps (tool calls) are compared with the original."""

    model_config = ConfigDict(extra="forbid")

    # Arguments left out of the comparison, e.g. free-text notes that never match exactly.
    ignore_arguments: list[str] = Field(default_factory=list)
    # Ask the judge whether a different next step is still a reasonable one.
    judge_alternatives: bool = False


class TaskRule(BaseModel):
    model_config = ConfigDict(extra="forbid")

    checks: list[Check] = Field(default_factory=list)
    judge: bool | None = None  # default: judge only when there are no checks
    agent: AgentRule = Field(default_factory=AgentRule)

    @property
    def uses_judge(self) -> bool:
        return self.judge if self.judge is not None else not self.checks


class JudgeSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model: str = "claude-opus-5-5"
    effort: Effort | None = None
    provider: ProviderName | None = None
    max_tokens: int = Field(default=4000, gt=0)

    def candidate(self) -> Candidate:
        return Candidate(
            model=self.model,
            effort=self.effort,
            provider=self.provider,
            max_tokens=self.max_tokens,
        )


class GradingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    judge: JudgeSettings = Field(default_factory=JudgeSettings)
    default: TaskRule = Field(default_factory=TaskRule)
    tasks: dict[str, TaskRule] = Field(default_factory=dict)

    def rule_for(self, task_type: str | None) -> TaskRule:
        return self.tasks.get(task_type or UNLABELLED, self.default)


def load_config(path: str | Path | None) -> GradingConfig:
    if path is None:
        return GradingConfig()
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    return GradingConfig.model_validate(data)


@dataclass
class Grade:
    record_id: str
    task_type: str | None
    candidate: str  # label, e.g. "claude-sonnet-5-5 @ low", or ORIGINAL
    outcome: str  # pass | fail | ungraded
    checks: list[CheckResult] = field(default_factory=list)
    judge: Result | None = None
    judge_votes: list[Result | None] = field(default_factory=list)
    reason: str | None = None
    step: str | None = None  # agent steps: "tool_call" or "answer" (what the original did)

    def to_dict(self) -> dict[str, Any]:
        return {
            "record_id": self.record_id,
            "task_type": self.task_type,
            "candidate": self.candidate,
            "outcome": self.outcome,
            "checks": [c.to_dict() for c in self.checks],
            "judge": self.judge,
            "judge_votes": self.judge_votes,
            "reason": self.reason,
            "step": self.step,
        }


@dataclass
class GradePlan:
    grades: list[Grade]
    judge: Candidate
    # (grade index, job for candidate-as-A, job for candidate-as-B)
    judge_pairs: list[tuple[int, Job, Job]] = field(default_factory=list)

    def estimate_cost(self, prices: PriceTable, cache: ResultCache) -> tuple[float | None, int]:
        """Estimated judge spend and the number of judge calls still to make."""
        cost: float | None = 0.0
        calls = 0
        for _, *jobs in self.judge_pairs:
            for job in jobs:
                if cache.get(request_key(job.candidate, job.messages)) is not None:
                    continue
                calls += 1
                one = candidate_cost(
                    prices,
                    job.candidate,
                    input_tokens=sum(estimate_tokens(m.content) for m in job.messages),
                    output_tokens=JUDGE_OUTPUT_TOKENS,
                )
                cost = None if one is None or cost is None else cost + one
        return cost, calls


def _label(result: ReplayResult) -> str:
    return f"{result.model} @ {result.effort}" if result.effort else result.model


def step_kind(record: LogRecord) -> str | None:
    """For agent steps, what the original model did: called tools or answered."""
    if not record.is_agent_step:
        return None
    return "tool_call" if record.response_tool_calls else "answer"


def plan_grades(
    records: list[LogRecord], results: list[ReplayResult], config: GradingConfig
) -> GradePlan:
    """Run the exact checks now and line up judge calls for answers that still need one."""
    by_id = {r.id: r for r in records}
    judge = config.judge.candidate()
    plan = GradePlan(grades=[], judge=judge)

    seen: set[str] = set()
    for result in results:
        record = by_id.get(result.record_id)
        if record is None or record.id in seen:
            continue
        seen.add(record.id)
        rule = config.rule_for(record.task_type)
        step = step_kind(record)
        # Answer checks don't apply to a step that called tools.
        checks = (
            []
            if step == "tool_call"
            else [c.run(record.response, record.response) for c in rule.checks]
        )
        failed = any(c.passed is False for c in checks)
        plan.grades.append(
            Grade(
                record.id,
                record.task_type,
                ORIGINAL,
                "fail" if failed else "pass",
                checks,
                step=step,
            )
        )

    for result in results:
        label = _label(result)
        record = by_id.get(result.record_id)
        if record is None:
            plan.grades.append(
                Grade(result.record_id, result.task_type, label, "ungraded", reason="not in log")
            )
            continue
        if result.status in ("error", "skipped") or result.response is None:
            plan.grades.append(
                Grade(record.id, record.task_type, label, "ungraded", reason=result.error)
            )
            continue
        step = step_kind(record)
        if result.status in ("refusal", "truncated"):
            plan.grades.append(
                Grade(record.id, record.task_type, label, "fail", reason=result.status, step=step)
            )
            continue

        rule = config.rule_for(record.task_type)
        calls = [ToolCall.model_validate(c) for c in result.tool_calls or []]
        if step is not None and (calls or record.response_tool_calls):
            plan.grades.append(_grade_step(plan, record, result, calls, rule, label, step))
            continue

        checks = [c.run(result.response, record.response) for c in rule.checks]
        grade = Grade(record.id, record.task_type, label, "pass", checks, step=step)
        if any(c.passed is False for c in checks):
            grade.outcome = "fail"
            grade.reason = "failed checks"
        elif rule.uses_judge:
            grade.outcome = "ungraded"
            grade.reason = "waiting for judge"
            _queue_judge(plan, record, result.response, record.response)
        plan.grades.append(grade)
    return plan


def _queue_judge(
    plan: GradePlan,
    record: LogRecord,
    candidate: str,
    original: str,
    instructions: str | None = None,
) -> None:
    """Line up both judge orders for the grade about to be appended."""
    request = record.conversation()
    extra = {"instructions": instructions} if instructions else {}
    plan.judge_pairs.append(
        (
            len(plan.grades),
            Job(plan.judge, judge_messages(request, candidate, original, record.tools, **extra)),
            Job(plan.judge, judge_messages(request, original, candidate, record.tools, **extra)),
        )
    )


def _grade_step(
    plan: GradePlan,
    record: LogRecord,
    result: ReplayResult,
    calls: list[ToolCall],
    rule: TaskRule,
    label: str,
    step: str,
) -> Grade:
    """An agent step where either side called tools: compare the calls, and optionally let
    the judge decide whether a different step is still reasonable."""
    match = compare_steps(calls, record.response_tool_calls, rule.agent.ignore_arguments)
    grade = Grade(record.id, record.task_type, label, "pass", [match], step=step)
    if match.passed:
        return grade
    grade.outcome, grade.reason = "fail", match.detail
    if rule.agent.judge_alternatives:
        grade.outcome, grade.reason = "ungraded", "waiting for judge"
        _queue_judge(
            plan,
            record,
            render_action(result.response or "", calls),
            render_action(record.response, record.response_tool_calls),
            NEXT_STEP_INSTRUCTIONS,
        )
    return grade


@dataclass
class GradeRun:
    grades: list[Grade]
    judge_spent: float = 0.0
    judged: int = 0
    agreed: int = 0  # both orders gave the same verdict
    stopped_reason: str | None = None
    held_back: int = 0


def run_judges(
    plan: GradePlan,
    cache: ResultCache,
    prices: PriceTable,
    provider_for: Callable[[str], Provider],
    concurrency: int = 4,
    on_progress: Callable[[int, int], None] | None = None,
    max_spend: float | None = None,
) -> GradeRun:
    jobs = [job for _, a, b in plan.judge_pairs for job in (a, b)]
    execution = execute(
        jobs,
        cache,
        provider_for,
        concurrency,
        on_progress,
        max_spend=max_spend,
        worst_case=lambda job: worst_case_cost(prices, job),
        actual_cost=lambda job, completion: completion_cost(prices, job.candidate, completion),
    )
    run = GradeRun(
        grades=plan.grades,
        stopped_reason=execution.stopped_reason,
        held_back=execution.held_back,
    )

    for n, (index, _, _) in enumerate(plan.judge_pairs):
        grade = plan.grades[index]
        votes: list[Result | None] = []
        problems: list[str] = []
        for outcome, candidate_is in zip(
            execution.outcomes[2 * n : 2 * n + 2], ("A", "B"), strict=True
        ):
            if outcome.completion is None or outcome.completion.status != "ok":
                votes.append(None)
                problems.append(outcome.error or f"judge answer {outcome.status}")
                continue
            if not outcome.cached:
                run.judge_spent += completion_cost(prices, plan.judge, outcome.completion) or 0.0
            votes.append(from_candidate_side(parse_verdict(outcome.completion.text), candidate_is))

        grade.judge_votes = votes
        grade.judge = combine(votes[0], votes[1])
        if grade.judge is None:
            grade.outcome = "ungraded"
            grade.reason = problems[0] if problems else "judge gave no usable verdict"
            continue
        run.judged += 1
        run.agreed += votes[0] == votes[1]
        grade.outcome = "fail" if grade.judge == "loss" else "pass"
        grade.reason = f"judge: {grade.judge}"
    return run


JUDGE_PROMPT_OVERHEAD_TOKENS = 300


def _may_be_judged(record: LogRecord, rule: TaskRule) -> bool:
    if rule.agent.judge_alternatives and record.is_agent_step:
        return True  # any step can differ and go to the judge
    return rule.uses_judge and step_kind(record) != "tool_call"


def judge_upper_bound(
    prices: PriceTable, judge: Candidate, config: GradingConfig, records: list[LogRecord]
) -> float | None:
    """The most judging could cost for these requests: every answer reaches the judge (two
    calls each) and answers are about as long as the originals. None if the judge has no price."""
    total = 0.0
    for record in records:
        if not _may_be_judged(record, config.rule_for(record.task_type)):
            continue
        request_tokens = estimate_tokens(render_request(record.conversation(), record.tools))
        answer = render_action(record.response, record.response_tool_calls)
        per_call = candidate_cost(
            prices,
            judge,
            input_tokens=request_tokens
            + 2 * estimate_tokens(answer)
            + JUDGE_PROMPT_OVERHEAD_TOKENS,
            output_tokens=JUDGE_OUTPUT_TOKENS,
        )
        if per_call is None:
            return None
        total += 2 * per_call
    return total


def write_grades(path: str | Path, grades: list[Grade]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for grade in grades:
            f.write(json.dumps(grade.to_dict(), ensure_ascii=False) + "\n")
