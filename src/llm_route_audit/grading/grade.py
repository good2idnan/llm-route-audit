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
    Result,
    combine,
    from_candidate_side,
    judge_messages,
    parse_verdict,
)
from llm_route_audit.providers.base import Provider
from llm_route_audit.records import LogRecord
from llm_route_audit.replay import ReplayResult, candidate_cost, completion_cost, worst_case_cost
from llm_route_audit.runner import Job, execute

ORIGINAL = "original (as logged)"
# Judges think before answering; this is a rough allowance for the estimate only.
JUDGE_OUTPUT_TOKENS = 500


class TaskRule(BaseModel):
    model_config = ConfigDict(extra="forbid")

    checks: list[Check] = Field(default_factory=list)
    judge: bool | None = None  # default: judge only when there are no checks

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
        checks = [c.run(record.response, record.response) for c in rule.checks]
        failed = any(c.passed is False for c in checks)
        plan.grades.append(
            Grade(record.id, record.task_type, ORIGINAL, "fail" if failed else "pass", checks)
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
        if result.status in ("refusal", "truncated"):
            plan.grades.append(
                Grade(record.id, record.task_type, label, "fail", reason=result.status)
            )
            continue

        rule = config.rule_for(record.task_type)
        checks = [c.run(result.response, record.response) for c in rule.checks]
        grade = Grade(record.id, record.task_type, label, "pass", checks)
        if any(c.passed is False for c in checks):
            grade.outcome = "fail"
            grade.reason = "failed checks"
        elif rule.uses_judge:
            grade.outcome = "ungraded"
            grade.reason = "waiting for judge"
            request = record.conversation()
            plan.judge_pairs.append(
                (
                    len(plan.grades),
                    Job(judge, judge_messages(request, result.response, record.response)),
                    Job(judge, judge_messages(request, record.response, result.response)),
                )
            )
        plan.grades.append(grade)
    return plan


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


def judge_upper_bound(
    prices: PriceTable, judge: Candidate, config: GradingConfig, records: list[LogRecord]
) -> float | None:
    """The most judging could cost for these requests: every answer reaches the judge (two
    calls each) and answers are about as long as the originals. None if the judge has no price."""
    total = 0.0
    for record in records:
        if not config.rule_for(record.task_type).uses_judge:
            continue
        request_tokens = sum(estimate_tokens(m.content) for m in record.conversation())
        per_call = candidate_cost(
            prices,
            judge,
            input_tokens=request_tokens
            + 2 * estimate_tokens(record.response)
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
