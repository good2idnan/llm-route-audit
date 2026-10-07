"""Sessions in the report: step by step, does each option do what the original model did?

Every sampled step was replayed with the exact history the original model saw, so each
step is a fair comparison. Per session type this collects how often an option made the
same tool calls, how its final answers graded, where sessions first went a different way,
and what a whole session costs.
"""

from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass, field
from statistics import median
from typing import Any

from llm_route_audit.records import LogRecord
from llm_route_audit.replay import ReplayResult
from llm_route_audit.sampling import session_type, sessions


@dataclass
class SessionOption:
    label: str
    tool_steps: int = 0  # graded steps where the original called tools
    tool_same: int = 0
    answers: int = 0  # graded steps where the original replied in text
    answers_passed: int = 0
    sessions: int = 0  # sessions with at least one graded step
    all_same: int = 0  # ... where every graded step passed
    first_splits: list[int] = field(default_factory=list)  # step number of each first failure
    cost: float = 0.0
    priced_sessions: int = 0

    @property
    def tool_rate(self) -> float | None:
        return self.tool_same / self.tool_steps if self.tool_steps else None

    @property
    def answer_rate(self) -> float | None:
        return self.answers_passed / self.answers if self.answers else None

    @property
    def typical_split(self) -> float | None:
        """Median step at which sessions that went a different way first did so."""
        return median(self.first_splits) if self.first_splits else None

    @property
    def cost_per_session(self) -> float | None:
        return self.cost / self.priced_sessions if self.priced_sessions else None

    def to_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "tool_steps": self.tool_steps,
            "tool_same": self.tool_same,
            "answers": self.answers,
            "answers_passed": self.answers_passed,
            "sessions": self.sessions,
            "all_same": self.all_same,
            "typical_first_split": self.typical_split,
            "cost_per_session": self.cost_per_session,
        }


@dataclass
class SessionTypeReport:
    task: str
    sessions: int
    steps: int
    original_cost: float | None  # average logged cost of one sampled session
    options: list[SessionOption]

    def to_dict(self) -> dict[str, Any]:
        return {
            "task": self.task,
            "sessions": self.sessions,
            "steps": self.steps,
            "original_cost_per_session": self.original_cost,
            "options": [o.to_dict() for o in self.options],
        }


def session_report(
    records: list[LogRecord],
    results: list[ReplayResult],
    grades: list[dict[str, Any]],
    step_cost: Callable[[LogRecord, ReplayResult], float | None],
    logged_cost: Callable[[LogRecord], float | None],
) -> list[SessionTypeReport]:
    """Per session type, how each option did on the sampled sessions. Empty for logs
    without session ids."""
    outcomes = {(g["candidate"], g["record_id"]): g["outcome"] for g in grades}
    replayed: dict[str, dict[str, ReplayResult]] = defaultdict(dict)
    for result in results:
        replayed[result.record_id][result.label] = result

    groups: dict[str, list[list[LogRecord]]] = defaultdict(list)
    for steps in sessions(records):
        if steps[0].session_id and any(s.id in replayed for s in steps):
            groups[session_type(steps)].append(steps)

    reports = []
    for task in sorted(groups):
        options: dict[str, SessionOption] = {}
        steps_seen = priced = 0
        original_total = 0.0
        for steps in groups[task]:
            steps_seen += sum(s.id in replayed for s in steps)
            costs = [logged_cost(s) for s in steps]
            if None not in costs:
                original_total += sum(c for c in costs if c is not None)
                priced += 1
            labels = sorted({label for s in steps for label in replayed.get(s.id, {})})
            for label in labels:
                option = options.setdefault(label, SessionOption(label))
                answers = [replayed.get(s.id, {}).get(label) for s in steps]
                verdicts = [outcomes.get((label, s.id)) for s in steps]
                _add_session(option, steps, answers, verdicts, step_cost)
        reports.append(
            SessionTypeReport(
                task=task,
                sessions=len(groups[task]),
                steps=steps_seen,
                original_cost=original_total / priced if priced else None,
                options=list(options.values()),
            )
        )
    return reports


def _add_session(
    option: SessionOption,
    steps: list[LogRecord],
    answers: list[ReplayResult | None],
    verdicts: list[str | None],
    step_cost: Callable[[LogRecord, ReplayResult], float | None],
) -> None:
    """Count one session's steps for one option."""
    first_split = None
    graded = 0
    cost: float | None = 0.0
    for number, (step, answer, verdict) in enumerate(
        zip(steps, answers, verdicts, strict=True), start=1
    ):
        one = None if answer is None else step_cost(step, answer)
        cost = None if one is None or cost is None else cost + one
        if verdict not in ("pass", "fail"):
            continue
        graded += 1
        passed = verdict == "pass"
        if step.response_tool_calls:
            option.tool_steps += 1
            option.tool_same += passed
        else:
            option.answers += 1
            option.answers_passed += passed
        if not passed and first_split is None:
            first_split = number
    if graded:
        option.sessions += 1
        if first_split is None:
            option.all_same += 1
        else:
            option.first_splits.append(first_split)
    if cost is not None:
        option.cost += cost
        option.priced_sessions += 1
