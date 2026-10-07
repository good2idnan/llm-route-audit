"""Shadow checks: confirm routed traffic still meets the quality the audit promised.

After you adopt a policy, your production logs hold answers from the cheaper models. `monitor`
samples requests from each switched task, replays them on the route's reference model (the
strong model it replaced), grades the production answer against that reference answer, and
compares the pass rate with the one the audit measured.
"""

import random
from collections import defaultdict
from dataclasses import dataclass, field

from routeaudit.analyze import UNLABELLED, estimate_tokens
from routeaudit.candidates import Candidate
from routeaudit.costs import PriceTable
from routeaudit.grading.grade import ORIGINAL, Grade, GradingConfig, judge_upper_bound
from routeaudit.policy import Policy, Route
from routeaudit.records import LogRecord
from routeaudit.replay import ReplayResult, candidate_cost
from routeaudit.report import wilson_interval
from routeaudit.runner import Execution, Job

DEFAULT_PER_TASK = 20
DEFAULT_TOLERANCE = 0.05
DEFAULT_MIN_CHECKS = 10


@dataclass
class Check:
    task: str
    route: Route
    record: LogRecord  # a production request, answered by the route's cheaper model
    reference: Candidate


@dataclass
class MonitorPlan:
    checks: list[Check]
    not_checked: dict[str, str] = field(default_factory=dict)

    def shadow_jobs(self) -> list[Job]:
        return [Job(c.reference, c.record.conversation()) for c in self.checks]

    def estimate_cost(
        self, prices: PriceTable, config: GradingConfig, judge: Candidate
    ) -> tuple[float | None, float | None]:
        """Shadow replay cost, and the most the judge could cost if every answer reaches it."""
        shadow: float | None = 0.0
        for check in self.checks:
            one = candidate_cost(
                prices,
                check.reference,
                input_tokens=sum(estimate_tokens(m.content) for m in check.record.conversation()),
                output_tokens=estimate_tokens(check.record.response),
            )
            shadow = None if one is None or shadow is None else shadow + one
        return shadow, judge_upper_bound(prices, judge, config, [c.record for c in self.checks])


def plan_monitor(
    records: list[LogRecord],
    policy: Policy,
    per_task: int = DEFAULT_PER_TASK,
    seed: int = 0,
    reference_model: str | None = None,
) -> MonitorPlan:
    """Pick up to `per_task` production requests from every route that switched models."""
    by_task: dict[str, list[LogRecord]] = defaultdict(list)
    for record in records:
        by_task[record.task_type or UNLABELLED].append(record)

    rng = random.Random(seed)
    plan = MonitorPlan(checks=[])
    for task, route in sorted(policy.routes.items()):
        if not route.switched:
            plan.not_checked[task] = "keeps its original model, so there is nothing to compare"
            continue
        task_records = sorted(by_task.get(task, []), key=lambda r: r.id)
        if not task_records:
            plan.not_checked[task] = "no requests for this task in these logs"
            continue
        if len(task_records) > per_task:
            task_records = sorted(rng.sample(task_records, per_task), key=lambda r: r.id)
        reference = Candidate(model=reference_model or route.reference)
        plan.checks += [Check(task, route, r, reference) for r in task_records]
    for task in sorted(set(by_task) - set(policy.routes)):
        plan.not_checked[task] = "not in the policy (served by the default model)"
    return plan


def grading_inputs(
    plan: MonitorPlan, execution: Execution
) -> tuple[list[LogRecord], list[ReplayResult], int]:
    """Turn shadow answers into grading inputs: the reference model's answer plays the
    original, and the production answer plays the candidate. Returns how many shadow calls
    failed, since those requests can't be checked."""
    references, answers, failed = [], [], 0
    for check, outcome in zip(plan.checks, execution.outcomes, strict=True):
        if outcome.completion is None or outcome.completion.status != "ok":
            failed += 1
            continue
        references.append(
            check.record.model_copy(
                update={"response": outcome.completion.text, "model": check.reference.model}
            )
        )
        answers.append(
            ReplayResult(
                record_id=check.record.id,
                task_type=check.task,
                model=check.route.model,
                effort=check.route.effort,
                provider=check.route.provider,
                status="ok",
                response=check.record.response,
            )
        )
    return references, answers, failed


@dataclass
class TaskHealth:
    task: str
    route: Route
    checked: int
    passed: int
    status: str  # OK | WARN | ALERT | WAIT | NO DATA
    note: str

    @property
    def pass_rate(self) -> float | None:
        return self.passed / self.checked if self.checked else None

    @property
    def interval(self) -> tuple[float, float] | None:
        return wilson_interval(self.passed, self.checked)


def assess(
    grades: list[Grade],
    plan: MonitorPlan,
    tolerance: float = DEFAULT_TOLERANCE,
    min_checks: int = DEFAULT_MIN_CHECKS,
) -> list[TaskHealth]:
    """Compare each switched task's pass rate now with the rate the audit measured.

    ALERT: we're 95% sure the pass rate is below the audited rate minus the tolerance.
    WARN: the pass rate is below that line, but the sample can't rule out bad luck.
    WAIT: no problem so far, but fewer than `min_checks` answers were checked.
    """
    routes = {c.task: c.route for c in plan.checks}
    counts: dict[str, list[int]] = {task: [0, 0] for task in routes}
    for grade in grades:
        task = grade.task_type or UNLABELLED
        if grade.candidate == ORIGINAL or grade.outcome == "ungraded" or task not in counts:
            continue
        counts[task][0] += 1
        counts[task][1] += grade.outcome == "pass"

    health = []
    for task, (checked, passed) in counts.items():
        route = routes[task]
        floor = (route.expected_pass_rate or 0.0) - tolerance
        if checked == 0:
            status, note = "NO DATA", "no answers could be checked"
        else:
            rate = passed / checked
            high = wilson_interval(passed, checked)[1]
            if high < floor:
                status, note = "ALERT", f"quality dropped below the audited {floor:.0%} floor"
            elif rate < floor:
                status, note = "WARN", f"below the {floor:.0%} floor; check more requests"
            elif checked < min_checks:
                status, note = "WAIT", f"fine so far; need {min_checks} checks to be sure"
            else:
                status, note = "OK", f"at or above the {floor:.0%} floor"
        health.append(TaskHealth(task, route, checked, passed, status, note))
    return health


def render_monitor(
    health: list[TaskHealth],
    plan: MonitorPlan,
    source: str,
    spent: float,
    shadow_failed: int,
    out_path: str,
) -> str:
    from routeaudit.display import INDENT, pct, table, usd
    from routeaudit.report_view import short_name

    references = sorted({c.reference.label for c in plan.checks})
    lines = [
        f"Quality monitor: {source}",
        f"{INDENT}Compared with: {', '.join(references)}. Spent {usd(spent)}.",
        f"{INDENT}Details saved to {out_path}",
        "",
    ]
    rows = []
    for h in health:
        interval = h.interval
        rows.append(
            [
                h.task,
                short_name(h.route.label),
                f"{h.checked}",
                "-" if h.pass_rate is None else pct(h.pass_rate),
                "-" if interval is None else f"{interval[0]:.0%}-{interval[1]:.0%}",
                pct(h.route.expected_pass_rate or 0.0),
                h.status,
                h.note,
            ]
        )
    lines += table(
        ["Task", "Model", "Checked", "Pass", "95% range", "Audited", "Status", "Why"],
        rows,
        numeric_columns={2, 3, 4, 5},
    )
    if plan.not_checked:
        lines += ["", "Not checked"]
        lines += [f"{INDENT}- {task}: {why}" for task, why in plan.not_checked.items()]
    if shadow_failed:
        lines += ["", f"{shadow_failed} shadow calls failed; those requests were not checked."]
    alerts = [h.task for h in health if h.status == "ALERT"]
    warnings = [h.task for h in health if h.status == "WARN"]
    lines.append("")
    if alerts:
        lines.append(
            f"ALERT: quality dropped on {', '.join(alerts)}. Consider routing these tasks back "
            "to the reference model, then re-run the audit."
        )
    elif warnings:
        lines.append(f"Warning on {', '.join(warnings)}. Check more requests with --per-task.")
    else:
        lines.append("No quality drop detected on the routed tasks.")
    return "\n".join(lines)
