"""Learn from outcomes: real-world feedback on routed traffic, such as thumbs up or down,
tests passed, or tasks completed.

Feedback comes from each log record's `outcome` field, or from a separate file of
(record_id, outcome) lines, such as the one the runtime router writes. For every task the
policy switched to a cheaper model, the share of good outcomes on that model is compared with
the share on the model it replaced (from logs before the switch, or from traffic you keep on
it). A route whose good-outcome rate is clearly lower is marked REVERT, and `--out` writes a
policy with those routes sent back to the model they replaced.
"""

import csv
import json
from collections import Counter
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from llm_route_audit.analyze import UNLABELLED
from llm_route_audit.policy import Policy, Route
from llm_route_audit.records import LogRecord
from llm_route_audit.report import wilson_interval

GOOD = {
    "good", "pass", "passed", "success", "succeeded", "positive", "thumbs_up", "up", "yes",
    "true", "1", "accepted", "resolved", "completed", "done",
}  # fmt: skip
BAD = {
    "bad", "fail", "failed", "failure", "negative", "thumbs_down", "down", "no", "false", "0",
    "rejected", "unresolved", "error", "escalated", "abandoned",
}  # fmt: skip
DEFAULT_MIN_OUTCOMES = 30
DEFAULT_TOLERANCE = 0.05
PREFIXES = ("openrouter/", "openai/", "gemini/", "ollama/", "anthropic/")


def outcome_key(value: str) -> str:
    return value.strip().lower().replace(" ", "_").replace("-", "_")


def normalize(value: object, good: set[str] = GOOD, bad: set[str] = BAD) -> bool | None:
    """True for a good outcome, False for a bad one, None when it isn't recognised."""
    if isinstance(value, bool):
        return value
    if value is None:
        return None
    key = outcome_key(str(value))
    if key in good:
        return True
    if key in bad:
        return False
    return None


def load_outcome_file(path: str | Path) -> dict[str, str]:
    """{record_id: outcome} from a CSV (with a header) or JSONL file. Later lines win."""
    path = Path(path)
    text = path.read_text(encoding="utf-8-sig")
    rows = (
        list(csv.DictReader(text.splitlines()))
        if path.suffix.lower() == ".csv"
        else [json.loads(line) for line in text.splitlines() if line.strip()]
    )
    found = {}
    for row in rows:
        if row.get("record_id") and row.get("outcome") is not None:
            found[str(row["record_id"])] = str(row["outcome"])
    return found


def _bare(model: str) -> str:
    for prefix in PREFIXES:
        if model.startswith(prefix):
            return _bare(model.removeprefix(prefix))
    return model


def same_model(logged: str, policy_model: str) -> bool:
    """Logs and policies may name a model with or without a provider prefix."""
    return logged == policy_model or _bare(logged) == _bare(policy_model)


@dataclass
class Tally:
    good: int = 0
    total: int = 0

    @property
    def rate(self) -> float | None:
        return self.good / self.total if self.total else None

    @property
    def interval(self) -> tuple[float, float] | None:
        return wilson_interval(self.good, self.total)


@dataclass
class RouteOutcome:
    task: str
    route: Route
    routed: Tally
    original: Tally
    status: str  # OK | WATCH | REVERT | WAIT
    note: str


@dataclass
class OutcomeReport:
    routes: list[RouteOutcome]
    with_outcome: int  # records with a recognised outcome
    unrecognised: Counter = field(default_factory=Counter)
    unmatched_ids: int = 0  # outcome-file lines whose record_id isn't in the logs
    not_switched: list[str] = field(default_factory=list)

    @property
    def reverts(self) -> list[RouteOutcome]:
        return [r for r in self.routes if r.status == "REVERT"]


def assess_outcomes(
    records: list[LogRecord],
    policy: Policy,
    extra: dict[str, str] | None = None,
    good: set[str] = GOOD,
    bad: set[str] = BAD,
    min_outcomes: int = DEFAULT_MIN_OUTCOMES,
    tolerance: float = DEFAULT_TOLERANCE,
) -> OutcomeReport:
    """Compare good-outcome rates on each switched route with the model it replaced.

    REVERT: we're 95% sure the routed model does worse (the ranges don't overlap).
    WATCH: the routed model's rate is lower by more than `tolerance`, but the ranges overlap.
    WAIT: fewer than `min_outcomes` outcomes on either model.
    """
    extra = extra or {}
    ids = {r.id for r in records}
    unrecognised: Counter = Counter()
    tallies: dict[tuple[str, str], Tally] = {}
    with_outcome = 0
    for record in records:
        raw = extra.get(record.id, record.outcome)
        if raw is None:
            continue
        verdict = normalize(raw, good, bad)
        if verdict is None:
            unrecognised[str(raw)] += 1
            continue
        with_outcome += 1
        tally = tallies.setdefault((record.task_type or UNLABELLED, record.model), Tally())
        tally.total += 1
        tally.good += verdict

    def tally_for(task: str, model: str) -> Tally:
        total = Tally()
        for (t, logged), tally in tallies.items():
            if t == task and same_model(logged, model):
                total.good += tally.good
                total.total += tally.total
        return total

    report = OutcomeReport(
        routes=[],
        with_outcome=with_outcome,
        unrecognised=unrecognised,
        unmatched_ids=sum(1 for record_id in extra if record_id not in ids),
    )
    for task, route in sorted(policy.routes.items()):
        if not route.reference or same_model(route.model, route.reference):
            report.not_switched.append(task)
            continue
        routed, original = tally_for(task, route.model), tally_for(task, route.reference)
        status, note = _judge(routed, original, min_outcomes, tolerance)
        report.routes.append(RouteOutcome(task, route, routed, original, status, note))
    return report


def _judge(routed: Tally, original: Tally, min_outcomes: int, tolerance: float) -> tuple[str, str]:
    if original.total == 0:
        return "WAIT", "no outcomes for the model it replaced; keep some traffic on it"
    if routed.total == 0:
        return "WAIT", "no outcomes yet for the routed model"
    low_routed, high_routed = routed.interval  # type: ignore[misc]
    low_original, _ = original.interval  # type: ignore[misc]
    gap = (original.rate or 0.0) - (routed.rate or 0.0)
    if high_routed < low_original:
        return "REVERT", f"clearly fewer good outcomes ({gap:.0%} lower)"
    if routed.total < min_outcomes or original.total < min_outcomes:
        return "WAIT", f"need {min_outcomes} outcomes on each model to be sure"
    if gap > tolerance:
        return "WATCH", f"{gap:.0%} lower, but it could still be chance"
    return "OK", "as good as the model it replaced"


def updated_policy_yaml(policy: Policy, report: OutcomeReport) -> str:
    """The policy with REVERT routes sent back to the model they replaced."""
    reverted = {r.task: r for r in report.reverts}
    lines = [
        "# llm-route-audit routing policy, updated from outcomes",
        f"# Updated {datetime.now(UTC):%Y-%m-%d %H:%M} UTC. Routes marked 'reverted' had "
        "clearly fewer good outcomes than the model they replaced.",
        f"version: {policy.version}",
        f"default: {_route_fields(policy.default)}",
        "routes:",
    ]
    for task, route in policy.routes.items():
        if task in reverted:
            r = reverted[task]
            back = Route(model=route.reference or route.model)
            lines.append(
                f"  {task}: {_route_fields(back)}  # reverted from {route.label}: good outcomes "
                f"{r.routed.rate:.0%} on {r.routed.total} vs {r.original.rate:.0%} on "
                f"{r.original.total}"
            )
        else:
            lines.append(f"  {task}: {_route_fields(route)}")
    return "\n".join(lines) + "\n"


def _route_fields(route: Route) -> str:
    fields = [f"model: {json.dumps(route.model)}"]
    if route.effort:
        fields.append(f"effort: {route.effort}")
    if route.provider:
        fields.append(f"provider: {route.provider}")
    if route.reference:
        fields.append(f"reference: {json.dumps(route.reference)}")
    if route.expected_pass_rate is not None:
        fields.append(f"expected_pass_rate: {route.expected_pass_rate:.3f}")
    return "{" + ", ".join(fields) + "}"


def render_outcomes(report: OutcomeReport, source: str) -> str:
    from llm_route_audit.display import INDENT, pct, table
    from llm_route_audit.report_view import short_name

    def cell(t: Tally) -> str:
        if not t.total:
            return "-"
        low, high = t.interval  # type: ignore[misc]
        return f"{t.good}/{t.total} {pct(t.rate or 0.0)} ({low:.0%}-{high:.0%})"

    lines = [
        f"Outcomes: {source}",
        f"{INDENT}{report.with_outcome} requests with a good or bad outcome.",
        "",
    ]
    lines += table(
        [
            "Task",
            "Routed to",
            "Good outcomes",
            "Model it replaced",
            "Good outcomes",
            "Status",
            "Why",
        ],
        [
            [
                r.task,
                short_name(r.route.label),
                cell(r.routed),
                short_name(r.route.reference or ""),
                cell(r.original),
                r.status,
                r.note,
            ]
            for r in report.routes
        ],
        text_columns=2,
    )
    if report.not_switched:
        lines += ["", f"Not compared (kept their model): {', '.join(report.not_switched)}"]
    if report.unrecognised:
        common = ", ".join(f"{v!r} ({n})" for v, n in report.unrecognised.most_common(5))
        lines += [
            "",
            f"Outcomes not recognised as good or bad: {common}. Add them with --good or --bad.",
        ]
    if report.unmatched_ids:
        lines.append(f"{report.unmatched_ids} outcome lines name a record_id not in these logs.")
    lines.append("")
    if report.reverts:
        names = ", ".join(r.task for r in report.reverts)
        lines.append(f"REVERT: {names}. Route these tasks back to the model they replaced.")
    else:
        lines.append("No route needs to be reverted.")
    return "\n".join(lines)
