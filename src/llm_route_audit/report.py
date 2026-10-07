"""Combine replay costs and grades into a cost/quality comparison and a routing policy.

For each task type, every option (the original model and each candidate) gets a pass rate
from the grades and a cost ratio against the original on the same sampled requests. The
policy picks, per task, the cheapest option that keeps quality within the target, and only
when there are enough graded answers to trust the numbers.

Agent logs are routed per session type, never per step: every step of a session counts
toward its session's type, so a whole session stays on one model and keeps its cache.
"""

import json
import math
import random
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import UTC, datetime
from functools import cached_property
from pathlib import Path
from typing import Any

from llm_route_audit.agent_report import SessionTypeReport, session_report
from llm_route_audit.analyze import UNLABELLED, build_profile, usage_of
from llm_route_audit.costs import PriceTable, UnknownModelError
from llm_route_audit.grading.grade import ORIGINAL
from llm_route_audit.policy import litellm_model
from llm_route_audit.records import LogRecord
from llm_route_audit.replay import ReplayResult
from llm_route_audit.router_report import RouterAudit, audit_routers
from llm_route_audit.sampling import session_type, sessions

DEFAULT_TARGET = 0.95  # keep at least 95% of the original's pass rate
DEFAULT_MIN_SAMPLES = 10
ROUTER_STRATEGY = "Router {label}"
BOOTSTRAP_ROUNDS = 1000


def wilson_interval(passed: int, n: int, z: float = 1.96) -> tuple[float, float] | None:
    """95% confidence interval for a pass rate; honest about small samples."""
    if n == 0:
        return None
    p = passed / n
    denominator = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denominator
    margin = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denominator
    return max(0.0, centre - margin), min(1.0, centre + margin)


def ratio_interval(
    pairs: list[tuple[float, float]], rounds: int = BOOTSTRAP_ROUNDS, seed: int = 0
) -> tuple[float, float] | None:
    """95% range for a cost ratio (sum of costs over sum of original costs), found by
    re-drawing the sampled requests at random many times (a bootstrap). The fixed seed
    gives the same range for the same data."""
    if len(pairs) < 2:
        return None
    rng = random.Random(seed)
    ratios = []
    for _ in range(rounds):
        drawn = rng.choices(pairs, k=len(pairs))
        base = sum(original for _, original in drawn)
        if base > 0:
            ratios.append(sum(cost for cost, _ in drawn) / base)
    if not ratios:
        return None
    ratios.sort()
    last = len(ratios) - 1
    return ratios[round(0.025 * last)], ratios[round(0.975 * last)]


@dataclass
class OptionStats:
    label: str
    model: str
    effort: str | None = None
    provider: str | None = None
    graded: int = 0
    passed: int = 0
    cost: float = 0.0  # this option's cost on its sampled requests
    original_cost: float = 0.0  # what the same requests cost as logged
    # (cost, original cost) per request, for the cost ratio's range
    pairs: list[tuple[float, float]] = field(default_factory=list, repr=False)
    router: bool = False  # a router candidate, which picks a model per request

    @property
    def pass_rate(self) -> float | None:
        return self.passed / self.graded if self.graded else None

    @property
    def cost_ratio(self) -> float | None:
        return self.cost / self.original_cost if self.original_cost > 0 else None

    @property
    def interval(self) -> tuple[float, float] | None:
        return wilson_interval(self.passed, self.graded)

    @cached_property
    def cost_interval(self) -> tuple[float, float] | None:
        """95% range for the cost ratio. Read it once all requests are added."""
        return ratio_interval(self.pairs)

    def to_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "model": self.model,
            "effort": self.effort,
            "provider": self.provider,
            "router": self.router,
            "graded": self.graded,
            "passed": self.passed,
            "pass_rate": self.pass_rate,
            "interval_95": self.interval,
            "cost_ratio": self.cost_ratio,
            "cost_interval_95": self.cost_interval,
        }


@dataclass
class TaskReport:
    task: str
    request_share: float
    logged_cost: float  # full-log cost of this task
    original: OptionStats
    options: list[OptionStats]
    choice: OptionStats
    reason: str

    @property
    def savings(self) -> float:
        ratio = self.choice.cost_ratio
        return 0.0 if ratio is None else self.logged_cost * (1 - ratio)


@dataclass
class Strategy:
    name: str
    quality: float | None  # traffic-weighted pass rate
    cost_ratio: float | None  # cost relative to the current setup
    coverage: float  # share of traffic with data behind these numbers


@dataclass
class Report:
    tasks: list[TaskReport]
    strategies: list[Strategy]
    target: float
    min_samples: int
    total_logged_cost: float
    monthly_cost: float | None
    judged: int = 0  # answers with a verdict from both judge orders
    judge_agreed: int = 0  # ... where both orders gave the same verdict
    sessions: list[SessionTypeReport] = field(default_factory=list)  # agent logs only
    routers: list[RouterAudit] = field(default_factory=list)  # router candidates only
    generated_at: datetime = field(default_factory=lambda: datetime.now(UTC))

    @property
    def policy(self) -> Strategy:
        return self.strategies[-1]

    @property
    def savings(self) -> float:
        return sum(t.savings for t in self.tasks)

    @property
    def savings_share(self) -> float:
        return self.savings / self.total_logged_cost if self.total_logged_cost else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "generated_at": self.generated_at.isoformat(),
            "target": self.target,
            "min_samples": self.min_samples,
            "savings_share": self.savings_share,
            "monthly_cost": self.monthly_cost,
            "monthly_savings": (
                self.monthly_cost * self.savings_share if self.monthly_cost is not None else None
            ),
            "judged": self.judged,
            "judge_agreed": self.judge_agreed,
            "sessions": [s.to_dict() for s in self.sessions],
            "routers": [r.to_dict() for r in self.routers],
            "strategies": [s.__dict__ for s in self.strategies],
            "tasks": [
                {
                    "task": t.task,
                    "request_share": t.request_share,
                    "logged_cost": t.logged_cost,
                    "choice": t.choice.label,
                    "reason": t.reason,
                    "original": t.original.to_dict(),
                    "options": [o.to_dict() for o in t.options],
                }
                for t in self.tasks
            ],
        }


def load_grades(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _logged_cost(prices: PriceTable, record: LogRecord) -> float | None:
    usage = usage_of(record)
    try:
        return prices.cost(
            record.model,
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
            cache_read_tokens=usage.cache_read_tokens,
            cache_write_tokens=usage.cache_write_tokens,
        )
    except UnknownModelError:
        return None


def routed_cost(prices: PriceTable, record: LogRecord, result: ReplayResult) -> float | None:
    """What a request would cost on the candidate if the candidate took over this traffic.

    A replay sends each request once, so it never reads from a prompt cache. In production
    the candidate would cache the way the original did (a whole session on one model reads
    most of its history from cache). So when the log shows cache use, the candidate's input
    is split into uncached, cache-read and cache-write tokens in the original's proportions.
    """
    original = usage_of(record)
    cached = original.cache_read_tokens + original.cache_write_tokens
    replayed = sum(
        t or 0 for t in (result.input_tokens, result.cache_read_tokens, result.cache_write_tokens)
    )
    if not cached or not replayed or result.output_tokens is None:
        return result.cost
    scale = replayed / (original.input_tokens + cached)
    try:
        return prices.cost(
            result.model,
            input_tokens=round(original.input_tokens * scale),
            output_tokens=result.output_tokens,
            cache_read_tokens=round(original.cache_read_tokens * scale),
            cache_write_tokens=round(original.cache_write_tokens * scale),
        )
    except UnknownModelError:
        return result.cost


def by_session_type(records: list[LogRecord]) -> list[LogRecord]:
    """Records with each agent step's task type set to its session's type."""
    if not any(r.session_id for r in records):
        return records
    kind = {}
    for steps in sessions(records):
        name = session_type(steps)
        kind.update({s.id: name for s in steps})
    return [
        r
        if (r.task_type or UNLABELLED) == kind[r.id]
        else r.model_copy(update={"task_type": kind[r.id]})
        for r in records
    ]


def _choose(
    original: OptionStats, options: list[OptionStats], target: float, min_samples: int
) -> tuple[OptionStats, str]:
    if original.pass_rate is None:
        return original, "the original answers were not graded"
    cheaper = [o for o in options if o.cost_ratio is not None and o.cost_ratio < 1]
    if not cheaper:
        return original, "no cheaper option was tested"
    enough = [o for o in cheaper if o.graded >= min_samples]
    if not enough:
        most = max(o.graded for o in cheaper)
        return (
            original,
            f"not enough data: {most} graded answer{'' if most == 1 else 's'}, need {min_samples}",
        )
    good = [o for o in enough if o.passed and o.pass_rate >= target * original.pass_rate]
    if not good:
        return original, f"no cheaper option kept {target:.0%} of the original's pass rate"
    best = min(good, key=lambda o: o.cost_ratio)
    return best, f"cheapest option within {target:.0%} of the original's pass rate"


def build_report(
    records: list[LogRecord],
    results: list[ReplayResult],
    grades: list[dict[str, Any]],
    prices: PriceTable,
    target: float = DEFAULT_TARGET,
    min_samples: int = DEFAULT_MIN_SAMPLES,
) -> Report:
    records = by_session_type(records)
    by_id = {r.id: r for r in records}
    profile = build_profile(records, prices)
    task_costs = {g.name: g.cost for g in profile.by_task}
    task_requests = {g.name: g.requests for g in profile.by_task}

    outcomes: dict[tuple[str, str], str] = {
        (g["candidate"], g["record_id"]): g["outcome"] for g in grades
    }
    votes = [g.get("judge_votes") or [] for g in grades]

    options: dict[str, dict[str, OptionStats]] = defaultdict(dict)
    originals: dict[str, OptionStats] = {}
    for result in results:
        record = by_id.get(result.record_id)
        if record is None:
            continue
        task = record.task_type or UNLABELLED
        label = result.label
        option = options[task].setdefault(
            label,
            OptionStats(label, result.model, result.effort, result.provider, router=result.router),
        )
        cost = routed_cost(prices, record, result)
        original_cost = _logged_cost(prices, record)
        if cost is not None and original_cost is not None:
            option.cost += cost
            option.original_cost += original_cost
            option.pairs.append((cost, original_cost))
        outcome = outcomes.get((label, record.id))
        if outcome in ("pass", "fail"):
            option.graded += 1
            option.passed += outcome == "pass"

    for (candidate, record_id), outcome in outcomes.items():
        record = by_id.get(record_id)
        if candidate != ORIGINAL or record is None or outcome not in ("pass", "fail"):
            continue
        task = record.task_type or UNLABELLED
        original = originals.setdefault(task, OptionStats(ORIGINAL, record.model))
        original.graded += 1
        original.passed += outcome == "pass"

    total_requests = len(records)
    tasks = []
    for task in sorted(options, key=lambda t: -task_costs.get(t, 0.0)):
        original = originals.get(task, OptionStats(ORIGINAL, "unknown"))
        # The original is the yardstick: its cost ratio is 1 by definition.
        original.cost = original.original_cost = 1.0
        tested = list(options[task].values())
        choice, reason = _choose(original, tested, target, min_samples)
        tasks.append(
            TaskReport(
                task=task,
                request_share=task_requests.get(task, 0) / total_requests,
                logged_cost=task_costs.get(task, 0.0),
                original=original,
                options=tested,
                choice=choice,
                reason=reason,
            )
        )

    strategies = _strategies(tasks)
    return Report(
        tasks=tasks,
        strategies=strategies,
        target=target,
        min_samples=min_samples,
        total_logged_cost=profile.total.cost,
        monthly_cost=profile.monthly_cost,
        judged=sum(1 for v in votes if len(v) == 2 and None not in v),
        judge_agreed=sum(1 for v in votes if len(v) == 2 and None not in v and v[0] == v[1]),
        sessions=session_report(
            records,
            results,
            grades,
            lambda record, result: routed_cost(prices, record, result),
            lambda record: _logged_cost(prices, record),
        ),
        routers=audit_routers(
            results,
            outcomes,
            {r.id: r.task_type or UNLABELLED for r in records},
            strategies,
            ROUTER_STRATEGY,
        ),
    )


def _weighted(tasks: list[TaskReport], pick) -> Strategy:  # type: ignore[no-untyped-def]
    """Traffic-weighted quality and cost for a strategy that uses `pick(task)` per task."""
    quality_weight = quality = cost_weight = cost = 0.0
    covered = 0.0
    for t in tasks:
        option = pick(t)
        if option is None or option.pass_rate is None or option.cost_ratio is None:
            continue
        covered += t.request_share
        quality += t.request_share * option.pass_rate
        quality_weight += t.request_share
        cost += t.logged_cost * option.cost_ratio
        cost_weight += t.logged_cost
    return Strategy(
        name="",
        quality=quality / quality_weight if quality_weight else None,
        cost_ratio=cost / cost_weight if cost_weight else None,
        coverage=covered / sum(t.request_share for t in tasks) if tasks else 0.0,
    )


def _strategies(tasks: list[TaskReport]) -> list[Strategy]:
    current = _weighted(tasks, lambda t: t.original)
    current.name = "Current setup (as logged)"
    labels = list(dict.fromkeys(o.label for t in tasks for o in t.options))
    routers = {o.label for t in tasks for o in t.options if o.router}
    always = []
    for label in labels:
        strategy = _weighted(tasks, lambda t, label=label: _find(t.options, label))
        strategy.name = (
            ROUTER_STRATEGY.format(label=label) if label in routers else f"Always {label}"
        )
        always.append(strategy)
    policy = _weighted(tasks, lambda t: t.choice)
    policy.name = "Per-task policy"
    return [current, *always, policy]


def _find(options: list[OptionStats], label: str) -> OptionStats | None:
    return next((o for o in options if o.label == label), None)


def policy_yaml(report: Report) -> str:
    """The recommended routing table, with the evidence for each line as a comment."""
    models = Counter(t.original.model for t in report.tasks)
    default = models.most_common(1)[0][0] if models else "unknown"
    lines = [
        "# llm-route-audit routing policy",
        f"# Generated {report.generated_at:%Y-%m-%d %H:%M} UTC. Quality target: "
        f"{report.target:.0%} of the original pass rate, at least {report.min_samples} "
        f"graded answer{'' if report.min_samples == 1 else 's'} per option.",
        "# Each route lists the model to use for that task type and why.",
        "version: 1",
        f"default: {{model: {json.dumps(default)}}}",
        "routes:",
    ]
    for t in report.tasks:
        c = t.choice
        model = t.original.model if c.label == ORIGINAL else c.model
        fields = [f"model: {json.dumps(model)}"]
        if c.label != ORIGINAL and c.effort:
            fields.append(f"effort: {c.effort}")
        if c.label != ORIGINAL and c.provider:
            fields.append(f"provider: {c.provider}")
        if c.label != ORIGINAL and c.pass_rate is not None:
            # What the audit measured, so `llm-route-audit monitor` can check it still holds.
            fields.append(f"reference: {json.dumps(t.original.model)}")
            fields.append(f"expected_pass_rate: {c.pass_rate:.3f}")
        evidence = t.reason
        if c.label != ORIGINAL and c.pass_rate is not None and c.cost_ratio is not None:
            evidence += f"; pass {c.pass_rate:.0%} on {c.graded}, cost {c.cost_ratio:.0%}"
            if c.cost_interval is not None:
                evidence += f" ({c.cost_interval[0]:.0%}-{c.cost_interval[1]:.0%})"
        lines.append(f"  {t.task}: {{{', '.join(fields)}}}  # {evidence}")
    return "\n".join(lines) + "\n"


PROVIDER_KEYS = {"anthropic": "ANTHROPIC_API_KEY", "openrouter": "OPENROUTER_API_KEY"}


def policy_litellm(report: Report) -> str:
    """The policy as a LiteLLM proxy config: one model alias per task type.

    Your app sends each request to the alias for its task (e.g. model="route/draft_reply"),
    and LiteLLM forwards it to the model the audit chose.
    """
    lines = [
        "# LiteLLM proxy config generated by llm-route-audit",
        f"# {report.generated_at:%Y-%m-%d %H:%M} UTC. Call the alias for each task type from "
        'your app, e.g. model="route/draft_reply".',
        "model_list:",
    ]
    models = Counter(t.original.model for t in report.tasks)
    routes = [(t.task, t.choice, t) for t in report.tasks]
    if models:
        default = models.most_common(1)[0][0]
        lines += _litellm_entry("route/default", default, None, None, "your current default")
    for task, choice, t in routes:
        if choice.label == ORIGINAL:
            lines += _litellm_entry(f"route/{task}", t.original.model, None, None, t.reason)
        else:
            lines += _litellm_entry(
                f"route/{task}", choice.model, choice.effort, choice.provider, t.reason
            )
    return "\n".join(lines) + "\n"


def _litellm_entry(
    alias: str, model: str, effort: str | None, provider: str | None, why: str
) -> list[str]:
    name, inferred = litellm_model(model, provider)
    entry = [
        f"  - model_name: {json.dumps(alias)}  # {why}",
        "    litellm_params:",
        f"      model: {json.dumps(name)}",
    ]
    if inferred in PROVIDER_KEYS:
        entry.append(f"      api_key: os.environ/{PROVIDER_KEYS[inferred]}")
    if inferred == "ollama":
        entry.append('      api_base: "http://localhost:11434"')
    if effort:
        entry.append(f"      reasoning_effort: {effort}")
    return entry
