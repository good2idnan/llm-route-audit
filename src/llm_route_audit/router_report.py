"""Audit a router: what did it pick, how good were the answers, and does anything simpler win?

A router (OpenRouter's Auto Router, Jev Router, a LiteLLM auto-router, ...) is replayed like
any candidate, with `router: true`. Every answer records the model the router picked, so the
report can show its choices per task, grade them, and compare the router with the other
strategies: keeping the current setup, always using one model, or the per-task policy.
"""

from collections import defaultdict
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from llm_route_audit.replay import ReplayResult

if TYPE_CHECKING:
    from llm_route_audit.report import Strategy

UNKNOWN_PICK = "(not reported)"


@dataclass
class RouterPick:
    task: str
    model: str  # the model the router picked
    answers: int
    graded: int = 0
    passed: int = 0

    @property
    def pass_rate(self) -> float | None:
        return self.passed / self.graded if self.graded else None


@dataclass
class RouterAudit:
    label: str
    result: "Strategy"
    beaten_by: list["Strategy"]  # cheaper and at least as good, or better for the same cost
    picks: list[RouterPick]

    @property
    def verdict(self) -> str:
        r = self.result
        if r.quality is None or r.cost_ratio is None:
            return "Not enough graded answers to judge the router yet."
        if not self.beaten_by:
            return "No other strategy tested here beats it on both cost and quality."
        best = min(self.beaten_by, key=lambda s: (s.cost_ratio, -(s.quality or 0)))
        return (
            f"Beaten by {best.name}: {best.quality:.0%} quality at {best.cost_ratio:.0%} of "
            f"today's cost, against the router's {r.quality:.0%} at {r.cost_ratio:.0%}."
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "quality": self.result.quality,
            "cost_ratio": self.result.cost_ratio,
            "coverage": self.result.coverage,
            "verdict": self.verdict,
            "beaten_by": [s.name for s in self.beaten_by],
            "picks": [
                {
                    "task": p.task,
                    "model": p.model,
                    "answers": p.answers,
                    "graded": p.graded,
                    "passed": p.passed,
                }
                for p in self.picks
            ],
        }


def _beats(other: "Strategy", router: "Strategy") -> bool:
    """At least as good and as cheap, better on one, and measured on as much traffic."""
    if None in (other.quality, other.cost_ratio, router.quality, router.cost_ratio):
        return False
    if other.coverage + 1e-9 < router.coverage:
        return False
    at_least = other.quality >= router.quality and other.cost_ratio <= router.cost_ratio
    better = other.quality > router.quality or other.cost_ratio < router.cost_ratio
    return at_least and better


def audit_routers(
    results: list[ReplayResult],
    outcomes: dict[tuple[str, str], str],
    task_of: dict[str, str],
    strategies: list["Strategy"],
    router_strategy_name: str,
) -> list[RouterAudit]:
    """One audit per router candidate. `outcomes` maps (option label, record id) to the
    grade outcome; `task_of` maps record ids to the task the report used."""
    picks: dict[str, dict[tuple[str, str], RouterPick]] = defaultdict(dict)
    for result in results:
        if not result.router or result.record_id not in task_of:
            continue
        if result.status in ("error", "skipped"):
            continue
        task = task_of[result.record_id]
        model = result.served_model or UNKNOWN_PICK
        pick = picks[result.label].setdefault((task, model), RouterPick(task, model, 0))
        pick.answers += 1
        outcome = outcomes.get((result.label, result.record_id))
        if outcome in ("pass", "fail"):
            pick.graded += 1
            pick.passed += outcome == "pass"

    audits = []
    for label in sorted(picks):
        name = router_strategy_name.format(label=label)
        own = next((s for s in strategies if s.name == name), None)
        if own is None:
            continue
        audits.append(
            RouterAudit(
                label=label,
                result=own,
                beaten_by=[s for s in strategies if s.name != name and _beats(s, own)],
                picks=sorted(picks[label].values(), key=lambda p: (p.task, -p.answers, p.model)),
            )
        )
    return audits
