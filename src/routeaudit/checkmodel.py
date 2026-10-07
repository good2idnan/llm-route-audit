"""Test one new model against an existing audit: replay the same sample, grade it, and show
whether it would change the policy."""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from routeaudit.grading.grade import ORIGINAL, Grade
from routeaudit.records import LogRecord
from routeaudit.replay import ReplayResult
from routeaudit.report import Report


def label(model: str, effort: str | None) -> str:
    return f"{model} @ {effort}" if effort else model


def sample_records(records: list[LogRecord], results: list[ReplayResult]) -> list[LogRecord]:
    """The requests the earlier replay used, in log order, so the new model faces the same test."""
    used = {r.record_id for r in results}
    return [r for r in records if r.id in used]


def merge_results(existing: list[ReplayResult], new: list[ReplayResult]) -> list[ReplayResult]:
    """Add the new answers, replacing earlier answers from the same model and effort."""
    replaced = {(label(r.model, r.effort), r.record_id) for r in new}
    kept = [r for r in existing if (label(r.model, r.effort), r.record_id) not in replaced]
    return kept + new


def merge_grades(existing: list[dict[str, Any]], new: list[Grade]) -> list[dict[str, Any]]:
    """Add the new grades. Originals keep their earlier grades; other candidates are replaced."""
    keys = {(g["candidate"], g["record_id"]) for g in existing}
    fresh = [
        g.to_dict()
        for g in new
        if not (g.candidate == ORIGINAL and (g.candidate, g.record_id) in keys)
    ]
    replaced = {(g["candidate"], g["record_id"]) for g in fresh}
    return [g for g in existing if (g["candidate"], g["record_id"]) not in replaced] + fresh


def write_grade_dicts(path: str | Path, grades: list[dict[str, Any]]) -> None:
    with Path(path).open("w", encoding="utf-8", newline="\n") as f:
        for grade in grades:
            f.write(json.dumps(grade, ensure_ascii=False) + "\n")


@dataclass
class TaskChange:
    task: str
    before: str
    after: str
    new_model_pass: float | None
    new_model_cost: float | None
    new_model_graded: int

    @property
    def changed(self) -> bool:
        return self.before != self.after


def compare(before: Report, after: Report, new_label: str) -> list[TaskChange]:
    """Per task: the choice without the new model, the choice with it, and how it did."""
    old_choice = {t.task: _name(t.choice.label, t.original.model) for t in before.tasks}
    changes = []
    for t in after.tasks:
        option = next((o for o in t.options if o.label == new_label), None)
        changes.append(
            TaskChange(
                task=t.task,
                before=old_choice.get(t.task, _name(ORIGINAL, t.original.model)),
                after=_name(t.choice.label, t.original.model),
                new_model_pass=option.pass_rate if option else None,
                new_model_cost=option.cost_ratio if option else None,
                new_model_graded=option.graded if option else 0,
            )
        )
    return changes


def _name(choice_label: str, original_model: str) -> str:
    return f"keep {original_model}" if choice_label == ORIGINAL else choice_label


def render_check(
    new_label: str, changes: list[TaskChange], before: Report, after: Report, spent: float
) -> str:
    from routeaudit.display import INDENT, pct, table, usd
    from routeaudit.report_view import short_name

    def rate(value: float | None) -> str:
        return "-" if value is None else pct(value)

    lines = [f"Model check: {new_label}", f"{INDENT}Spent {usd(spent)}.", ""]
    lines += table(
        ["Task", "New model pass", "Its cost", "Graded", "Use before", "Use now"],
        [
            [
                c.task,
                rate(c.new_model_pass),
                rate(c.new_model_cost),
                f"{c.new_model_graded}",
                short_name(c.before),
                short_name(c.after) + ("  <- changed" if c.changed else ""),
            ]
            for c in changes
        ],
        text_columns=1,
    )
    wins = [c.task for c in changes if c.changed and c.after == new_label]
    lines += [
        "",
        f"Projected savings: {pct(before.savings_share)} before, {pct(after.savings_share)} "
        "with the new model in the running.",
    ]
    if wins:
        lines.append(
            f"{short_name(new_label)} becomes the best choice for: {', '.join(wins)}. "
            "Run `routeaudit export` to update the policy."
        )
    else:
        lines.append(f"{short_name(new_label)} does not change the policy.")
    if any(c.new_model_graded < after.min_samples for c in changes):
        lines.append(
            f"Some tasks have fewer than {after.min_samples} graded answers, so the policy keeps "
            "the current model there. Replay a bigger sample for a firm answer."
        )
    return "\n".join(lines)
