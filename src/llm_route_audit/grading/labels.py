"""Your own verdicts: a file of pass/fail labels that override the checks and the judge.

One label per answer, identified by the record id and the candidate label as it appears in
grades.jsonl (for example "claude-sonnet-5-5 @ low", or "original" for the logged answer).
CSV (with a header row) or JSONL:

    record_id,candidate,outcome,note
    req_0042,claude-sonnet-5-5 @ low,fail,wrong refund amount

Labelled answers are not sent to the judge, so they cost nothing to grade.
"""

import csv
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from llm_route_audit.grading.grade import ORIGINAL, GradePlan


class LabelError(ValueError):
    """A labels file that can't be read; the message says where."""


class HumanLabel(BaseModel):
    model_config = ConfigDict(extra="forbid")

    record_id: str = Field(min_length=1)
    candidate: str = Field(min_length=1)
    outcome: Literal["pass", "fail"]
    note: str = ""

    @field_validator("outcome", mode="before")
    @classmethod
    def _lowercase(cls, value: object) -> object:
        return value.strip().lower() if isinstance(value, str) else value

    @field_validator("candidate")
    @classmethod
    def _original(cls, value: str) -> str:
        value = value.strip()
        return ORIGINAL if value.lower() in ("original", ORIGINAL) else value


def _rows(path: Path) -> list[tuple[int, dict]]:
    text = path.read_text(encoding="utf-8-sig")
    if path.suffix.lower() == ".csv":
        reader = csv.DictReader(text.splitlines())
        return [
            (n, {k: v for k, v in row.items() if k and v not in (None, "")})
            for n, row in enumerate(reader, start=2)
        ]
    rows = []
    for n, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            rows.append((n, json.loads(line)))
        except json.JSONDecodeError:
            raise LabelError(f"{path.name} line {n}: not valid JSON") from None
    return rows


def load_human_labels(path: str | Path) -> list[HumanLabel]:
    """Read a labels file. A later label for the same answer replaces an earlier one."""
    path = Path(path)
    labels: dict[tuple[str, str], HumanLabel] = {}
    for n, row in _rows(path):
        try:
            label = HumanLabel.model_validate(row)
        except ValidationError as e:
            problem = e.errors()[0]
            where = ".".join(str(p) for p in problem["loc"]) or "row"
            raise LabelError(f"{path.name} line {n}: {where}: {problem['msg']}") from None
        labels[(label.record_id, label.candidate)] = label
    return list(labels.values())


@dataclass
class LabelsApplied:
    applied: int = 0
    unmatched: list[HumanLabel] = field(default_factory=list)  # match no answer in this run


def apply_human_labels(plan: GradePlan, labels: list[HumanLabel]) -> LabelsApplied:
    """Override grades with your labels and drop their judge calls."""
    pending = {(label.record_id, label.candidate): label for label in labels}
    result = LabelsApplied()
    labelled = set()
    for index, grade in enumerate(plan.grades):
        label = pending.pop((grade.record_id, grade.candidate), None)
        if label is None:
            continue
        grade.outcome = label.outcome
        grade.reason = f"your label: {label.outcome}" + (f" ({label.note})" if label.note else "")
        grade.human = True
        labelled.add(index)
        result.applied += 1
    plan.judge_pairs = [pair for pair in plan.judge_pairs if pair[0] not in labelled]
    result.unmatched = list(pending.values())
    return result
