"""Give requests a task type, so reports can recommend a model per kind of request.

Two labelers:
- by system prompt: requests that share the same system instructions are almost always the
  same job. Free, instant, no extra install.
- by laya: an open, local decision model (convaiinnovations/laya, Apache-2.0) sorts each
  request into task types you describe. Needs `pip install "llm-route-audit[laya]"`.
"""

import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field

from llm_route_audit.records import LogRecord

STOPWORDS = {
    "a", "am", "an", "and", "are", "as", "be", "for", "i", "is", "of", "please", "the", "this",
    "to", "you", "your", "we", "our", "with", "will", "in", "on", "it",
}  # fmt: skip
NAME_WORDS = 3
STATE_MAX_CHARS = 1500
OTHER = "other"


@dataclass
class Label:
    task: str | None
    confidence: float | None = None


def _system_prompt(record: LogRecord) -> str | None:
    texts = [m.content for m in record.conversation() if m.role == "system"]
    text = " ".join(" ".join(texts).split())
    return text or None


def name_from_prompt(prompt: str) -> str:
    """A short readable name from a system prompt's first meaningful words."""
    words = [w for w in re.findall(r"[a-z0-9]+", prompt.lower()) if w not in STOPWORDS]
    return "_".join(words[:NAME_WORDS]) or "task"


def label_by_system_prompt(records: list[LogRecord]) -> dict[str, Label]:
    """Group requests by identical system instructions and name each group."""
    groups: dict[str, list[str]] = defaultdict(list)
    for record in records:
        prompt = _system_prompt(record)
        if prompt:
            groups[prompt].append(record.id)

    labels = {r.id: Label(None) for r in records}
    used: Counter = Counter()
    for prompt in sorted(groups, key=lambda p: -len(groups[p])):
        name = name_from_prompt(prompt)
        used[name] += 1
        if used[name] > 1:
            name = f"{name}_{used[name]}"
        for record_id in groups[prompt]:
            labels[record_id] = Label(name, 1.0)
    return labels


class TaskList(BaseModel):
    """Task types to sort requests into: {name: one-line description}."""

    model_config = ConfigDict(extra="forbid")

    tasks: dict[str, str] = Field(min_length=1)


def load_tasks(path: str | Path) -> dict[str, str]:
    return TaskList.model_validate(yaml.safe_load(Path(path).read_text(encoding="utf-8"))).tasks


def request_state(record: LogRecord) -> str:
    """What the decision model reads: system instructions first (most telling), then the
    request, trimmed to fit the model's input."""
    system = _system_prompt(record) or ""
    user = " ".join(m.content for m in record.conversation() if m.role == "user")
    text = f"Instructions: {system}\n\nRequest: {user}" if system else f"Request: {user}"
    return text[:STATE_MAX_CHARS]


class LayaUnavailable(RuntimeError):
    pass


class LayaLabeler:
    """Sort requests into task types with the laya decision model, running locally."""

    def __init__(self, tasks: dict[str, str], min_confidence: float = 0.6, router: Any = None):
        self.tasks = tasks
        self.min_confidence = min_confidence
        if router is None:
            try:
                from laya import Router  # type: ignore[import-not-found]
            except ImportError as e:
                raise LayaUnavailable(
                    'laya is not installed. Run: pip install "llm-route-audit[laya]"'
                ) from e
            router = Router()
        self.router = router

    def question(self) -> dict[str, Any]:
        criteria = {**self.tasks, OTHER: "none of the task types above"}
        return {
            "task": {
                "type": "choice",
                "instructions": "Which type of task is this request?",
                "criteria": criteria,
            }
        }

    def label(self, record: LogRecord) -> Label:
        answer = self.router.predict(request_state(record), self.question())["answers"]["task"]
        choice = answer.get("choice")
        confidence = answer.get("confidence")
        if confidence is None:
            confidence = (answer.get("probabilities") or {}).get(choice, 0.0)
        if choice == OTHER or choice not in self.tasks or confidence < self.min_confidence:
            return Label(None, confidence)
        return Label(choice, confidence)


def apply_labels(
    records: list[LogRecord], labels: dict[str, Label], overwrite: bool = False
) -> list[LogRecord]:
    """Records with their new task type. Existing task types stay unless `overwrite`."""
    out = []
    for record in records:
        label = labels.get(record.id)
        if label is None or label.task is None or (record.task_type and not overwrite):
            out.append(record)
        else:
            out.append(record.model_copy(update={"task_type": label.task}))
    return out
