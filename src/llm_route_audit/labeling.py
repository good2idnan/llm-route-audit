"""Give requests a task type, so reports can recommend a model per kind of request.

Three labelers:
- by system prompt: requests that share the same system instructions are almost always the
  same job. Free, instant, no extra install.
- by laya: an open, local decision model (convaiinnovations/laya, Apache-2.0) sorts each
  request into task types you describe. Needs `pip install "llm-route-audit[laya]"`.
- by jev: TypeSafe's hosted Jev decision model does the same through its API. Fast and
  cheap, but the request text is sent to TypeSafe. Needs TYPESAFE_API_KEY.

laya and Jev take the same typed questions: a state (the request) and a choice question
whose options are your task types, answered with calibrated probabilities.
"""

import json
import os
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field

from llm_route_audit.analyze import estimate_tokens
from llm_route_audit.providers.http import RETRY_STATUSES, HTTPFailure, post_json
from llm_route_audit.records import LogRecord

STOPWORDS = {
    "a", "am", "an", "and", "are", "as", "be", "for", "i", "is", "of", "please", "the", "this",
    "to", "you", "your", "we", "our", "with", "will", "in", "on", "it",
}  # fmt: skip
NAME_WORDS = 3
STATE_MAX_CHARS = 1500
OTHER = "other"
TYPESAFE_URL = "https://api.typesafe.ai"
JEV_MODEL = "jev-latest"
JEV_PRICE_PER_MILLION = 0.042  # USD per 1M input tokens at launch; output is free


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


class LabelerUnavailable(RuntimeError):
    """The labeler can't run at all: not installed, no key, or the key was rejected."""


class LayaUnavailable(LabelerUnavailable):
    pass


class DecisionLabeler:
    """Sort requests into task types with a decision model that answers typed questions
    (laya or Jev). `router` is anything with predict(state, questions)."""

    def __init__(self, tasks: dict[str, str], router: Any, min_confidence: float = 0.6):
        self.tasks = tasks
        self.min_confidence = min_confidence
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

    def input_tokens(self, records: list[LogRecord]) -> int:
        """Roughly how many tokens labelling these records sends, for a cost estimate."""
        per_question = estimate_tokens(json.dumps(self.question()))
        return sum(estimate_tokens(request_state(r)) + per_question for r in records)


class LayaLabeler(DecisionLabeler):
    """Sort requests into task types with the laya decision model, running locally."""

    def __init__(self, tasks: dict[str, str], min_confidence: float = 0.6, router: Any = None):
        if router is None:
            try:
                from laya import Router  # type: ignore[import-not-found]
            except ImportError as e:
                raise LayaUnavailable(
                    'laya is not installed. Run: pip install "llm-route-audit[laya]"'
                ) from e
            router = Router()
        super().__init__(tasks, router, min_confidence)


class JevClient:
    """TypeSafe's Jev decision model over its API (POST /v1/systemone). Set
    TYPESAFE_API_BASE to go through a proxy, such as LiteLLM's /typesafe pass-through."""

    def __init__(
        self, api_key: str | None = None, base_url: str | None = None, model: str = JEV_MODEL
    ) -> None:
        self.key = api_key or os.environ.get("TYPESAFE_API_KEY")
        if not self.key:
            raise LabelerUnavailable(
                "TYPESAFE_API_KEY is not set. Add it to the .env file in your project folder."
            )
        base = base_url or os.environ.get("TYPESAFE_API_BASE") or TYPESAFE_URL
        self.url = f"{base.rstrip('/')}/v1/systemone"
        self.model = model
        self._sleep = None  # tests can replace the backoff sleep

    def predict(self, state: str, questions: dict[str, Any]) -> dict[str, Any]:
        payload = {"model": self.model, "state": state, "questions": questions}
        extra = {"sleep": self._sleep} if self._sleep else {}
        try:
            return post_json(
                self.url,
                payload,
                {"Authorization": f"Bearer {self.key}"},
                timeout=60,
                retry_statuses=RETRY_STATUSES | {529},  # 529: TypeSafe is overloaded
                **extra,
            )
        except HTTPFailure as e:
            if e.status in (401, 402, 403):
                raise LabelerUnavailable(f"TypeSafe refused the request: {e}") from e
            raise


class JevLabeler(DecisionLabeler):
    """Sort requests into task types with TypeSafe's Jev, through its API."""

    def __init__(self, tasks: dict[str, str], min_confidence: float = 0.6, router: Any = None):
        super().__init__(tasks, router or JevClient(), min_confidence)

    def cost(self, records: list[LogRecord]) -> float:
        """Estimated USD to label these records at the launch price."""
        return self.input_tokens(records) * JEV_PRICE_PER_MILLION / 1e6


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
