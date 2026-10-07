import json
import sys
from collections import defaultdict
from pathlib import Path

import pytest
from typer.testing import CliRunner

from llm_route_audit.cli import app
from llm_route_audit.ingest.jsonl import load_jsonl
from llm_route_audit.labeling import (
    Label,
    LayaLabeler,
    LayaUnavailable,
    apply_labels,
    label_by_system_prompt,
    name_from_prompt,
    request_state,
)
from llm_route_audit.records import LogRecord

SAMPLE = Path(__file__).resolve().parent.parent / "examples" / "sample_logs.jsonl"


def record(rid, system=None, user="hello", task=None):
    messages = ([{"role": "system", "content": system}] if system else []) + [
        {"role": "user", "content": user}
    ]
    return LogRecord.model_validate(
        {
            "id": rid,
            "timestamp": "2026-10-01T00:00:00Z",
            "model": "m",
            "task_type": task,
            "messages": messages,
            "response": "ok",
        }
    )


def test_system_prompt_groups_match_the_real_task_types():
    records = load_jsonl(SAMPLE).records
    labels = label_by_system_prompt([r.model_copy(update={"task_type": None}) for r in records])
    groups = defaultdict(set)
    for r in records:
        groups[labels[r.id].task].add(r.task_type)
    assert len(groups) == 5
    assert all(len(real) == 1 for real in groups.values())  # every group is one real task


def test_names_come_from_the_first_meaningful_words():
    assert (
        name_from_prompt("You are the support triage assistant for X.")
        == "support_triage_assistant"
    )
    assert name_from_prompt("Extract the invoice fields.") == "extract_invoice_fields"


def test_same_name_for_different_prompts_gets_a_suffix():
    labels = label_by_system_prompt(
        [
            record("1", "Summarize the call. Be brief."),
            record("2", "Summarize the call. Be detailed."),
            record("3", "Summarize the call. Be detailed."),
        ]
    )
    # Both prompts start "summarize call brief/detailed"; names differ, and the bigger group
    # keeps the plain name.
    assert labels["2"].task == labels["3"].task
    assert labels["1"].task != labels["2"].task


def test_requests_without_system_instructions_stay_unlabelled():
    assert label_by_system_prompt([record("1")])["1"].task is None


def test_apply_labels_keeps_existing_types_unless_overwrite():
    records = [record("1", task="mine"), record("2")]
    labels = {"1": Label("new"), "2": Label("new")}
    assert [r.task_type for r in apply_labels(records, labels)] == ["mine", "new"]
    assert [r.task_type for r in apply_labels(records, labels, overwrite=True)] == ["new", "new"]


class FakeRouter:
    def __init__(self, choice, confidence):
        self.choice, self.confidence = choice, confidence
        self.seen = []

    def predict(self, state, questions):
        self.seen.append((state, questions))
        return {
            "answers": {
                "task": {"type": "choice", "choice": self.choice, "confidence": self.confidence}
            }
        }


TASKS = {"billing": "payments and refunds", "tech": "bugs and outages"}


def test_laya_labeler_asks_one_choice_question_with_an_other_option():
    router = FakeRouter("billing", 0.9)
    label = LayaLabeler(TASKS, router=router).label(record("1", "Sort tickets.", "Charged twice"))
    assert (label.task, label.confidence) == ("billing", 0.9)
    state, questions = router.seen[0]
    assert state.startswith("Instructions: Sort tickets.")
    assert set(questions["task"]["criteria"]) == {"billing", "tech", "other"}


@pytest.mark.parametrize(
    ("choice", "confidence"), [("billing", 0.4), ("other", 0.99), ("nope", 0.9)]
)
def test_laya_labeler_leaves_unsure_answers_unlabelled(choice, confidence):
    label = LayaLabeler(TASKS, router=FakeRouter(choice, confidence)).label(record("1"))
    assert label.task is None


def test_request_state_is_trimmed():
    assert len(request_state(record("1", "s" * 5000, "u" * 5000))) == 1500


def test_missing_laya_install_gives_a_clear_message(monkeypatch):
    monkeypatch.setitem(sys.modules, "laya", None)
    with pytest.raises(LayaUnavailable, match='pip install "llm-route-audit\\[laya\\]"'):
        LayaLabeler(TASKS)


def test_label_command(tmp_path, monkeypatch):
    rows = [json.loads(line) for line in SAMPLE.read_text("utf-8").splitlines()[:20]]
    for row in rows:
        row.pop("task_type")
    logs = tmp_path / "logs.jsonl"
    logs.write_text("\n".join(json.dumps(r) for r in rows), "utf-8")
    out = tmp_path / "labelled.jsonl"
    result = CliRunner().invoke(app, ["label", str(logs), "--out", str(out)])
    assert result.exit_code == 0, result.output
    assert "Labelled 20 requests" in result.output
    assert all(r.task_type for r in load_jsonl(out).records)

    needs_tasks = CliRunner().invoke(app, ["label", str(logs), "--out", str(out), "--by", "laya"])
    assert needs_tasks.exit_code == 1 and "needs --tasks" in needs_tasks.output

    monkeypatch.setitem(sys.modules, "laya", None)
    tasks = tmp_path / "tasks.yaml"
    tasks.write_text("tasks:\n  billing: payments\n", "utf-8")
    missing = CliRunner().invoke(
        app, ["label", str(logs), "--out", str(out), "--by", "laya", "--tasks", str(tasks)]
    )
    assert missing.exit_code == 1 and "pip install" in missing.output
