import io
import json
import sys
import urllib.error
from collections import defaultdict
from pathlib import Path

import pytest
from typer.testing import CliRunner

from llm_route_audit.cli import app
from llm_route_audit.ingest.jsonl import load_jsonl
from llm_route_audit.labeling import (
    JevLabeler,
    Label,
    LabelerUnavailable,
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


# --- Jev (TypeSafe's hosted decision model) -------------------------------------------------


class FakeResponse(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


def jev_reply(choice="billing", confidence=0.9):
    return {
        "model": "jev-1.13.0",
        "answers": {
            "task": {
                "type": "choice",
                "choice": choice,
                "probabilities": {choice: confidence},
                "confidence": confidence,
            }
        },
        "usage": {"input_tokens": 120, "output_tokens": 4},
    }


def test_jev_client_sends_typed_questions(monkeypatch):
    monkeypatch.setenv("TYPESAFE_API_KEY", "ts-test")
    monkeypatch.delenv("TYPESAFE_API_BASE", raising=False)
    seen = {}

    def fake_urlopen(request, timeout):
        seen["url"] = request.full_url
        seen["auth"] = request.get_header("Authorization")
        seen["body"] = json.loads(request.data)
        return FakeResponse(json.dumps(jev_reply()).encode())

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    labeler = JevLabeler(TASKS)
    label = labeler.label(record("1", "Sort tickets.", "Charged twice"))
    assert (label.task, label.confidence) == ("billing", 0.9)
    assert seen["url"] == "https://api.typesafe.ai/v1/systemone"
    assert seen["auth"] == "Bearer ts-test"
    assert seen["body"]["model"] == "jev-latest"
    assert seen["body"]["questions"]["task"]["type"] == "choice"
    assert "other" in seen["body"]["questions"]["task"]["criteria"]
    assert seen["body"]["state"].startswith("Instructions: Sort tickets.")
    assert 0 < labeler.cost([record("1", "Sort tickets.", "Charged twice")]) < 0.0001


def test_jev_needs_a_key_and_a_working_one(monkeypatch):
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    with pytest.raises(LabelerUnavailable, match="TYPESAFE_API_KEY"):
        JevLabeler(TASKS)

    monkeypatch.setenv("TYPESAFE_API_KEY", "bad")

    def reject(request, timeout):
        body = io.BytesIO(json.dumps({"error": {"message": "invalid key"}}).encode())
        raise urllib.error.HTTPError("u", 401, "unauthorized", {}, body)

    monkeypatch.setattr("urllib.request.urlopen", reject)
    with pytest.raises(LabelerUnavailable, match="invalid key"):
        JevLabeler(TASKS).label(record("1"))


def test_label_command_with_jev(tmp_path, monkeypatch):
    monkeypatch.setenv("TYPESAFE_API_KEY", "ts-test")
    calls = []

    def fake_urlopen(request, timeout):
        state = json.loads(request.data)["state"]
        calls.append(state)
        if "broken" in state:
            body = io.BytesIO(json.dumps({"error": {"message": "bad state"}}).encode())
            raise urllib.error.HTTPError("u", 422, "unprocessable", {}, body)
        choice = "billing" if "charged" in state.lower() else "other"
        return FakeResponse(json.dumps(jev_reply(choice)).encode())

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    logs = tmp_path / "logs.jsonl"
    rows = [
        record("1", user="Charged twice"),
        record("2", user="Hello"),
        record("3", user="broken"),
    ]
    logs.write_text("".join(r.model_dump_json() + "\n" for r in rows), "utf-8")
    tasks = tmp_path / "tasks.yaml"
    tasks.write_text("tasks:\n  billing: payments and refunds\n", "utf-8")
    out = tmp_path / "labelled.jsonl"
    args = ["label", str(logs), "--out", str(out), "--by", "jev", "--tasks", str(tasks)]

    declined = CliRunner().invoke(app, args, input="n\n")
    assert declined.exit_code != 0 and "sent to TypeSafe" in declined.output
    assert calls == []  # nothing sent before you agree

    result = CliRunner().invoke(app, [*args, "--yes"])
    assert result.exit_code == 0, result.output
    assert "Estimated cost" in result.output
    assert "1 requests could not be labelled" in result.output
    labelled = {r.id: r.task_type for r in load_jsonl(out).records}
    assert labelled == {"1": "billing", "2": None, "3": None}
