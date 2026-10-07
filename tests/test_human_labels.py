"""Your own pass/fail labels override the checks and the judge."""

import json

import pytest
from typer.testing import CliRunner

from llm_route_audit.cli import app
from llm_route_audit.grading.grade import ORIGINAL, GradingConfig, plan_grades
from llm_route_audit.grading.labels import LabelError, apply_human_labels, load_human_labels
from llm_route_audit.records import LogRecord
from llm_route_audit.replay import ReplayResult


def record(rid: str, task: str, response: str) -> LogRecord:
    return LogRecord(
        id=rid,
        timestamp="2026-10-01T00:00:00Z",
        model="big",
        task_type=task,
        prompt=f"request {rid}",
        response=response,
    )


def answer(rid: str, task: str, text: str) -> ReplayResult:
    return ReplayResult(
        record_id=rid,
        task_type=task,
        model="small",
        effort="low",
        provider="anthropic",
        status="ok",
        response=text,
    )


RECORDS = [record("c1", "classify", '{"category": "billing"}'), record("r1", "reply", "Hi!")]
RESULTS = [answer("c1", "classify", '{"category": "account"}'), answer("r1", "reply", "Hello!")]
CONFIG = GradingConfig.model_validate(
    {
        "judge": {"model": "claude-judge"},
        "tasks": {"classify": {"checks": [{"type": "match_reference", "fields": ["category"]}]}},
    }
)


def test_labels_load_from_csv_and_jsonl(tmp_path):
    csv_file = tmp_path / "labels.csv"
    csv_file.write_text(
        "record_id,candidate,outcome,note\n"
        "c1,small @ low,PASS,account is also right here\n"
        "r1,original,fail,\n"
        "c1,small @ low,fail,changed my mind\n",
        "utf-8",
    )
    labels = load_human_labels(csv_file)
    assert [(lb.record_id, lb.candidate, lb.outcome, lb.note) for lb in labels] == [
        ("c1", "small @ low", "fail", "changed my mind"),  # the later label wins
        ("r1", ORIGINAL, "fail", ""),
    ]
    jsonl_file = tmp_path / "labels.jsonl"
    jsonl_file.write_text(
        json.dumps({"record_id": "r1", "candidate": "small @ low", "outcome": "pass"}) + "\n\n",
        "utf-8",
    )
    assert load_human_labels(jsonl_file)[0].outcome == "pass"


@pytest.mark.parametrize(
    ("content", "error"),
    [
        ("record_id,candidate,outcome\nc1,small,maybe\n", "line 2: outcome"),
        ("record_id,outcome\nc1,pass\n", "line 2: candidate: Field required"),
        ("record_id,candidate,outcome,extra\nc1,small,pass,x\n", "line 2: extra"),
    ],
)
def test_bad_label_files_say_where(tmp_path, content, error):
    path = tmp_path / "labels.csv"
    path.write_text(content, "utf-8")
    with pytest.raises(LabelError, match=error):
        load_human_labels(path)


def test_bad_json_line_says_where(tmp_path):
    path = tmp_path / "labels.jsonl"
    path.write_text('{"record_id": "c1"}\n{oops\n', "utf-8")
    with pytest.raises(LabelError, match="line 2: not valid JSON"):
        load_human_labels(path)


def test_labels_override_checks_and_skip_the_judge(tmp_path):
    plan = plan_grades(RECORDS, RESULTS, CONFIG)
    assert len(plan.judge_pairs) == 1  # the reply goes to the judge
    path = tmp_path / "labels.csv"
    path.write_text(
        "record_id,candidate,outcome,note\n"
        "c1,small @ low,pass,both categories fit\n"
        "r1,small @ low,fail,\n"
        "zz,small @ low,pass,\n",
        "utf-8",
    )
    applied = apply_human_labels(plan, load_human_labels(path))
    grades = {(g.record_id, g.candidate): g for g in plan.grades}
    classify = grades[("c1", "small @ low")]
    assert (classify.outcome, classify.reason, classify.human) == (
        "pass",
        "your label: pass (both categories fit)",
        True,
    )
    assert classify.checks[0].passed is False  # the check result is kept for reference
    assert grades[("r1", "small @ low")].outcome == "fail"
    assert plan.judge_pairs == []
    assert applied.applied == 2
    assert [lb.record_id for lb in applied.unmatched] == ["zz"]
    assert grades[("c1", ORIGINAL)].human is False


def test_grade_command_applies_labels(tmp_path):
    logs = tmp_path / "logs.jsonl"
    logs.write_text("".join(r.model_dump_json() + "\n" for r in RECORDS), "utf-8")
    replay = tmp_path / "replay.jsonl"
    replay.write_text("".join(json.dumps(r.to_dict()) + "\n" for r in RESULTS), "utf-8")
    config = tmp_path / "grading.yaml"
    config.write_text(
        "tasks:\n"
        "  classify: {checks: [{type: match_reference, fields: [category]}]}\n"
        "  reply: {checks: [{type: contains, values: ['Hello']}], judge: false}\n",
        "utf-8",
    )
    labels = tmp_path / "labels.csv"
    labels.write_text("record_id,candidate,outcome\nc1,small @ low,pass\nnope,x,fail\n", "utf-8")
    out = tmp_path / "grades.jsonl"
    result = CliRunner().invoke(
        app,
        [
            "grade",
            str(logs),
            "--replay",
            str(replay),
            "--config",
            str(config),
            "--labels",
            str(labels),
            "--out",
            str(out),
            "--cache",
            str(tmp_path / "cache.sqlite"),
            "--yes",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "Your labels: 1 applied from labels.csv." in result.output
    assert "1 labels match no answer in this run" in result.output and "nope / x" in result.output
    assert "Settled by your labels: 1 (pass 1, fail 0)" in result.output
    assert "1 answers graded by your labels" in result.output
    grades = [json.loads(line) for line in out.read_text("utf-8").splitlines()]
    labelled = next(g for g in grades if g["record_id"] == "c1" and g["candidate"] == "small @ low")
    assert (labelled["outcome"], labelled["human"]) == ("pass", True)
