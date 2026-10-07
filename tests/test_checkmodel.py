import json

import test_report
from typer.testing import CliRunner

from llm_route_audit.checkmodel import compare, merge_grades, merge_results, sample_records
from llm_route_audit.cli import app
from llm_route_audit.grading.grade import ORIGINAL, Grade
from llm_route_audit.ingest.jsonl import load_jsonl
from llm_route_audit.providers.base import Completion
from llm_route_audit.replay import ReplayResult, load_results
from llm_route_audit.report import build_report, load_grades


def result(record_id: str, model: str, effort: str | None = None, cost: float = 0.001):
    return ReplayResult(
        record_id=record_id,
        task_type="easy",
        model=model,
        effort=effort,
        provider="anthropic",
        status="ok",
        response="x",
        cost=cost,
    )


def test_merge_results_replaces_only_the_same_model_and_effort():
    old = [result("1", "a"), result("1", "b", "low"), result("2", "a")]
    merged = merge_results(old, [result("1", "b", "low", cost=0.5), result("1", "c")])
    assert len(merged) == 4
    assert [r.cost for r in merged if r.model == "b"] == [0.5]


def test_merge_grades_keeps_earlier_original_grades():
    old = [
        {"record_id": "1", "candidate": ORIGINAL, "outcome": "pass"},
        {"record_id": "1", "candidate": "b", "outcome": "fail"},
    ]
    new = [
        Grade("1", "easy", ORIGINAL, "fail"),
        Grade("1", "easy", "b", "pass"),
        Grade("1", "easy", "c", "pass"),
    ]
    merged = merge_grades(old, new)
    by_key = {(g["candidate"], g["record_id"]): g["outcome"] for g in merged}
    assert by_key == {(ORIGINAL, "1"): "pass", ("b", "1"): "pass", ("c", "1"): "pass"}


def test_sample_records_are_the_ones_replayed_before():
    records, results, _ = test_report.scenario(n=3)
    assert {r.id for r in sample_records(records, results)} == {r.record_id for r in results}


def test_a_cheaper_passing_model_takes_over_tasks():
    records, results, grades = test_report.scenario()
    before = build_report(records, results, grades, test_report.PRICES)
    new_results = [test_report.answer(r, "newcheap", 0.00075) for r in records]  # 5% of original
    new_grades = [Grade(r.id, r.task_type, "newcheap", "pass") for r in records]
    after = build_report(
        records,
        merge_results(results, new_results),
        merge_grades(grades, new_grades),
        test_report.PRICES,
    )
    changes = {c.task: c for c in compare(before, after, "newcheap")}
    assert changes["easy"].before == "cheap" and changes["easy"].after == "newcheap"
    assert changes["hard"].before == "mid" and changes["hard"].after == "newcheap"
    assert after.savings_share > before.savings_share


class Copycat:
    """Answers every request with "same", which matches the logged answers below."""

    calls = 0

    def complete(self, candidate, messages):
        Copycat.calls += 1
        return Completion(text="same", input_tokens=10, output_tokens=2)


def test_check_model_command_updates_the_audit(tmp_path, monkeypatch):
    logs = tmp_path / "logs.jsonl"
    logs.write_text(
        "\n".join(
            json.dumps(
                {
                    "id": f"r{i}",
                    "timestamp": "2026-10-01T00:00:00Z",
                    "model": "claude-big",
                    "task_type": "echo",
                    "prompt": f"q{i}",
                    "response": "same",
                    "input_tokens": 1000,
                    "output_tokens": 100,
                }
            )
            for i in range(3)
        ),
        "utf-8",
    )
    replay = tmp_path / "replay.jsonl"
    replay.write_text(
        "\n".join(
            json.dumps(result(f"r{i}", "claude-old").__dict__ | {"task_type": "echo"})
            for i in range(3)
        ),
        "utf-8",
    )
    grades = tmp_path / "grades.jsonl"
    grades.write_text(
        "\n".join(
            json.dumps({"record_id": f"r{i}", "task_type": "echo", "candidate": c, "outcome": o})
            for i in range(3)
            for c, o in ((ORIGINAL, "pass"), ("claude-old", "fail"))
        ),
        "utf-8",
    )
    config = tmp_path / "grading.yaml"
    config.write_text("tasks:\n  echo:\n    checks:\n      - type: exact_match\n", "utf-8")
    prices = tmp_path / "prices.yaml"
    prices.write_text(
        "updated: 2026-10-01\nmodels:\n  claude-big: {input: 10, output: 50}\n"
        "  claude-old: {input: 1, output: 5}\n  claude-new: {input: 1, output: 5}\n",
        "utf-8",
    )
    monkeypatch.setattr("llm_route_audit.cli.get_provider", lambda name: Copycat())
    args = [
        "check-model",
        str(logs),
        "-m",
        "claude-new",
        "--replay",
        str(replay),
        "--grades",
        str(grades),
        "--config",
        str(config),
        "--prices",
        str(prices),
        "--min-samples",
        "3",
        "--cache",
        str(tmp_path / "cache.sqlite"),
        "--yes",
    ]
    result_ = CliRunner().invoke(app, args)
    assert result_.exit_code == 0, result_.output
    assert "claude-new becomes the best choice for: echo" in result_.output
    assert Copycat.calls == 3

    assert {r.model for r in load_results(replay)} == {"claude-old", "claude-new"}
    merged = load_grades(grades)
    assert sum(g["candidate"] == "claude-new" and g["outcome"] == "pass" for g in merged) == 3
    assert sum(g["candidate"] == ORIGINAL for g in merged) == 3  # not duplicated
    assert load_jsonl(logs).ok
