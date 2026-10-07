"""Learning from outcomes: real-world feedback on routed traffic."""

import json

import yaml
from typer.testing import CliRunner

from llm_route_audit.cli import app
from llm_route_audit.outcomes import (
    assess_outcomes,
    load_outcome_file,
    normalize,
    same_model,
    updated_policy_yaml,
)
from llm_route_audit.policy import Policy, load_policy
from llm_route_audit.records import LogRecord

POLICY = Policy.model_validate(
    {
        "default": {"model": "claude-opus-5-5"},
        "routes": {
            "classify": {
                "model": "claude-haiku-4-5",
                "reference": "claude-opus-5-5",
                "expected_pass_rate": 0.98,
            },
            "draft": {
                "model": "openrouter/anthropic/claude-sonnet-5.5",
                "effort": "low",
                "reference": "claude-opus-5-5",
                "expected_pass_rate": 0.95,
            },
            "review": {"model": "claude-opus-5-5"},
        },
    }
)


def logs(task: str, model: str, good: int, bad: int, start: int = 0) -> list[LogRecord]:
    outcomes = ["thumbs_up"] * good + ["thumbs down"] * bad
    return [
        LogRecord(
            id=f"{task}-{model}-{start + i}",
            timestamp="2026-10-01T00:00:00Z",
            model=model,
            task_type=task,
            prompt="x",
            response="y",
            outcome=outcome,
        )
        for i, outcome in enumerate(outcomes)
    ]


def test_outcome_values_are_normalised():
    assert normalize("Thumbs Up") is True and normalize("thumbs-down") is False
    assert normalize("resolved") is True and normalize(True) is True
    assert normalize("meh") is None and normalize(None) is None
    assert normalize("kept", good={"kept"}) is True


def test_model_names_match_with_or_without_provider_prefixes():
    assert same_model("anthropic/claude-sonnet-5.5", "openrouter/anthropic/claude-sonnet-5.5")
    assert not same_model("claude-haiku-4-5", "claude-opus-5-5")


def test_routes_are_judged_against_the_model_they_replaced():
    records = (
        logs("classify", "claude-opus-5-5", 45, 5)  # before the switch: 90% good
        + logs("classify", "claude-haiku-4-5", 25, 25)  # after: 50% good
        + logs("draft", "claude-opus-5-5", 40, 10)
        + logs("draft", "anthropic/claude-sonnet-5.5", 41, 9)  # LiteLLM-style name
        + logs("review", "claude-opus-5-5", 5, 5)
    )
    report = assess_outcomes(records, POLICY)
    by_task = {r.task: r for r in report.routes}
    assert by_task["classify"].status == "REVERT"
    assert (by_task["classify"].routed.good, by_task["classify"].routed.total) == (25, 50)
    assert by_task["draft"].status == "OK"
    assert by_task["draft"].routed.total == 50
    assert report.not_switched == ["review"]
    assert report.with_outcome == 210


def test_small_samples_wait_and_missing_baselines_say_why():
    few = logs("classify", "claude-opus-5-5", 9, 1) + logs("classify", "claude-haiku-4-5", 8, 2)
    assert {r.task: r.status for r in assess_outcomes(few, POLICY).routes}["classify"] == "WAIT"
    only_routed = logs("classify", "claude-haiku-4-5", 30, 30)
    [classify, _] = assess_outcomes(only_routed, POLICY).routes
    assert classify.status == "WAIT" and "keep some traffic" in classify.note


def test_a_clear_but_uncertain_gap_is_watched():
    records = logs("classify", "claude-opus-5-5", 27, 3) + logs(
        "classify", "claude-haiku-4-5", 23, 7
    )
    classify = {r.task: r for r in assess_outcomes(records, POLICY).routes}["classify"]
    assert classify.status == "WATCH"


def test_updated_policy_reverts_only_bad_routes():
    records = logs("classify", "claude-opus-5-5", 45, 5) + logs(
        "classify", "claude-haiku-4-5", 20, 30
    )
    text = updated_policy_yaml(POLICY, assess_outcomes(records, POLICY))
    data = yaml.safe_load(text)
    assert data["routes"]["classify"] == {"model": "claude-opus-5-5"}
    assert data["routes"]["draft"]["model"] == "openrouter/anthropic/claude-sonnet-5.5"
    assert "# reverted from claude-haiku-4-5" in text
    assert Policy.model_validate(data).routes["review"].model == "claude-opus-5-5"


def test_outcome_files_override_the_logs(tmp_path):
    records = logs("classify", "claude-haiku-4-5", 30, 0)
    feedback = tmp_path / "outcomes.csv"
    rows = "\n".join(f"{r.id},bad" for r in records[:20])
    feedback.write_text(f"record_id,outcome\n{rows}\nunknown-id,good\n", "utf-8")
    extra = load_outcome_file(feedback)
    report = assess_outcomes(records + logs("classify", "claude-opus-5-5", 30, 0), POLICY, extra)
    classify = {r.task: r for r in report.routes}["classify"]
    assert (classify.routed.good, classify.routed.total) == (10, 30)
    assert report.unmatched_ids == 1

    jsonl = tmp_path / "outcomes.jsonl"
    jsonl.write_text(json.dumps({"record_id": "a", "outcome": "good"}) + "\n", "utf-8")
    assert load_outcome_file(jsonl) == {"a": "good"}


def test_outcomes_command(tmp_path):
    records = (
        logs("classify", "claude-opus-5-5", 45, 5)
        + logs("classify", "claude-haiku-4-5", 20, 30)
        + logs("draft", "claude-opus-5-5", 1, 0)
    )
    log = tmp_path / "prod.jsonl"
    log.write_text("".join(r.model_dump_json() + "\n" for r in records), "utf-8")
    log.write_text(
        log.read_text("utf-8")
        + LogRecord(
            id="odd",
            timestamp="2026-10-01T00:00:00Z",
            model="claude-haiku-4-5",
            task_type="classify",
            prompt="x",
            outcome="meh",
        ).model_dump_json()
        + "\n",
        "utf-8",
    )
    policy = tmp_path / "policy.yaml"
    policy.write_text(yaml.safe_dump(POLICY.model_dump(exclude_none=True)), "utf-8")
    out = tmp_path / "updated.yaml"
    result = CliRunner().invoke(
        app, ["outcomes", str(log), "--policy", str(policy), "--out", str(out)]
    )
    assert result.exit_code == 2, result.output  # a route should be reverted
    assert "REVERT: classify" in result.output
    assert "'meh' (1)" in result.output and "--good or --bad" in result.output
    assert load_policy(out).routes["classify"].model == "claude-opus-5-5"

    kept = CliRunner().invoke(app, ["outcomes", str(log), "--policy", str(policy), "--good", "MEH"])
    assert "'meh'" not in kept.output  # now counted as good
