import json
from datetime import date
from pathlib import Path

import pytest
import yaml
from typer.testing import CliRunner

from llm_route_audit.cache import ResultCache
from llm_route_audit.cli import app
from llm_route_audit.costs import ModelPrice, PriceTable
from llm_route_audit.grading.grade import GradingConfig, plan_grades, run_judges
from llm_route_audit.monitor import assess, grading_inputs, plan_monitor
from llm_route_audit.policy import Policy, load_policy
from llm_route_audit.providers.base import Completion
from llm_route_audit.records import LogRecord
from llm_route_audit.runner import execute

PRICES = PriceTable(
    updated=date(2026, 10, 1),
    models={
        "claude-big": ModelPrice(input=10, output=50),
        "claude-small": ModelPrice(input=1, output=5),
    },
)
POLICY = Policy.model_validate(
    {
        "default": {"model": "claude-big"},
        "routes": {
            "classify": {
                "model": "claude-small",
                "reference": "claude-big",
                "expected_pass_rate": 1.0,
            },
            "contract": {"model": "claude-big"},
        },
    }
)
# Grading: the production answer's "category" must match the reference model's answer.
CONFIG = GradingConfig.model_validate(
    {
        "judge": {"model": "claude-big"},
        "tasks": {
            "classify": {
                "checks": [
                    {"type": "json"},
                    {"type": "match_reference", "fields": ["category"]},
                ]
            }
        },
    }
)


def production(n: int, wrong: int, task: str = "classify") -> list[LogRecord]:
    """n production answers from the cheap model, `wrong` of them with the wrong category."""
    return [
        LogRecord.model_validate(
            {
                "id": f"{task}-{i:02d}",
                "timestamp": "2026-10-05T00:00:00Z",
                "model": "claude-small",
                "task_type": task,
                "prompt": f"ticket {i}",
                "response": json.dumps({"category": "account" if i < wrong else "billing"}),
            }
        )
        for i in range(n)
    ]


class ReferenceModel:
    """The strong model always says billing."""

    calls = 0

    def complete(self, candidate, messages):
        ReferenceModel.calls += 1
        return Completion(
            text=json.dumps({"category": "billing"}), input_tokens=50, output_tokens=10
        )


def run_monitor(records, per_task=20):
    plan = plan_monitor(records, POLICY, per_task=per_task)
    cache = ResultCache(":memory:")
    execution = execute(plan.shadow_jobs(), cache, lambda _: ReferenceModel())
    references, answers, failed = grading_inputs(plan, execution)
    run = run_judges(plan_grades(references, answers, CONFIG), cache, PRICES, lambda _: None)
    return plan, assess(run.grades, plan), failed


def test_only_switched_routes_are_checked():
    records = production(3, 0) + production(2, 0, task="contract") + production(1, 0, task="other")
    plan = plan_monitor(records, POLICY)
    assert {c.task for c in plan.checks} == {"classify"}
    assert "keeps its original model" in plan.not_checked["contract"]
    assert "not in the policy" in plan.not_checked["other"]


def test_sampling_caps_requests_per_task():
    plan = plan_monitor(production(50, 0), POLICY, per_task=12)
    assert len(plan.checks) == 12
    assert plan.checks[0].reference.model == "claude-big"


def test_healthy_route_is_ok():
    _, [health], failed = run_monitor(production(20, 0))
    assert (health.checked, health.passed, health.status) == (20, 20, "OK")
    assert failed == 0


def test_quality_drop_raises_an_alert():
    _, [health], _ = run_monitor(production(20, 8))
    assert health.status == "ALERT"
    assert health.pass_rate == pytest.approx(0.6)


def test_small_dip_is_a_warning_not_an_alert():
    _, [health], _ = run_monitor(production(20, 2))  # 90%, floor 95%
    assert health.status == "WARN"


def test_too_few_checks_waits():
    _, [health], _ = run_monitor(production(4, 0))
    assert health.status == "WAIT"


def test_policy_fields_from_export():
    import test_report

    from llm_route_audit.report import build_report, policy_yaml

    report = build_report(*test_report.scenario(), test_report.PRICES)
    policy = Policy.model_validate(yaml.safe_load(policy_yaml(report)))
    easy = policy.routes["easy"]
    assert easy.switched and easy.reference == "big" and easy.expected_pass_rate == 1.0


def test_monitor_command_exits_2_on_alert(tmp_path, monkeypatch):
    logs = tmp_path / "prod.jsonl"
    logs.write_text("\n".join(r.model_dump_json() for r in production(20, 8)), "utf-8")
    policy = tmp_path / "policy.yaml"
    policy.write_text(
        "version: 1\ndefault: {model: claude-big}\nroutes:\n"
        "  classify: {model: claude-small, reference: claude-big, expected_pass_rate: 1.0}\n",
        "utf-8",
    )
    config = tmp_path / "grading.yaml"
    config.write_text(yaml.safe_dump(CONFIG.model_dump(mode="json")), "utf-8")
    prices = tmp_path / "prices.yaml"
    prices.write_text(
        "updated: 2026-10-01\nmodels:\n  claude-big: {input: 10, output: 50}\n"
        "  claude-small: {input: 1, output: 5}\n",
        "utf-8",
    )
    monkeypatch.setattr("llm_route_audit.cli.get_provider", lambda name: ReferenceModel())
    args = [
        "monitor",
        str(logs),
        "--policy",
        str(policy),
        "--config",
        str(config),
        "--prices",
        str(prices),
        "--cache",
        str(tmp_path / "cache.sqlite"),
        "--out",
        str(tmp_path / "monitor.jsonl"),
        "--yes",
    ]
    result = CliRunner().invoke(app, args)
    assert result.exit_code == 2, result.output
    assert "ALERT: quality dropped on classify" in result.output
    assert Path(tmp_path / "monitor.jsonl").exists()


def test_old_policy_without_audited_rates_explains_itself(tmp_path):
    logs = tmp_path / "prod.jsonl"
    logs.write_text("\n".join(r.model_dump_json() for r in production(3, 0)), "utf-8")
    policy = tmp_path / "policy.yaml"
    policy.write_text(
        "version: 1\ndefault: {model: claude-big}\nroutes:\n  classify: {model: claude-small}\n",
        "utf-8",
    )
    result = CliRunner().invoke(app, ["monitor", str(logs), "--policy", str(policy)])
    assert result.exit_code == 0
    assert "Re-export it" in result.output


def test_load_policy_reads_the_file(tmp_path):
    path = tmp_path / "p.yaml"
    path.write_text(
        'version: 1\ndefault: {model: "a"}\nroutes:\n  t: {model: "b", effort: low}\n', "utf-8"
    )
    policy = load_policy(path)
    assert policy.routes["t"].label == "b @ low"
    assert not policy.routes["t"].switched
