import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from routeaudit import __version__
from routeaudit.cli import app

runner = CliRunner()
SAMPLE = Path(__file__).resolve().parent.parent / "examples" / "sample_logs.jsonl"


def test_version():
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0
    assert __version__ in result.output


def test_validate_sample_log():
    result = runner.invoke(app, ["validate", str(SAMPLE)])
    assert result.exit_code == 0
    assert "OK: 200 records" in result.output


def test_validate_reports_bad_lines(tmp_path):
    bad = tmp_path / "bad.jsonl"
    bad.write_text("{oops\n", encoding="utf-8")
    result = runner.invoke(app, ["validate", str(bad)])
    assert result.exit_code == 1
    assert "line 1: invalid JSON" in result.output


def test_analyze_sample_log():
    result = runner.invoke(app, ["analyze", str(SAMPLE)])
    assert result.exit_code == 0
    assert "200 over" in result.output
    assert "review_contract" in result.output
    assert "Prices as of  2026-09-25" in result.output


def test_analyze_json_output():
    result = runner.invoke(app, ["analyze", str(SAMPLE), "--json"])
    assert result.exit_code == 0
    data = json.loads(result.output)
    assert data["requests"] == 200
    assert len(data["by_task"]) == 5
    assert sum(t["requests"] for t in data["by_task"]) == 200


def test_analyze_with_custom_prices(tmp_path):
    prices = tmp_path / "prices.yaml"
    prices.write_text(
        "updated: 2026-10-01\nmodels:\n  claude-opus-5-5: {input: 0, output: 0}\n",
        encoding="utf-8",
    )
    result = runner.invoke(app, ["analyze", str(SAMPLE), "--prices", str(prices), "--json"])
    assert result.exit_code == 0
    assert json.loads(result.output)["total"]["cost"] == 0


def test_analyze_reports_unpriced_models(tmp_path):
    logs = tmp_path / "logs.jsonl"
    logs.write_text(
        json.dumps(
            {
                "id": "a",
                "timestamp": "2026-10-01T09:00:00Z",
                "model": "some-other-model",
                "prompt": "hi",
                "response": "hello",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    result = runner.invoke(app, ["analyze", str(logs)])
    assert result.exit_code == 0
    assert "No price for: some-other-model (1)" in result.output
    assert "estimated from text length" in result.output


def test_analyze_rejects_bad_prices_file(tmp_path):
    prices = tmp_path / "prices.yaml"
    prices.write_text("models: [oops\n", encoding="utf-8")
    result = runner.invoke(app, ["analyze", str(SAMPLE), "--prices", str(prices)])
    assert result.exit_code == 1
    assert "Could not read prices file" in result.output


CANDIDATES = SAMPLE.parent / "candidates.yaml"


class EchoProvider:
    def complete(self, candidate, messages):
        from routeaudit.providers.base import Completion

        return Completion(text="ok", input_tokens=100, output_tokens=10)


def _replay_args(tmp_path, *extra):
    return [
        "replay",
        str(SAMPLE),
        "-c",
        str(CANDIDATES),
        "--sample",
        "10",
        "--cache",
        str(tmp_path / "cache.sqlite"),
        "--out",
        str(tmp_path / "replay.jsonl"),
        *extra,
    ]


def test_replay_dry_run_sends_nothing(tmp_path, monkeypatch):
    monkeypatch.setattr("routeaudit.cli.get_provider", lambda name: pytest.fail("no calls"))
    result = runner.invoke(app, _replay_args(tmp_path, "--dry-run"))
    assert result.exit_code == 0
    assert "10 requests x 3 candidates = 30 answers" in result.output
    assert "Dry run: nothing was sent." in result.output
    assert not (tmp_path / "replay.jsonl").exists()


def test_replay_asks_before_spending(tmp_path, monkeypatch):
    monkeypatch.setattr("routeaudit.cli.get_provider", lambda name: pytest.fail("no calls"))
    result = runner.invoke(app, _replay_args(tmp_path), input="n\n")
    assert result.exit_code == 1
    assert "Spend about $" in result.output
    assert "Cancelled. Nothing was sent." in result.output


def test_replay_respects_budget(tmp_path, monkeypatch):
    monkeypatch.setattr("routeaudit.cli.get_provider", lambda name: pytest.fail("no calls"))
    result = runner.invoke(app, _replay_args(tmp_path, "--budget", "0.001", "--yes"))
    assert result.exit_code == 1
    assert "over your budget" in result.output


def test_replay_writes_answers(tmp_path, monkeypatch):
    monkeypatch.setattr("routeaudit.cli.get_provider", lambda name: EchoProvider())
    result = runner.invoke(app, _replay_args(tmp_path, "--yes"))
    assert result.exit_code == 0, result.output
    assert "Replay finished: 30 new answers, 0 from cache." in result.output
    rows = [
        json.loads(line) for line in (tmp_path / "replay.jsonl").read_text("utf-8").splitlines()
    ]
    assert len(rows) == 30
    assert {r["status"] for r in rows} == {"ok"}

    again = runner.invoke(app, _replay_args(tmp_path))
    assert again.exit_code == 0
    assert "0 new answers, 30 from cache" in again.output
