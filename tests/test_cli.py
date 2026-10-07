from pathlib import Path

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
