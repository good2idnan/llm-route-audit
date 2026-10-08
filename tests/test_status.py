"""Route history and the status page."""

import json
from pathlib import Path

import yaml
from typer.testing import CliRunner

from llm_route_audit.cli import app
from llm_route_audit.records import LogRecord
from llm_route_audit.status import (
    append_history,
    build_tracks,
    load_history,
    render_status_html,
    render_status_text,
)


def monitor_run(at: str, rate: float, status: str, task: str = "classify") -> dict:
    return {
        "kind": "monitor",
        "at": at,
        "logs": "prod.jsonl",
        "policy": "policy.yaml",
        "routes": [
            {
                "task": task,
                "model": "claude-haiku-4-5",
                "status": status,
                "rate": rate,
                "low": rate - 0.1,
                "high": min(1.0, rate + 0.05),
                "count": 20,
                "target": 0.9,
                "note": "",
            }
        ],
    }


def outcomes_run(at: str, rate: float, original: float, status: str) -> dict:
    return {
        "kind": "outcomes",
        "at": at,
        "logs": "prod.jsonl",
        "policy": "policy.yaml",
        "routes": [
            {
                "task": "draft",
                "model": "claude-sonnet-5-5 @ low",
                "status": status,
                "rate": rate,
                "low": rate - 0.1,
                "high": rate + 0.1,
                "count": 60,
                "target": original,
                "target_count": 60,
                "note": "clearly fewer good outcomes",
            }
        ],
    }


HISTORY = [
    monitor_run("2026-10-01T09:00:00+00:00", 0.97, "OK"),
    monitor_run("2026-10-04T09:00:00+00:00", 0.95, "OK"),
    outcomes_run("2026-10-05T09:00:00+00:00", 0.7, 0.85, "REVERT"),
    monitor_run("2026-10-08T09:00:00+00:00", 0.93, "OK"),
]


def test_history_round_trips_and_skips_bad_lines(tmp_path):
    path = tmp_path / "history.jsonl"
    for run in reversed(HISTORY):
        append_history(path, run)
    with path.open("a", encoding="utf-8") as f:
        f.write("not json\n" + json.dumps({"kind": "other", "at": "x"}) + "\n")
    runs = load_history(path)
    assert [r["at"][:10] for r in runs] == ["2026-10-01", "2026-10-04", "2026-10-05", "2026-10-08"]
    assert load_history(tmp_path / "missing.jsonl") == []


def test_tracks_follow_each_route_and_flag_attention_first():
    tracks = build_tracks(HISTORY)
    assert [(t.task, t.kind) for t in tracks] == [("draft", "outcomes"), ("classify", "monitor")]
    draft, classify = tracks
    assert draft.needs_attention and not classify.needs_attention
    assert len(classify.points) == 3 and classify.latest.rate == 0.93
    assert round(classify.change) == -2  # percentage points since the previous run
    assert draft.change is None


def test_text_view():
    text = render_status_text(build_tracks(HISTORY), "history.jsonl", "status.html")
    assert "4 runs, 2026-10-01 to 2026-10-08" in text
    assert "Needs attention: draft (good outcomes: REVERT)." in text
    assert "-2 pts" in text and "Status page saved to status.html" in text
    assert "No route history yet" in render_status_text([], "history.jsonl")


def test_status_page_is_self_contained_with_one_chart_per_route():
    page = render_status_html(build_tracks(HISTORY), "history.jsonl")
    assert page.count("<svg class='chart'") == 2
    assert "http" not in page and "<script" not in page  # opens offline
    assert "<polygon class='band'" in page  # the 95% range
    assert "audited floor 90%" in page  # monitor target line, labelled
    assert "replaced model 85%" in page and "model it replaced</span>" in page  # legend
    assert "<span class='pill bad'>REVERT</span>" in page  # status as text, not color alone
    assert "<title>2026-10-08 09:00 UTC - OK: 93%" in page  # hover details
    assert "@media (prefers-color-scheme:dark)" in page
    empty = render_status_html([], "history.jsonl")
    assert "No history yet" in empty


def test_one_run_still_draws():
    page = render_status_html(build_tracks(HISTORY[:1]), "history.jsonl")
    assert "<circle class='dot'" in page and "<polyline class='series'" not in page


def test_status_and_outcomes_json(tmp_path):
    history = tmp_path / "history.jsonl"
    for run in HISTORY:
        append_history(history, run)
    out = tmp_path / "status.html"
    result = CliRunner().invoke(
        app, ["status", "--history", str(history), "--html", str(out), "--json"]
    )
    assert result.exit_code == 2
    data = json.loads(result.stdout)
    assert [r["latest_status"] for r in data["routes"]] == ["REVERT", "OK"]
    assert out.exists()

    clean = tmp_path / "clean.jsonl"
    append_history(clean, HISTORY[0])
    ok = CliRunner().invoke(app, ["status", "--history", str(clean), "--html", str(out)])
    assert ok.exit_code == 0 and "No route needs attention." in ok.output


def test_outcomes_command_saves_history_and_prints_json(tmp_path):
    records = [
        LogRecord(
            id=f"{model}-{i}",
            timestamp="2026-10-01T00:00:00Z",
            model=model,
            task_type="classify",
            prompt="x",
            outcome="good" if i < good else "bad",
        )
        for model, good in (("claude-opus-5-5", 45), ("claude-haiku-4-5", 20))
        for i in range(50)
    ]
    logs = tmp_path / "prod.jsonl"
    logs.write_text("".join(r.model_dump_json() + "\n" for r in records), "utf-8")
    policy = tmp_path / "policy.yaml"
    policy.write_text(
        yaml.safe_dump(
            {
                "default": {"model": "claude-opus-5-5"},
                "routes": {
                    "classify": {
                        "model": "claude-haiku-4-5",
                        "reference": "claude-opus-5-5",
                        "expected_pass_rate": 0.98,
                    }
                },
            }
        ),
        "utf-8",
    )
    result = CliRunner().invoke(app, ["outcomes", str(logs), "--policy", str(policy), "--json"])
    assert result.exit_code == 2
    data = json.loads(result.stdout)
    assert data["kind"] == "outcomes" and data["routes"][0]["status"] == "REVERT"
    assert data["routes"][0]["target"] == 0.9 and data["with_outcome"] == 100
    [saved] = load_history(Path(".llm-route-audit/history.jsonl"))  # tests run in tmp_path
    assert saved["routes"][0]["rate"] == 0.4
