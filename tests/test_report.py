from datetime import date

import pytest
import yaml

from llm_route_audit.costs import ModelPrice, PriceTable
from llm_route_audit.grading.grade import ORIGINAL
from llm_route_audit.records import LogRecord
from llm_route_audit.replay import ReplayResult
from llm_route_audit.report import (
    OptionStats,
    build_report,
    policy_litellm,
    policy_yaml,
    wilson_interval,
)
from llm_route_audit.report_view import render_html, render_text

PRICES = PriceTable(updated=date(2026, 10, 1), models={"big": ModelPrice(input=10, output=50)})


def record(i: int, task: str) -> LogRecord:
    return LogRecord.model_validate(
        {
            "id": f"{task}-{i:02d}",
            "timestamp": f"2026-10-{1 + i % 20:02d}T00:00:00Z",
            "model": "big",
            "task_type": task,
            "prompt": "q",
            "response": "a",
            "input_tokens": 1000,
            "output_tokens": 100,
        }
    )  # logged cost: 1000*10 + 100*50 = $0.015


def answer(rec: LogRecord, model: str, cost: float) -> ReplayResult:
    return ReplayResult(
        record_id=rec.id,
        task_type=rec.task_type,
        model=model,
        effort=None,
        provider="anthropic",
        status="ok",
        response="x",
        cost=cost,
    )


def grade(rec: LogRecord, candidate: str, passed: bool) -> dict:
    return {
        "record_id": rec.id,
        "task_type": rec.task_type,
        "candidate": candidate,
        "outcome": "pass" if passed else "fail",
    }


def scenario(n: int = 10, cheap_passes_on_hard: int = 4):
    """'easy': the cheap model passes everything. 'hard': it mostly fails."""
    easy = [record(i, "easy") for i in range(n)]
    hard = [record(i, "hard") for i in range(n)]
    records = easy + hard
    results, grades = [], []
    for r in records:
        results.append(answer(r, "cheap", 0.0015))  # 10% of the original's cost
        results.append(answer(r, "mid", 0.0075))  # 50%
        grades.append(grade(r, ORIGINAL, True))
    for r in easy:
        grades.append(grade(r, "cheap", True))
        grades.append(grade(r, "mid", True))
    for i, r in enumerate(hard):
        grades.append(grade(r, "cheap", i < cheap_passes_on_hard))
        grades.append(grade(r, "mid", True))
    return records, results, grades


def by_task(report):
    return {t.task: t for t in report.tasks}


def test_policy_picks_the_cheapest_option_that_keeps_quality():
    report = build_report(*scenario(), PRICES)
    tasks = by_task(report)
    assert tasks["easy"].choice.label == "cheap"
    assert tasks["hard"].choice.label == "mid"
    assert tasks["easy"].choice.cost_ratio == pytest.approx(0.1)


def test_strategies_show_why_per_task_routing_wins():
    report = build_report(*scenario(), PRICES)
    named = {s.name: s for s in report.strategies}
    assert named["Current setup (as logged)"].quality == pytest.approx(1.0)
    assert named["Always cheap"].quality == pytest.approx(0.7)  # (10 + 4) / 20
    assert report.policy.quality == pytest.approx(1.0)
    assert report.policy.cost_ratio == pytest.approx((0.1 + 0.5) / 2)
    assert report.savings_share == pytest.approx(0.7)


def test_small_samples_keep_the_original():
    report = build_report(*scenario(n=3), PRICES)
    for task in report.tasks:
        assert task.choice.label == ORIGINAL
        assert task.reason == "not enough data: 3 graded answers, need 10"
    assert report.savings_share == 0


def test_min_samples_and_target_are_configurable():
    report = build_report(*scenario(n=3, cheap_passes_on_hard=2), PRICES, min_samples=3, target=0.5)
    assert by_task(report)["hard"].choice.label == "cheap"  # 2/3 passes a 50% target


def test_wilson_interval_is_wide_for_small_samples():
    low, high = wilson_interval(1, 1)
    assert low < 0.25 and high == 1.0
    low, high = wilson_interval(95, 100)
    assert 0.88 < low < high < 0.99
    assert wilson_interval(0, 0) is None


def test_option_without_cost_data_has_no_ratio():
    assert OptionStats("x", "x").cost_ratio is None


def test_policy_yaml_is_valid_and_explains_each_route():
    report = build_report(*scenario(), PRICES)
    text = policy_yaml(report)
    data = yaml.safe_load(text)
    assert data["default"] == {"model": "big"}
    assert data["routes"]["easy"]["model"] == "cheap"
    assert "pass 100% on 10, cost 10%" in text


def test_renderers_produce_text_and_self_contained_html():
    report = build_report(*scenario(), PRICES)
    text = render_text(report, html_path="r.html")
    assert "Per-task policy" in text and "HTML report saved to r.html" in text
    page = render_html(report, source="logs.jsonl")
    assert page.startswith("<!doctype html>")
    assert "<svg" in page and "Projected savings" in page
    assert "http://" not in page and "https://" not in page  # nothing loaded from the internet


def test_litellm_export_is_a_valid_proxy_config():
    config = yaml.safe_load(policy_litellm(build_report(*scenario(), PRICES)))
    aliases = {m["model_name"]: m["litellm_params"] for m in config["model_list"]}
    assert set(aliases) == {"route/default", "route/easy", "route/hard"}
    assert aliases["route/easy"]["model"] == "anthropic/cheap"
    assert aliases["route/easy"]["api_key"] == "os.environ/ANTHROPIC_API_KEY"


def test_judge_agreement_is_counted_from_grades():
    records, results, grades = scenario()
    grades[0]["judge_votes"] = ["win", "win"]
    grades[1]["judge_votes"] = ["win", "loss"]
    grades[2]["judge_votes"] = ["tie", None]
    report = build_report(records, results, grades, PRICES)
    assert (report.judged, report.judge_agreed) == (2, 1)
    assert "Judge consistency" in render_text(report)


def test_standalone_svg_has_fixed_colours_and_parses():
    import xml.etree.ElementTree as ET

    from llm_route_audit.report_view import render_svg, short_name

    svg = render_svg(build_report(*scenario(), PRICES), title="Demo", subtitle="tiny sample")
    ET.fromstring(svg)  # valid XML
    assert "var(--" not in svg  # GitHub can't resolve CSS variables inside images
    assert "Per-task policy" in svg
    assert short_name("Always openrouter/anthropic/claude-haiku-4.5") == "Always claude-haiku-4.5"


def test_demo_results_in_the_repo_still_load():
    from pathlib import Path

    from llm_route_audit.costs import load_prices
    from llm_route_audit.ingest.jsonl import load_jsonl
    from llm_route_audit.replay import load_results
    from llm_route_audit.report import load_grades

    root = Path(__file__).resolve().parent.parent
    report = build_report(
        load_jsonl(root / "examples" / "sample_logs.jsonl").records,
        load_results(root / "examples" / "demo" / "replay.jsonl"),
        load_grades(root / "examples" / "demo" / "grades.jsonl"),
        load_prices(),
        min_samples=1,
    )
    assert len(report.tasks) == 5
    assert report.policy.quality == 1.0
