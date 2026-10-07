"""Regenerate docs/images/demo-chart.svg from the demo results in examples/demo/.

The demo is a real replay and grading run (Claude Haiku 4.5 and Sonnet 5.5 via OpenRouter,
judged by Sonnet 5.5) on 5 requests from the synthetic sample log, so it uses --min-samples 1.

Run from the repo root:  uv run python scripts/make_demo_chart.py
"""

from pathlib import Path

from routeaudit.costs import load_prices
from routeaudit.ingest.jsonl import load_jsonl
from routeaudit.replay import load_results
from routeaudit.report import build_report, load_grades
from routeaudit.report_view import render_svg

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "docs" / "images" / "demo-chart.svg"


def main() -> None:
    report = build_report(
        load_jsonl(ROOT / "examples" / "sample_logs.jsonl").records,
        load_results(ROOT / "examples" / "demo" / "replay.jsonl"),
        load_grades(ROOT / "examples" / "demo" / "grades.jsonl"),
        load_prices(),
        min_samples=1,
    )
    svg = render_svg(
        report,
        title="Per-task routing vs. one cheaper model",
        subtitle="Demo: 5 requests from the synthetic sample log, 1 answer per task",
    )
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(svg + "\n", encoding="utf-8", newline="\n")
    print(f"Wrote {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
