"""The bundled sample log is used in docs and demos, so keep it valid and priced."""

from collections import Counter
from pathlib import Path

from routeaudit.costs import load_prices
from routeaudit.ingest.jsonl import load_jsonl

SAMPLE = Path(__file__).resolve().parent.parent / "examples" / "sample_logs.jsonl"


def test_sample_log_is_valid():
    result = load_jsonl(SAMPLE)
    assert result.ok, result.errors[:3]
    assert len(result.records) == 200


def test_sample_log_covers_five_task_types():
    tasks = Counter(r.task_type for r in load_jsonl(SAMPLE).records)
    assert set(tasks) == {
        "classify_ticket",
        "extract_invoice",
        "draft_reply",
        "summarize_call",
        "review_contract",
    }


def test_every_sample_model_has_a_price():
    prices = load_prices()
    assert {r.model for r in load_jsonl(SAMPLE).records} <= set(prices.models)
