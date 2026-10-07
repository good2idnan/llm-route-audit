from datetime import date

import pytest

from routeaudit.costs import ModelPrice, PriceTable, UnknownModelError, load_prices


def test_bundled_price_table_loads():
    table = load_prices()
    assert "claude-opus-5-5" in table.models


def test_cost_of_plain_request():
    table = load_prices()
    # 1,000 input at $4/M + 500 output at $20/M
    assert table.cost("claude-opus-5-5", input_tokens=1000, output_tokens=500) == pytest.approx(
        0.014
    )


def test_cache_reads_use_the_cache_price():
    table = load_prices()
    assert table.cost("claude-opus-5-5", cache_read_tokens=1_000_000) == pytest.approx(0.20)


def test_missing_cache_price_falls_back_to_input_price():
    table = PriceTable(updated=date(2026, 1, 1), models={"m": ModelPrice(input=3, output=9)})
    assert table.cost("m", cache_read_tokens=1_000_000, cache_write_tokens=1_000_000) == 6


def test_unknown_model_has_a_helpful_error():
    with pytest.raises(UnknownModelError, match="no price for model 'gpt-x'"):
        load_prices().cost("gpt-x", input_tokens=1)
