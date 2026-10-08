"""Shared test setup."""

import pytest


@pytest.fixture(autouse=True)
def _run_in_a_temporary_folder(tmp_path, monkeypatch):
    """Commands write defaults such as .llm-route-audit/history.jsonl relative to the
    current folder; run each test in its own, so nothing lands in the repo."""
    monkeypatch.chdir(tmp_path)
