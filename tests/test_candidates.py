from pathlib import Path

import pytest
from pydantic import ValidationError

from llm_route_audit.candidates import Candidate, CandidateFile, load_candidates

EXAMPLE = Path(__file__).resolve().parent.parent / "examples" / "candidates.yaml"


def test_provider_is_inferred_from_the_model_name():
    assert Candidate(model="claude-haiku-4-5").provider == "anthropic"
    local = Candidate(model="ollama/llama3.2")
    assert (local.provider, local.api_model) == ("ollama", "llama3.2")


def test_unknown_model_needs_an_explicit_provider():
    with pytest.raises(ValidationError, match="Set provider"):
        Candidate(model="some-model")


def test_effort_is_rejected_for_ollama():
    with pytest.raises(ValidationError, match="effort is not supported"):
        Candidate(model="ollama/llama3.2", effort="low")


def test_label_includes_effort():
    assert Candidate(model="claude-opus-5-5", effort="low").label == "claude-opus-5-5 @ low"


def test_duplicate_candidates_are_rejected():
    with pytest.raises(ValidationError, match="listed more than once"):
        CandidateFile.model_validate(
            {"candidates": [{"model": "claude-haiku-4-5"}, {"model": "claude-haiku-4-5"}]}
        )


def test_bundled_example_file_loads():
    candidates = load_candidates(EXAMPLE)
    assert [c.label for c in candidates] == [
        "claude-haiku-4-5",
        "claude-sonnet-5-5 @ low",
        "claude-opus-5-5 @ low",
    ]
