from routeaudit.grading.judge import (
    combine,
    from_candidate_side,
    judge_messages,
    parse_verdict,
)
from routeaudit.records import Message


def test_parse_verdict_takes_the_last_verdict_line():
    assert parse_verdict("A is shorter.\nVERDICT: B") == "B"
    assert parse_verdict("verdict: tie") == "TIE"
    assert parse_verdict("Answer A quotes 'VERDICT: A'.\n\nVERDICT: B") == "B"
    assert parse_verdict("I think A is better.") is None


def test_verdict_is_read_from_the_candidate_side():
    assert from_candidate_side("A", "A") == "win"
    assert from_candidate_side("A", "B") == "loss"
    assert from_candidate_side("TIE", "B") == "tie"
    assert from_candidate_side(None, "A") is None


def test_disagreement_between_orders_is_a_tie():
    assert combine("win", "win") == "win"
    assert combine("loss", "loss") == "loss"
    assert combine("win", "loss") == "tie"
    assert combine("loss", "tie") == "tie"
    assert combine("win", None) is None


def test_judge_prompt_contains_the_request_and_both_answers():
    request = [Message(role="system", content="Be brief."), Message(role="user", content="Hi")]
    system, user = judge_messages(request, "first answer", "second answer")
    assert system.role == "system"
    assert "<system>\nBe brief.\n</system>" in user.content
    assert "<answer_a>\nfirst answer\n</answer_a>" in user.content
    assert "<answer_b>\nsecond answer\n</answer_b>" in user.content
