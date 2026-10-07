import pytest
from pydantic import ValidationError

from routeaudit.records import LogRecord

BASE = {"id": "r1", "timestamp": "2026-10-01T09:00:00Z", "model": "m", "response": "ok"}


def test_prompt_only_record_becomes_one_user_message():
    record = LogRecord.model_validate({**BASE, "prompt": "hello"})
    [msg] = record.conversation()
    assert (msg.role, msg.content) == ("user", "hello")


def test_messages_record_keeps_order():
    record = LogRecord.model_validate(
        {**BASE, "messages": [{"role": "system", "content": "s"}, {"role": "user", "content": "u"}]}
    )
    assert [m.role for m in record.conversation()] == ["system", "user"]


def test_record_without_any_input_is_rejected():
    with pytest.raises(ValidationError, match="messages' or 'prompt"):
        LogRecord.model_validate(BASE)


def test_negative_token_count_is_rejected():
    with pytest.raises(ValidationError):
        LogRecord.model_validate({**BASE, "prompt": "x", "output_tokens": -1})


def test_timestamp_without_timezone_is_treated_as_utc():
    record = LogRecord.model_validate({**BASE, "timestamp": "2026-10-01T09:00:00", "prompt": "x"})
    assert record.timestamp.utcoffset().total_seconds() == 0


def test_unknown_fields_are_ignored():
    record = LogRecord.model_validate({**BASE, "prompt": "x", "user_id": "u-42"})
    assert not hasattr(record, "user_id")
