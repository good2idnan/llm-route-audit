import json
from pathlib import Path

import pytest
from pydantic import ValidationError
from typer.testing import CliRunner

from llm_route_audit.cli import app
from llm_route_audit.ingest.jsonl import load_jsonl
from llm_route_audit.records import LogRecord
from llm_route_audit.redaction import (
    RedactionConfig,
    Redactor,
    iban_ok,
    luhn_ok,
    redact_record,
)

SAMPLE = Path(__file__).resolve().parent.parent / "examples" / "sample_logs.jsonl"


def hide(text, **config):
    return Redactor(RedactionConfig(**config).rules()).text(text)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("mail amara@lumenharbor.io now", "mail [EMAIL_1] now"),
        ("call +44 7700 900123", "call [PHONE_1]"),
        ("call (555) 010-2030", "call [PHONE_1]"),
        ("call 555-010-2030", "call [PHONE_1]"),
        ("call 07700 900123", "call [PHONE_1]"),
        ("card 4111 1111 1111 1111", "card [CARD_1]"),
        ("iban GB82 WEST 1234 5698 7654 32", "iban [IBAN_1]"),
        ("ssn 123-45-6789", "ssn [ID_NUMBER_1]"),
        ("ni AB 12 34 56 C", "ni [ID_NUMBER_1]"),
        ("host 10.0.0.12", "host [IP_1]"),
        ("go to https://x.io/reset?token=abc", "go to [URL_1]"),
        # Fake keys are built from pieces so no key-shaped text sits in the repository.
        ("key " + "sk" + "-test-" + "x" * 20, "key [SECRET_1]"),
        ("pass" + "word: not-a-real-one", "pass" + "word: [SECRET_1]"),
        ("DOB: 12/04/1990", "DOB: [DATE_OF_BIRTH_1]"),
        ("born on 4 March 1985", "born on [DATE_OF_BIRTH_1]"),
    ],
)
def test_private_values_are_hidden(text, expected):
    assert hide(text) == expected


@pytest.mark.parametrize(
    "text",
    [
        "Invoice INV-2026-31805 dated 2026-09-25",
        "Total $1,091.00 for 400 units",
        "Ticket #82667, version 1.2.3",
        "random 1234 5678 9012 3456 is not a card",
        "VAT number GB123456789",
        "docs at https://docs.example.com/guide",
        "meeting at 10:30 on 2026-10-07",
    ],
)
def test_ordinary_values_are_left_alone(text):
    assert hide(text) == text


def test_checksums():
    assert luhn_ok("4111111111111111") and not luhn_ok("4111111111111112")
    assert iban_ok("GB82WEST12345698765432") and not iban_ok("GB00WEST12345698765432")


def test_same_value_same_placeholder_across_request_and_answer():
    record = LogRecord.model_validate(
        {
            "id": "1",
            "timestamp": "2026-10-01T00:00:00Z",
            "model": "m",
            "messages": [
                {"role": "system", "content": "Extract the customer's email."},
                {"role": "user", "content": "From a@x.io, cc b@y.io, reply to a@x.io"},
            ],
            "response": '{"email": "a@x.io"}',
        }
    )
    cleaned, counts = redact_record(record, RedactionConfig().rules())
    assert cleaned.conversation()[1].content == "From [EMAIL_1], cc [EMAIL_2], reply to [EMAIL_1]"
    assert cleaned.response == '{"email": "[EMAIL_1]"}'
    assert counts == {"EMAIL": 2}


def test_choosing_types_and_adding_your_own():
    text = "Customer CUST-004211 wrote from a@x.io"
    assert hide(text, types=["phone"]) == text
    custom = hide(text, types=[], custom=[{"name": "customer_id", "pattern": r"CUST-\d{6}"}])
    assert custom == "Customer [CUSTOMER_ID_1] wrote from a@x.io"


def test_bad_config_is_rejected():
    with pytest.raises(ValidationError, match="unknown types"):
        RedactionConfig(types=["emails"])
    with pytest.raises(ValidationError, match="invalid regular expression"):
        RedactionConfig(custom=[{"name": "x", "pattern": "("}])


def test_redact_command(tmp_path):
    out = tmp_path / "safe.jsonl"
    result = CliRunner().invoke(app, ["redact", str(SAMPLE), "--out", str(out)])
    assert result.exit_code == 0, result.output
    assert "Checked 200 requests; 2 contained private data." in result.output
    assert "Hidden: 2 email" in result.output
    assert "@" not in result.output.split("Hidden")[0]  # values are never printed
    cleaned = load_jsonl(out)
    assert cleaned.ok and len(cleaned.records) == 200
    assert "[EMAIL_1]" in json.dumps([r.model_dump(mode="json") for r in cleaned.records])
