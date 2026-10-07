import json

import pytest
from pydantic import TypeAdapter, ValidationError

from llm_route_audit.grading.checks import (
    Check,
    ContainsCheck,
    ExactMatchCheck,
    JsonCheck,
    LengthCheck,
    MatchReferenceCheck,
    RegexCheck,
    same,
    strip_code_fence,
)

REF = json.dumps({"category": "billing", "priority": "high", "total": 663.66})


def test_json_check_rejects_code_fences_unless_allowed():
    fenced = '```json\n{"a": 1}\n```'
    strict = JsonCheck(type="json").run(fenced, REF)
    assert strict.passed is False and "code fence" in strict.detail
    assert JsonCheck(type="json", allow_code_fence=True).run(fenced, REF).passed


def test_strip_code_fence_leaves_plain_text_alone():
    assert strip_code_fence('{"a": 1}') == '{"a": 1}'
    assert strip_code_fence('```\n{"a": 1}\n```') == '{"a": 1}'


def test_match_reference_compares_fields_loosely():
    check = MatchReferenceCheck(type="match_reference", fields=["category", "priority", "total"])
    answer = json.dumps({"category": " Billing ", "priority": "high", "total": "663.66"})
    assert check.run(answer, REF).passed


def test_match_reference_reports_differences():
    check = MatchReferenceCheck(type="match_reference", fields=["priority", "total"])
    result = check.run(json.dumps({"priority": "low"}), REF)
    assert result.passed is False
    assert 'priority: "low" vs "high"' in result.detail
    assert "total: missing vs 663.66" in result.detail


def test_match_reference_reads_fenced_answers():
    check = MatchReferenceCheck(type="match_reference", fields=["category"])
    assert check.run('```json\n{"category": "billing"}\n```', REF).passed


def test_match_reference_is_skipped_when_the_original_is_not_json():
    check = MatchReferenceCheck(type="match_reference", fields=["x"])
    assert check.run('{"x": 1}', "plain text").passed is None


def test_same_handles_numbers_lists_and_booleans():
    assert same(10, 10.004)
    assert not same(10, 10.01)
    assert same([{"a": "X"}], [{"a": "x"}])
    assert not same(True, 1)
    assert not same([1], [1, 2])


def test_text_checks():
    assert (
        ContainsCheck(type="contains", values=["brightpath support"])
        .run("Thanks,\nBrightpath Support", "")
        .passed
    )
    assert not RegexCheck(type="regex", pattern=r"risk: (low|high)").run("no rating", "").passed
    assert ExactMatchCheck(type="exact_match").run("Hello  World", "hello world").passed
    long = LengthCheck(type="length", max=5).run("too long", "")
    assert long.passed is False and "more than 5" in long.detail


def test_checks_load_from_config_by_type():
    checks = TypeAdapter(list[Check]).validate_python(
        [{"type": "json"}, {"type": "contains", "values": ["x"]}]
    )
    assert [c.name for c in checks] == ["json", "contains(x)"]


def test_bad_config_is_rejected():
    with pytest.raises(ValidationError):
        TypeAdapter(Check).validate_python({"type": "regex", "pattern": "("})
    with pytest.raises(ValidationError):
        TypeAdapter(Check).validate_python({"type": "nope"})


TICKET_SCHEMA = {
    "type": "object",
    "required": ["category", "priority"],
    "properties": {
        "category": {"enum": ["billing", "technical", "account"]},
        "priority": {"enum": ["low", "normal", "high", "urgent"]},
    },
    "additionalProperties": False,
}


def schema_check(**fields):
    return TypeAdapter(Check).validate_python({"type": "json_schema", **fields})


def test_json_schema_check_names_the_first_problem():
    check = schema_check(schema=TICKET_SCHEMA)
    assert check.run('{"category": "billing", "priority": "high"}', REF).passed
    wrong = check.run('{"category": "billing", "priority": "critical"}', REF)
    assert wrong.passed is False
    assert wrong.detail.startswith("$.priority: 'critical' is not one of")
    assert "required" in check.run('{"category": "billing"}', REF).detail
    assert check.run("not json", REF).detail == "not valid JSON"
    fenced = '```json\n{"category": "billing", "priority": "low"}\n```'
    assert check.run(fenced, REF).detail == "JSON is wrapped in a code fence"
    assert schema_check(schema=TICKET_SCHEMA, allow_code_fence=True).run(fenced, REF).passed


def test_schema_file_is_found_next_to_the_grading_file(tmp_path):
    from llm_route_audit.grading.grade import load_config

    (tmp_path / "schemas").mkdir()
    (tmp_path / "schemas" / "ticket.json").write_text(json.dumps(TICKET_SCHEMA), "utf-8")
    config = tmp_path / "grading.yaml"
    config.write_text(
        "tasks:\n  classify:\n    checks:\n"
        "      - {type: json_schema, schema_file: schemas/ticket.json}\n",
        "utf-8",
    )
    [check] = load_config(config).tasks["classify"].checks
    assert check.name == "json_schema(ticket.json)"
    assert check.run('{"category": "account", "priority": "low"}', REF).passed


@pytest.mark.parametrize(
    ("fields", "error"),
    [
        ({}, "either 'schema' or 'schema_file'"),
        ({"schema": TICKET_SCHEMA, "schema_file": "x.json"}, "either 'schema' or 'schema_file'"),
        ({"schema": {"type": "not-a-type"}}, "invalid JSON schema"),
        ({"schema_file": "does/not/exist.json"}, "can't read schema file"),
    ],
)
def test_bad_schemas_are_rejected_when_the_config_loads(fields, error):
    with pytest.raises(ValidationError, match=error):
        schema_check(**fields)
