import json

from llm_route_audit.ingest.jsonl import load_jsonl


def _line(**overrides):
    record = {
        "id": "r1",
        "timestamp": "2026-10-01T09:00:00Z",
        "model": "m",
        "prompt": "hi",
        "response": "ok",
    }
    record.update(overrides)
    return json.dumps(record)


def test_errors_report_their_line_numbers(tmp_path):
    path = tmp_path / "logs.jsonl"
    path.write_text(
        "\n".join(
            [
                _line(id="a"),
                "{not json",
                "",
                _line(id="b", model=""),
                _line(id="a"),
                _line(id="c"),
            ]
        ),
        encoding="utf-8",
    )
    result = load_jsonl(path)

    assert [r.id for r in result.records] == ["a", "c"]
    assert [e.line for e in result.errors] == [2, 4, 5]
    assert "invalid JSON" in result.errors[0].message
    assert result.errors[1].message.startswith("model:")
    assert "duplicate id 'a'" in result.errors[2].message
    assert not result.ok
