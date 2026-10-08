"""Re-running whole agent sessions: the candidate drives, tools answer from the log or yours."""

import json
from datetime import date

import pytest
import yaml
from typer.testing import CliRunner

from llm_route_audit.cache import ResultCache
from llm_route_audit.candidates import Candidate
from llm_route_audit.cli import app
from llm_route_audit.costs import ModelPrice, PriceTable
from llm_route_audit.grading.grade import GradingConfig, plan_grades
from llm_route_audit.providers.base import Completion
from llm_route_audit.records import LogRecord, Message, ToolCall, ToolDef
from llm_route_audit.rerun import (
    Budget,
    FunctionTools,
    RecordedTools,
    ToolsUnavailable,
    apply_grades,
    grading_inputs,
    render_reruns,
    run_session,
)

PRICES = PriceTable(
    updated=date(2026, 10, 1),
    models={"big": ModelPrice(input=10, output=50), "small": ModelPrice(input=1, output=5)},
)
SMALL = Candidate(model="small", provider="anthropic", max_tokens=100)
TOOLS = [ToolDef(name="find_invoice"), ToolDef(name="issue_refund")]
FIND = ToolCall(id="c1", name="find_invoice", arguments={"id": "INV-1"})
REFUND = ToolCall(
    id="c2", name="issue_refund", arguments={"id": "INV-1", "amount": 40, "reason": "dup"}
)
OPENING = [
    Message(role="system", content="Refund up to $100. Sign as Brightpath Support."),
    Message(role="user", content="Refund INV-1 please."),
]
AFTER_FIND = [
    *OPENING,
    Message(role="assistant", tool_calls=[FIND]),
    Message(role="tool", tool_call_id="c1", content='{"amount": 40}'),
]
AFTER_REFUND = [
    *AFTER_FIND,
    Message(role="assistant", tool_calls=[REFUND]),
    Message(role="tool", tool_call_id="c2", content='{"ok": true}'),
]


def step(n: int, messages, response="", calls=None) -> LogRecord:
    return LogRecord(
        id=f"s1-{n}",
        timestamp=f"2026-10-01T00:00:0{n}Z",
        model="big",
        task_type="refunds",
        session_id="s1",
        messages=messages,
        tools=TOOLS,
        response=response,
        response_tool_calls=calls,
        input_tokens=200,
        output_tokens=20,
    )


SESSION = [
    step(1, OPENING, calls=[FIND]),
    step(2, AFTER_FIND, calls=[REFUND]),
    step(3, AFTER_REFUND, response="Refunded $40. Brightpath Support"),
]


class Scripted:
    """Answers by how many tool results the history holds: a fixed plan of actions."""

    def __init__(self, plan):
        self.plan = plan
        self.calls = 0

    def complete(self, candidate, messages, tools=None):
        self.calls += 1
        done = sum(m.role == "tool" for m in messages)
        action = self.plan[min(done, len(self.plan) - 1)]
        if isinstance(action, str):
            return Completion(action, 100, 10)
        return Completion("", 100, 10, tool_calls=action)


def calls(*items):
    return [ToolCall(name=name, arguments=args) for name, args in items]


SAME_PATH = [
    calls(("find_invoice", {"id": "INV-1"})),
    calls(("issue_refund", {"id": "INV-1", "amount": 40.0, "reason": "dup"})),
    "Done, $40 refunded. Brightpath Support",
]


def rerun(plan, sources=None, budget=None, **options):
    return run_session(
        SMALL,
        SESSION,
        "refunds",
        Scripted(plan),
        sources if sources is not None else [RecordedTools(SESSION)],
        ResultCache(":memory:"),
        PRICES,
        budget or Budget(),
        **options,
    )


def test_recorded_tools_answer_the_calls_the_original_made():
    recorded = RecordedTools(SESSION)
    assert recorded.resolve(ToolCall(name="find_invoice", arguments={"id": "inv-1 "})) == (
        '{"amount": 40}'
    )
    assert recorded.resolve(ToolCall(name="find_invoice", arguments={"id": "INV-2"})) is None
    reworded = ToolCall(name="issue_refund", arguments={"id": "INV-1", "amount": 40, "reason": "x"})
    assert recorded.resolve(reworded) is None
    assert RecordedTools(SESSION, ignore=["reason"]).resolve(reworded) == '{"ok": true}'


def test_recorded_tools_pair_results_without_ids():
    no_ids = [
        *OPENING,
        Message(role="assistant", tool_calls=[ToolCall(name="find_invoice", arguments={"id": 1})]),
        Message(role="tool", content="found"),
    ]
    session = [step(1, no_ids, response="ok")]
    assert (
        RecordedTools(session).resolve(ToolCall(name="find_invoice", arguments={"id": 1}))
        == "found"
    )


def test_a_candidate_on_the_same_path_finishes():
    run = rerun(SAME_PATH)
    assert (run.status, run.turns, run.final_text) == (
        "finished",
        3,
        "Done, $40 refunded. Brightpath Support",
    )
    assert (run.calls_matched, run.original_calls, run.live_calls) == (2, 2, 0)
    assert run.cost == pytest.approx(3 * (100 * 1 + 10 * 5) / 1e6)
    assert run.original_cost == pytest.approx(3 * (200 * 10 + 20 * 50) / 1e6)


def test_leaving_the_logged_path_ends_the_session():
    run = rerun([calls(("issue_refund", {"id": "INV-1", "amount": 400, "reason": "dup"}))])
    assert run.status == "left_path" and "issue_refund" in run.detail
    assert run.calls_matched == 0 and run.turns == 1


def test_your_tools_answer_what_the_log_cannot(tmp_path):
    handler = tmp_path / "my_tools.py"
    handler.write_text(
        "def handle(name, arguments):\n"
        "    if name == 'check_fraud':\n"
        "        return {'risk': 'low'}\n"
        "    raise RuntimeError('no such tool')\n",
        "utf-8",
    )
    tools = FunctionTools.load(f"{handler}:handle")
    plan = [calls(("check_fraud", {"id": "INV-1"})), *SAME_PATH]  # an extra, new first call
    run = rerun(plan, sources=[RecordedTools(SESSION), tools])
    assert run.status == "finished" and run.live_calls == 1
    assert tools.resolve(ToolCall(name="other")) == "ERROR: RuntimeError: no such tool"
    with pytest.raises(ToolsUnavailable, match="no function"):
        FunctionTools.load(f"{handler}:missing")
    with pytest.raises(ToolsUnavailable, match="FILE.py:FUNCTION"):
        FunctionTools.load("just_a_name")


def test_turn_limit_and_spend_limit():
    looping = [calls(("find_invoice", {"id": "INV-1"}))]
    assert rerun(looping, max_turns=4).status == "max_turns"
    held = rerun(SAME_PATH, budget=Budget(limit=0.0))
    assert held.status == "held_back" and held.turns == 0


def test_mcp_server_tools():
    pytest.importorskip("mcp")
    from mcp.server.mcpserver import MCPServer

    from llm_route_audit.rerun import MCPTools

    server = MCPServer("test-tools")

    @server.tool()
    def check_fraud(id: str) -> str:
        """Fraud risk for an invoice."""
        return f"low risk for {id}"

    tools = MCPTools(server)
    try:
        assert tools.resolve(ToolCall(name="check_fraud", arguments={"id": "INV-1"})) == (
            "low risk for INV-1"
        )
        assert tools.resolve(ToolCall(name="nope")).startswith("ERROR:")
    finally:
        tools.close()


def test_final_answers_are_graded_and_shown():
    finished = rerun(SAME_PATH)
    stuck = rerun([calls(("issue_refund", {"id": "INV-9", "amount": 1, "reason": "x"}))])
    config = GradingConfig.model_validate(
        {"tasks": {"refunds": {"checks": [{"type": "contains", "values": ["Brightpath Support"]}]}}}
    )
    records, answers = grading_inputs([finished, stuck], {"s1": SESSION})
    assert records[0].conversation() == OPENING  # graded against the opening request
    apply_grades([finished, stuck], plan_grades(records, answers, config).grades)
    assert finished.outcome == "pass" and finished.succeeded
    assert stuck.outcome == "fail" and stuck.reason == "did not finish (left path)"
    text = render_reruns([finished, stuck], "agent.jsonl", "the log", 0.0)
    assert "Succeeded" in text and "1/2 (50%)" in text
    assert "left the logged path" in text and "--tool-handler or --mcp" in text


def test_rerun_command(tmp_path, monkeypatch):
    provider = Scripted(SAME_PATH)
    monkeypatch.setattr("llm_route_audit.cli.get_provider", lambda name: provider)
    logs = tmp_path / "agent.jsonl"
    logs.write_text("".join(r.model_dump_json() + "\n" for r in SESSION), "utf-8")
    candidates = tmp_path / "candidates.yaml"
    candidates.write_text(yaml.safe_dump({"candidates": [{"model": "claude-haiku-4-5"}]}), "utf-8")
    rules = tmp_path / "grading.yaml"
    rules.write_text(
        "tasks:\n  refunds:\n    checks: [{type: contains, values: [Brightpath Support]}]\n",
        "utf-8",
    )
    out = tmp_path / "reruns.jsonl"
    result = CliRunner().invoke(
        app,
        [
            "rerun", str(logs), "-c", str(candidates), "--config", str(rules),
            "--out", str(out), "--cache", str(tmp_path / "cache.sqlite"), "--yes",
        ],
    )  # fmt: skip
    assert result.exit_code == 0, result.output
    assert "Re-run plan: 1 sessions x 1 candidates" in result.output
    [saved] = [json.loads(line) for line in out.read_text("utf-8").splitlines()]
    assert (saved["status"], saved["outcome"], saved["calls_matched"]) == ("finished", "pass", 2)

    as_json = CliRunner().invoke(
        app,
        ["rerun", str(logs), "-c", str(candidates), "--config", str(rules), "--out", str(out),
         "--cache", str(tmp_path / "cache.sqlite"), "--yes", "--json"],
    )  # fmt: skip
    data = json.loads(as_json.stdout)
    assert data["runs"][0]["status"] == "finished" and data["tools"] == "the log"

    no_sessions = tmp_path / "plain.jsonl"
    no_sessions.write_text(
        SESSION[2].model_copy(update={"session_id": None}).model_dump_json(), "utf-8"
    )
    missing = CliRunner().invoke(app, ["rerun", str(no_sessions), "-c", str(candidates)])
    assert missing.exit_code == 1 and "session_id" in missing.output


def test_session_cost_assumes_the_originals_cache_hits():
    from llm_route_audit.rerun import cache_shares, cached_cost

    cached_session = [
        s.model_copy(update={"input_tokens": 20, "cache_read_tokens": 180}) for s in SESSION
    ]
    assert cache_shares(cached_session) == (0.9, 0.0)
    assert cache_shares(SESSION) == (0.0, 0.0)
    completion = Completion("ok", 1000, 10)
    assert cached_cost(PRICES, SMALL, completion, (0.9, 0.0)) == pytest.approx(
        (100 * 1 + 900 * 1 + 10 * 5) / 1e6  # no cache price listed: reads cost full input
    )
    priced = PriceTable(
        updated=PRICES.updated,
        models={"small": ModelPrice(input=1, output=5, cache_read=0.1)},
    )
    assert cached_cost(priced, SMALL, completion, (0.9, 0.0)) == pytest.approx(
        (100 * 1 + 900 * 0.1 + 10 * 5) / 1e6
    )
    run = run_session(
        SMALL, cached_session, "refunds", Scripted(SAME_PATH), [RecordedTools(SESSION)],
        ResultCache(":memory:"), priced, Budget(),
    )  # fmt: skip
    assert run.routed_cost < run.cost
