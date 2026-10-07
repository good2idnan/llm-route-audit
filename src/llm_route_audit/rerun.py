"""Re-run whole agent sessions on a candidate model, one turn after another.

The next-step audit gives every step the logged history. A re-run lets the candidate drive
the whole session instead: its own tool calls, its own path, its own final answer. That
measures whether it actually finishes the job, not just whether each next step matches.

Tool results come from:
- the log (default; free, and nothing runs): when the candidate makes a call the original
  session made (same tool, same arguments), it gets the logged result back. A call the log
  has no answer for ends the session there: it "left the logged path".
- your tools (opt-in): a Python function (`--tool-handler tools.py:handle`) or an MCP server
  (`--mcp "python server.py"`, or a URL). New calls run for real, so point them at test
  accounts or a sandbox.
"""

import asyncio
import concurrent.futures
import importlib
import importlib.util
import json
import shlex
import threading
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Protocol

from llm_route_audit.analyze import usage_of
from llm_route_audit.cache import ResultCache, request_key
from llm_route_audit.candidates import Candidate
from llm_route_audit.costs import PriceTable, UnknownModelError
from llm_route_audit.grading.checks import same
from llm_route_audit.providers.base import Completion, Provider, ProviderError
from llm_route_audit.records import LogRecord, Message, ToolCall, ToolDef
from llm_route_audit.replay import (
    ReplayResult,
    candidate_cost,
    completion_cost,
    worst_case_cost,
)
from llm_route_audit.runner import Job

MAX_TURNS_CAP = 40
TOOL_TIMEOUT_SECONDS = 120


class ToolsUnavailable(RuntimeError):
    """Your tools can't be used: missing install, bad path, or the server didn't start."""


class ToolSource(Protocol):
    def resolve(self, call: ToolCall) -> str | None: ...


def _without(arguments: dict[str, Any], ignore: set[str]) -> dict[str, Any]:
    return {k: v for k, v in arguments.items() if k not in ignore}


# --- where tool results come from ----------------------------------------------------------


class RecordedTools:
    """Results the original session got, handed back when the candidate makes the same call."""

    def __init__(self, steps: list[LogRecord], ignore: list[str] | None = None) -> None:
        self.ignore = set(ignore or [])
        self.entries: list[tuple[ToolCall, str]] = []
        by_id: dict[str, ToolCall] = {}
        pending: list[ToolCall] = []  # calls of the latest assistant turn, for id-less results
        for message in steps[-1].conversation():
            if message.tool_calls:
                pending = list(message.tool_calls)
                by_id.update({c.id: c for c in message.tool_calls if c.id})
            elif message.role == "tool":
                call = by_id.get(message.tool_call_id or "")
                if call is None and pending:
                    call = pending.pop(0)
                if call is not None:
                    self.entries.append((call, message.content))

    def resolve(self, call: ToolCall) -> str | None:
        wanted = _without(call.arguments, self.ignore)
        for logged, content in self.entries:
            if logged.name == call.name and same(_without(logged.arguments, self.ignore), wanted):
                return content
        return None


class FunctionTools:
    """Your own Python function: handle(name, arguments) returns the tool's result."""

    def __init__(self, function: Callable[[str, dict[str, Any]], Any]) -> None:
        self.function = function

    @classmethod
    def load(cls, spec: str) -> "FunctionTools":
        """From "path/to/tools.py:handle" or "package.module:handle"."""
        target, _, name = spec.rpartition(":")
        if not target or not name:
            raise ToolsUnavailable(
                f"--tool-handler needs FILE.py:FUNCTION or MODULE:FUNCTION, got {spec!r}"
            )
        try:
            if target.endswith(".py"):
                path = Path(target).resolve()
                module_spec = importlib.util.spec_from_file_location(path.stem, path)
                if module_spec is None or module_spec.loader is None:
                    raise ToolsUnavailable(f"can't load {target}")
                module = importlib.util.module_from_spec(module_spec)
                module_spec.loader.exec_module(module)
            else:
                module = importlib.import_module(target)
        except (OSError, ImportError) as e:
            raise ToolsUnavailable(f"can't load {target}: {e}") from None
        function = getattr(module, name, None)
        if not callable(function):
            raise ToolsUnavailable(f"{target} has no function named {name!r}")
        return cls(function)

    def resolve(self, call: ToolCall) -> str | None:
        try:
            result = self.function(call.name, dict(call.arguments))
        except Exception as e:  # the tool failed: tell the model, as a real tool would
            return f"ERROR: {type(e).__name__}: {e}"
        if result is None or isinstance(result, str):
            return result
        return json.dumps(result, ensure_ascii=False, default=str)


class MCPTools:
    """Tools served by an MCP server: a command to launch (stdio) or a URL."""

    def __init__(self, server: Any, timeout: float = TOOL_TIMEOUT_SECONDS) -> None:
        try:
            from mcp import Client, StdioServerParameters
        except ImportError:
            raise ToolsUnavailable(
                'MCP support needs: pip install "llm-route-audit[mcp]"'
            ) from None
        if isinstance(server, str) and not server.startswith(("http://", "https://")):
            parts = shlex.split(server)
            if not parts:
                raise ToolsUnavailable("--mcp needs a command or a URL")
            server = StdioServerParameters(command=parts[0], args=parts[1:])
        self.timeout = timeout
        self._client_class = Client
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._loop.run_forever, daemon=True)
        self._thread.start()
        self._ready: concurrent.futures.Future[bool] = concurrent.futures.Future()
        self._queue: asyncio.Queue | None = None
        self._serving = asyncio.run_coroutine_threadsafe(self._serve(server), self._loop)
        try:
            self._ready.result(timeout)
        except Exception as e:
            self.close()
            raise ToolsUnavailable(f"could not connect to the MCP server: {e}") from None

    async def _serve(self, server: Any) -> None:
        """One task opens the connection, answers every call, and closes it: MCP clients
        must be closed by the task that opened them."""
        self._queue = asyncio.Queue()
        try:
            async with self._client_class(server) as client:
                self._ready.set_result(True)
                while (item := await self._queue.get()) is not None:
                    name, arguments, reply = item
                    try:
                        reply.set_result(await client.call_tool(name, arguments))
                    except Exception as e:
                        reply.set_exception(e)
        except Exception as e:
            if not self._ready.done():
                self._ready.set_exception(e)

    def resolve(self, call: ToolCall) -> str | None:
        assert self._queue is not None
        reply: concurrent.futures.Future[Any] = concurrent.futures.Future()
        self._loop.call_soon_threadsafe(self._queue.put_nowait, (call.name, call.arguments, reply))
        try:
            result = reply.result(self.timeout)
        except Exception as e:
            return f"ERROR: {type(e).__name__}: {e}"
        text = "\n".join(c.text for c in result.content if getattr(c, "type", None) == "text")
        if not text and result.structured_content is not None:
            text = json.dumps(result.structured_content, ensure_ascii=False)
        return f"ERROR: {text}" if result.is_error else text

    def close(self) -> None:
        if self._queue is not None and self._loop.is_running():
            self._loop.call_soon_threadsafe(self._queue.put_nowait, None)
            try:
                self._serving.result(10)
            except Exception:
                pass
        self._loop.call_soon_threadsafe(self._loop.stop)
        self._thread.join(10)


# --- running one session --------------------------------------------------------------------


@dataclass
class Budget:
    """Stops new calls once `limit` (USD) could be crossed."""

    limit: float | None = None
    spent: float = 0.0

    def allows(self, worst: float | None) -> bool:
        if self.limit is None:
            return True
        return worst is not None and self.spent + worst <= self.limit


@dataclass
class SessionRun:
    session_id: str
    task: str
    candidate: str
    model: str
    effort: str | None
    provider: str | None
    status: str  # finished | left_path | max_turns | held_back | error
    turns: int = 0
    calls: list[dict[str, Any]] = field(default_factory=list)
    live_calls: int = 0  # calls answered by your tools rather than the log
    final_text: str | None = None
    cost: float | None = 0.0  # what the re-run cost (no prompt cache, as in a replay)
    # What the session would cost served for good, with the original's share of cache hits.
    routed_cost: float | None = 0.0
    original_cost: float | None = None
    original_steps: int = 0
    original_calls: int = 0
    calls_matched: int = 0  # original calls the candidate also made
    detail: str | None = None
    outcome: str | None = None  # the final answer's grade: pass | fail | ungraded
    reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def succeeded(self) -> bool:
        return self.status == "finished" and self.outcome == "pass"


def original_calls(steps: list[LogRecord]) -> list[ToolCall]:
    """Every tool call the original session made, in order."""
    calls = [c for m in steps[-1].conversation() for c in m.tool_calls or []]
    return calls + list(steps[-1].response_tool_calls or [])


def count_matched(made: list[ToolCall], original: list[ToolCall], ignore: set[str]) -> int:
    unused = list(made)
    matched = 0
    for wanted in original:
        target = _without(wanted.arguments, ignore)
        hit = next(
            (
                c
                for c in unused
                if c.name == wanted.name and same(_without(c.arguments, ignore), target)
            ),
            None,
        )
        if hit is not None:
            unused.remove(hit)
            matched += 1
    return matched


def logged_session_cost(prices: PriceTable, steps: list[LogRecord]) -> float | None:
    total = 0.0
    for step in steps:
        usage = usage_of(step)
        try:
            total += prices.cost(
                step.model,
                input_tokens=usage.input_tokens,
                output_tokens=usage.output_tokens,
                cache_read_tokens=usage.cache_read_tokens,
                cache_write_tokens=usage.cache_write_tokens,
            )
        except UnknownModelError:
            return None
    return total


def cache_shares(steps: list[LogRecord]) -> tuple[float, float]:
    """The share of prompt tokens the original session read from and wrote to its cache."""
    read = write = total = 0
    for step in steps:
        usage = usage_of(step)
        read += usage.cache_read_tokens
        write += usage.cache_write_tokens
        total += usage.input_tokens + usage.cache_read_tokens + usage.cache_write_tokens
    return (read / total, write / total) if total else (0.0, 0.0)


def cached_cost(
    prices: PriceTable,
    candidate: Candidate,
    completion: Completion,
    shares: tuple[float, float],
) -> float | None:
    """A call's cost if the candidate cached the way the original session did."""
    read_share, write_share = shares
    if not (read_share or write_share):
        return completion_cost(prices, candidate, completion)
    prompt = completion.input_tokens + completion.cache_read_tokens + completion.cache_write_tokens
    try:
        return prices.cost(
            candidate.model,
            input_tokens=round(prompt * (1 - read_share - write_share)),
            output_tokens=completion.output_tokens,
            cache_read_tokens=round(prompt * read_share),
            cache_write_tokens=round(prompt * write_share),
        )
    except UnknownModelError:
        return completion_cost(prices, candidate, completion)


def session_estimate(
    prices: PriceTable, candidate: Candidate, steps: list[LogRecord]
) -> float | None:
    """About what re-running a session costs: each original step's input and output, priced
    for the candidate (no prompt cache, as in a replay)."""
    total = 0.0
    for step in steps:
        usage = usage_of(step)
        one = candidate_cost(
            prices,
            candidate,
            input_tokens=usage.input_tokens + usage.cache_read_tokens + usage.cache_write_tokens,
            output_tokens=usage.output_tokens,
        )
        if one is None:
            return None
        total += one
    return total


def _complete(
    provider: Provider,
    cache: ResultCache,
    candidate: Candidate,
    messages: list[Message],
    tools: list[ToolDef] | None,
) -> tuple[Completion, bool]:
    key = request_key(candidate, messages, tools)
    hit = cache.get(key)
    if hit is not None:
        return hit[0], True
    if tools:
        completion = provider.complete(candidate, messages, tools=tools)
    else:
        completion = provider.complete(candidate, messages)
    if completion.status == "ok":
        cache.put(key, candidate, completion, None)
    return completion, False


def run_session(
    candidate: Candidate,
    steps: list[LogRecord],
    task: str,
    provider: Provider,
    sources: list[ToolSource],
    cache: ResultCache,
    prices: PriceTable,
    budget: Budget,
    max_turns: int | None = None,
    ignore: list[str] | None = None,
    live_from: int = 1,
) -> SessionRun:
    """Let the candidate run the session from its opening messages. `sources` answer tool
    calls in order; sources from index `live_from` on count as live (your own tools)."""
    first = steps[0]
    tools = next((s.tools for s in steps if s.tools), None)
    messages = list(first.conversation())
    run = SessionRun(
        session_id=first.session_id or first.id,
        task=task,
        candidate=candidate.label,
        model=candidate.model,
        effort=candidate.effort,
        provider=candidate.provider,
        status="max_turns",
        original_cost=logged_session_cost(prices, steps),
        original_steps=len(steps),
    )
    limit = max_turns or min(MAX_TURNS_CAP, 2 * len(steps) + 2)
    shares = cache_shares(steps)
    made: list[ToolCall] = []
    for _ in range(limit):
        worst = worst_case_cost(prices, Job(candidate, messages, tools))
        if not budget.allows(worst):
            run.status, run.detail = "held_back", "stopped to stay under --max-spend"
            break
        try:
            completion, cached = _complete(provider, cache, candidate, messages, tools)
        except ProviderError as e:
            run.status, run.detail = "error", str(e)
            if e.fatal:
                raise
            break
        run.turns += 1
        cost = completion_cost(prices, candidate, completion)
        run.cost = None if cost is None or run.cost is None else run.cost + cost
        routed = cached_cost(prices, candidate, completion, shares)
        run.routed_cost = (
            None if routed is None or run.routed_cost is None else run.routed_cost + routed
        )
        if not cached:
            budget.spent += cost if cost is not None else (worst or 0.0)
        if completion.status != "ok":
            run.status, run.detail = "error", f"answer {completion.status}"
            break
        if not completion.tool_calls:
            run.status, run.final_text = "finished", completion.text
            break
        calls = [
            c.model_copy(update={"id": c.id or f"call_{run.turns}_{i}"})
            for i, c in enumerate(completion.tool_calls)
        ]
        made += calls
        messages.append(Message(role="assistant", content=completion.text, tool_calls=calls))
        for call in calls:
            result, live = None, False
            for index, source in enumerate(sources):
                result = source.resolve(call)
                if result is not None:
                    live = index >= live_from
                    break
            if result is None:
                run.status = "left_path"
                run.detail = f"called {call.name} with arguments the log has no result for"
                break
            run.live_calls += live
            messages.append(
                Message(role="tool", tool_call_id=call.id, name=call.name, content=result)
            )
        if run.status == "left_path":
            break
    original = original_calls(steps)
    run.calls = [c.model_dump() for c in made]
    run.original_calls = len(original)
    run.calls_matched = count_matched(made, original, set(ignore or []))
    return run


# --- grading and showing the results -------------------------------------------------------


def judge_record(run: SessionRun, steps: list[LogRecord]) -> LogRecord | None:
    """The original session as one request (its opening messages) with its final answer, for
    grading the candidate's final answer. None when the original never answered."""
    last = steps[-1]
    if last.response_tool_calls or not last.response:
        return None
    return last.model_copy(
        update={
            "id": run.session_id,
            "messages": steps[0].conversation(),
            "prompt": None,
            "response_tool_calls": None,
            "task_type": run.task,
        }
    )


def grading_inputs(
    runs: list[SessionRun], steps_of: dict[str, list[LogRecord]]
) -> tuple[list[LogRecord], list[ReplayResult]]:
    """Records and answers for plan_grades. Runs that can't be graded get their outcome now."""
    records: dict[str, LogRecord] = {}
    results = []
    for run in runs:
        record = judge_record(run, steps_of[run.session_id])
        if record is None:
            run.outcome, run.reason = "ungraded", "the original session has no final answer"
            continue
        if run.status != "finished":
            run.outcome, run.reason = "fail", f"did not finish ({run.status.replace('_', ' ')})"
            continue
        records[record.id] = record
        results.append(
            ReplayResult(
                record_id=run.session_id,
                task_type=run.task,
                model=run.model,
                effort=run.effort,
                provider=run.provider,
                status="ok",
                response=run.final_text,
            )
        )
    return list(records.values()), results


def apply_grades(runs: list[SessionRun], grades: list[Any]) -> None:
    by_key = {(g.candidate, g.record_id): g for g in grades}
    for run in runs:
        grade = by_key.get((run.candidate, run.session_id))
        if grade is not None and run.outcome is None:
            run.outcome, run.reason = grade.outcome, grade.reason


def write_runs(path: str | Path, runs: list[SessionRun]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for run in runs:
            f.write(json.dumps(run.to_dict(), ensure_ascii=False) + "\n")


STATUS_NOTES = {
    "left_path": "left the logged path (made a call the log has no result for)",
    "max_turns": "hit the turn limit without a final answer",
    "held_back": "were stopped to stay under --max-spend",
    "error": "failed with an API error",
}


def render_reruns(runs: list[SessionRun], source: str, tools_note: str, spent: float) -> str:
    from llm_route_audit.display import INDENT, pct, table, usd

    def share(part: int, whole: int) -> str:
        return "-" if not whole else f"{part}/{whole} ({part / whole:.0%})"

    def money(total: float, count: int) -> str:
        return "-" if not count else f"${total / count:.4f}"

    lines = [
        f"Session re-runs: {source}",
        f"{INDENT}Tool results from: {tools_note}. Spent {usd(spent)}.",
        "",
    ]
    rows = []
    for task in sorted({r.task for r in runs}):
        task_runs = [r for r in runs if r.task == task]
        originals = {r.session_id: r.original_cost for r in task_runs}
        known = [c for c in originals.values() if c is not None]
        original_each = sum(known) / len(known) if known else None
        rows.append(
            [task, "original (as logged)", str(len(originals)), "-", "-", "-", "-",
             "-" if original_each is None else f"${original_each:.4f}"]
        )  # fmt: skip
        for label in sorted({r.candidate for r in task_runs}):
            mine = [r for r in task_runs if r.candidate == label]
            finished = [r for r in mine if r.status == "finished"]
            graded = [r for r in finished if r.outcome in ("pass", "fail")]
            priced = [r for r in mine if r.routed_cost is not None]
            total_cost = sum(r.routed_cost or 0.0 for r in priced)
            cost = money(total_cost, len(priced))
            if priced and original_each:
                cost += f" ({pct(total_cost / len(priced) / original_each)})"
            rows.append(
                [
                    "",
                    label,
                    str(len(mine)),
                    share(len(finished), len(mine)),
                    share(sum(r.outcome == "pass" for r in graded), len(graded)),
                    share(sum(r.succeeded for r in mine), len(mine)),
                    share(sum(r.calls_matched for r in mine), sum(r.original_calls for r in mine)),
                    cost,
                ]
            )
    lines += table(
        [
            "Session type",
            "Option",
            "Sessions",
            "Finished",
            "Answer passed",
            "Succeeded",
            "Same tool calls",
            "Cost per session",
        ],
        rows,
        text_columns=2,
    )
    lines += [
        "",
        "Succeeded = finished with a final answer that passed grading. Same tool calls = calls "
        "the original made that the candidate made too, with the same arguments. Cost per "
        "session assumes the same share of prompt-cache hits as the original, as a session "
        "served for good would get.",
    ]
    notes = []
    for status, note in STATUS_NOTES.items():
        count = sum(r.status == status for r in runs)
        if count:
            example = next((r.detail for r in runs if r.status == status and r.detail), None)
            notes.append(f"{count} sessions {note}" + (f", e.g. {example}" if example else ""))
    live = sum(r.live_calls for r in runs)
    if live:
        notes.append(f"{live} tool calls were answered by your tools.")
    if any(r.status == "left_path" for r in runs) and not live:
        notes.append(
            "To let sessions continue past the logged path, give your tools with "
            "--tool-handler or --mcp (use test accounts or a sandbox)."
        )
    if notes:
        lines += ["", "Notes"] + [f"{INDENT}- {n}" for n in notes]
    return "\n".join(lines)
