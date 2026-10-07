"""Pairwise AI judge: is the candidate's answer as good as the original one?

Each pair is judged twice with the answers swapped. Judges tend to favour whichever answer
they read first; asking both ways cancels that out, and when the two verdicts disagree the
result counts as a tie.
"""

import re
from typing import Literal

from llm_route_audit.providers.tooling import render_tool_calls
from llm_route_audit.records import Message, ToolCall, ToolDef

Verdict = Literal["A", "B", "TIE"]
Result = Literal["win", "tie", "loss"]

JUDGE_SYSTEM = "You are a careful, impartial evaluator of answers written by AI assistants."

INSTRUCTIONS = """Two assistants answered the same request. Decide which answer better does \
what the request asks. Consider correctness, following the instructions and any required \
format, completeness, and tone. Do not prefer an answer for being longer. If both are \
equally good, or differ only in ways the requester would not care about, call it a tie.

{request}

<answer_a>
{a}
</answer_a>

<answer_b>
{b}
</answer_b>

Explain briefly, then end with one final line in exactly one of these forms:
VERDICT: A
VERDICT: B
VERDICT: TIE"""

NEXT_STEP_INSTRUCTIONS = """Two assistants were working on the same task, with the same \
tools, and reached the same point. Each chose its next step: calling tools (shown as \
CALL tool_name({{arguments}})) or replying. Decide which next step better moves the task \
toward what the user needs while following the instructions. A different step that is just \
as reasonable is a tie. A step that breaks the instructions, uses wrong values, skips a \
needed check or does something the user did not ask for is worse.

{request}

<next_step_a>
{a}
</next_step_a>

<next_step_b>
{b}
</next_step_b>

Explain briefly, then end with one final line in exactly one of these forms:
VERDICT: A
VERDICT: B
VERDICT: TIE"""

VERDICT_LINE = re.compile(r"VERDICT:\s*(A|B|TIE)\b", re.IGNORECASE)


def render_action(text: str, calls: list[ToolCall] | None) -> str:
    """What an assistant did at one step: its text, then any tool calls."""
    return "\n".join(part for part in (text, render_tool_calls(calls)) if part) or "(nothing)"


def render_request(messages: list[Message], tools: list[ToolDef] | None = None) -> str:
    parts = ["<request>"]
    if tools:
        parts.append("<tools>")
        parts += [f"- {t.name}: {t.description}" if t.description else f"- {t.name}" for t in tools]
        parts.append("</tools>")
    for m in messages:
        if m.role == "tool":
            name = f' name="{m.name}"' if m.name else ""
            parts.append(f"<tool_result{name}>\n{m.content}\n</tool_result>")
        elif m.tool_calls:
            parts.append(f"<{m.role}>\n{render_action(m.content, m.tool_calls)}\n</{m.role}>")
        else:
            parts.append(f"<{m.role}>\n{m.content}\n</{m.role}>")
    parts.append("</request>")
    return "\n".join(parts)


def judge_messages(
    request: list[Message],
    a: str,
    b: str,
    tools: list[ToolDef] | None = None,
    instructions: str = INSTRUCTIONS,
) -> list[Message]:
    prompt = instructions.format(request=render_request(request, tools), a=a, b=b)
    return [Message(role="system", content=JUDGE_SYSTEM), Message(role="user", content=prompt)]


def parse_verdict(text: str) -> Verdict | None:
    """The judge's last VERDICT line, or None if it never gave one."""
    matches = VERDICT_LINE.findall(text)
    return matches[-1].upper() if matches else None  # type: ignore[return-value]


def from_candidate_side(verdict: Verdict | None, candidate_is: Literal["A", "B"]) -> Result | None:
    if verdict is None:
        return None
    if verdict == "TIE":
        return "tie"
    return "win" if verdict == candidate_is else "loss"


def combine(first: Result | None, second: Result | None) -> Result | None:
    """Agreeing verdicts stand; disagreeing ones count as a tie."""
    if first is None or second is None:
        return None
    return first if first == second else "tie"
