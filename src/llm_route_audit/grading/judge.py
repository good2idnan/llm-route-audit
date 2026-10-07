"""Pairwise AI judge: is the candidate's answer as good as the original one?

Each pair is judged twice with the answers swapped. Judges tend to favour whichever answer
they read first; asking both ways cancels that out, and when the two verdicts disagree the
result counts as a tie.
"""

import re
from typing import Literal

from llm_route_audit.records import Message

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

VERDICT_LINE = re.compile(r"VERDICT:\s*(A|B|TIE)\b", re.IGNORECASE)


def render_request(messages: list[Message]) -> str:
    parts = ["<request>"]
    parts += [f"<{m.role}>\n{m.content}\n</{m.role}>" for m in messages]
    parts.append("</request>")
    return "\n".join(parts)


def judge_messages(request: list[Message], a: str, b: str) -> list[Message]:
    prompt = INSTRUCTIONS.format(request=render_request(request), a=a, b=b)
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
