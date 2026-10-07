"""Compare an agent's next step with the original: same tools, same arguments?

A step is either tool calls or a text reply. Calls made in one step are compared as a set,
since parallel calls can come back in any order. Arguments use the same loose comparison as
the match_reference check, and arguments named in `ignore` (free text such as notes) are
left out of the comparison.
"""

from typing import Any

from llm_route_audit.grading.checks import CheckResult, same
from llm_route_audit.records import ToolCall

CHECK_NAME = "next_step"


def _arguments(call: ToolCall, ignore: set[str]) -> dict[str, Any]:
    return {k: v for k, v in call.arguments.items() if k not in ignore}


def _names(calls: list[ToolCall]) -> str:
    return ", ".join(c.name for c in calls)


def compare_steps(
    calls: list[ToolCall] | None,
    original: list[ToolCall] | None,
    ignore: list[str] | None = None,
) -> CheckResult:
    """Did the candidate take the same next step as the original model?"""
    calls, original = calls or [], original or []
    if not original and not calls:
        return CheckResult(CHECK_NAME, True, "both replied in text")
    if not calls:
        return CheckResult(
            CHECK_NAME, False, f"replied in text instead of calling {_names(original)}"
        )
    if not original:
        return CheckResult(CHECK_NAME, False, f"called {_names(calls)} instead of replying")

    skip = set(ignore or [])
    unmatched = list(calls)
    missing = []
    for wanted in original:
        match = next(
            (
                c
                for c in unmatched
                if c.name == wanted.name and same(_arguments(c, skip), _arguments(wanted, skip))
            ),
            None,
        )
        if match is None:
            missing.append(wanted)
        else:
            unmatched.remove(match)
    if not missing and not unmatched:
        return CheckResult(CHECK_NAME, True, f"same calls: {_names(original)}")

    for wanted in missing:
        alike = next((c for c in unmatched if c.name == wanted.name), None)
        if alike is not None:
            ours, theirs = _arguments(alike, skip), _arguments(wanted, skip)
            differ = sorted(
                k for k in ours.keys() | theirs.keys() if not same(ours.get(k), theirs.get(k))
            )
            return CheckResult(CHECK_NAME, False, f"{wanted.name}: different {', '.join(differ)}")
    if missing and unmatched:
        detail = f"called {_names(unmatched)} instead of {_names(missing)}"
    elif missing:
        detail = f"did not call {_names(missing)}"
    else:
        detail = f"also called {_names(unmatched)}"
    return CheckResult(CHECK_NAME, False, detail)
