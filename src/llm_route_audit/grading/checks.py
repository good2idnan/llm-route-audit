"""Exact checks: cheap, certain tests of an answer's format and key facts.

Each check compares a candidate's answer with the original (reference) answer from the log.
"""

import json
import re
from dataclasses import asdict, dataclass
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

CODE_FENCE = re.compile(r"^\s*```[\w-]*[ \t]*\n(.*?)\n?[ \t]*```\s*$", re.DOTALL)
NUMBER_TOLERANCE = 0.005
MISSING = object()


@dataclass
class CheckResult:
    name: str
    passed: bool | None  # None: the check could not be applied, e.g. the original isn't JSON
    detail: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def strip_code_fence(text: str) -> str:
    match = CODE_FENCE.match(text)
    return match.group(1) if match else text


def parse_json(text: str, allow_code_fence: bool) -> Any:
    """Parse an answer as JSON. Raises ValueError when it isn't."""
    if allow_code_fence:
        text = strip_code_fence(text)
    return json.loads(text)


def _number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int | float):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value.replace(",", "").strip())
        except ValueError:
            return None
    return None


def same(a: Any, b: Any) -> bool:
    """Loose equality: case and surrounding spaces in text don't matter, numbers match to
    within half a cent, and lists and objects are compared item by item."""
    if isinstance(a, bool) or isinstance(b, bool):
        return a is b
    na, nb = _number(a), _number(b)
    if na is not None and nb is not None:
        return abs(na - nb) <= NUMBER_TOLERANCE
    if isinstance(a, str) and isinstance(b, str):
        return a.strip().casefold() == b.strip().casefold()
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(same(x, y) for x, y in zip(a, b, strict=True))
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(same(a[k], b[k]) for k in a)
    return a == b


def _short(value: Any) -> str:
    text = "missing" if value is MISSING else json.dumps(value, ensure_ascii=False)
    return text if len(text) <= 40 else text[:37] + "..."


class _Check(BaseModel):
    model_config = ConfigDict(extra="forbid")


class JsonCheck(_Check):
    """The answer must be valid JSON. Code fences around it fail unless allowed."""

    type: Literal["json"]
    allow_code_fence: bool = False

    @property
    def name(self) -> str:
        return "json"

    def run(self, answer: str, reference: str) -> CheckResult:
        try:
            parse_json(answer, self.allow_code_fence)
        except ValueError:
            if not self.allow_code_fence and strip_code_fence(answer) != answer:
                return CheckResult(self.name, False, "JSON is wrapped in a code fence")
            return CheckResult(self.name, False, "not valid JSON")
        return CheckResult(self.name, True)


class MatchReferenceCheck(_Check):
    """Listed JSON fields must equal the original answer's (see `same` for how loosely)."""

    type: Literal["match_reference"]
    fields: list[str] = Field(min_length=1)

    @property
    def name(self) -> str:
        return f"match_reference({', '.join(self.fields)})"

    def run(self, answer: str, reference: str) -> CheckResult:
        try:
            expected = parse_json(reference, allow_code_fence=True)
        except ValueError:
            return CheckResult(self.name, None, "the original answer is not JSON")
        try:
            actual = parse_json(answer, allow_code_fence=True)
        except ValueError:
            return CheckResult(self.name, False, "answer is not JSON")
        if not isinstance(expected, dict) or not isinstance(actual, dict):
            return CheckResult(self.name, False, "answer or original is not a JSON object")
        diffs = [
            f"{f}: {_short(actual.get(f, MISSING))} vs {_short(expected.get(f, MISSING))}"
            for f in self.fields
            if not same(actual.get(f, MISSING), expected.get(f, MISSING))
        ]
        return CheckResult(self.name, not diffs, "; ".join(diffs))


class ExactMatchCheck(_Check):
    """The whole answer must equal the original, ignoring case and extra whitespace."""

    type: Literal["exact_match"]

    @property
    def name(self) -> str:
        return "exact_match"

    def run(self, answer: str, reference: str) -> CheckResult:
        def norm(text: str) -> str:
            return " ".join(text.split()).casefold()

        return CheckResult(self.name, norm(answer) == norm(reference))


class ContainsCheck(_Check):
    """Every listed text must appear in the answer."""

    type: Literal["contains"]
    values: list[str] = Field(min_length=1)
    ignore_case: bool = True

    @property
    def name(self) -> str:
        return f"contains({', '.join(self.values)})"

    def run(self, answer: str, reference: str) -> CheckResult:
        haystack = answer.casefold() if self.ignore_case else answer
        missing = [
            v for v in self.values if (v.casefold() if self.ignore_case else v) not in haystack
        ]
        return CheckResult(
            self.name, not missing, f"missing: {', '.join(missing)}" if missing else ""
        )


class RegexCheck(_Check):
    """The answer must match a regular expression somewhere."""

    type: Literal["regex"]
    pattern: str

    @field_validator("pattern")
    @classmethod
    def _compiles(cls, value: str) -> str:
        try:
            re.compile(value)
        except re.error as e:
            raise ValueError(f"invalid regular expression: {e}") from None
        return value

    @property
    def name(self) -> str:
        return f"regex({self.pattern})"

    def run(self, answer: str, reference: str) -> CheckResult:
        found = re.search(self.pattern, answer) is not None
        return CheckResult(self.name, found, "" if found else "pattern not found")


class LengthCheck(_Check):
    """Answer length in characters must fall within the limits."""

    type: Literal["length"]
    min: int | None = Field(default=None, ge=0)
    max: int | None = Field(default=None, ge=0)

    @property
    def name(self) -> str:
        return f"length({self.min or 0}-{self.max if self.max is not None else 'any'})"

    def run(self, answer: str, reference: str) -> CheckResult:
        n = len(answer)
        if self.min is not None and n < self.min:
            return CheckResult(self.name, False, f"{n} characters, fewer than {self.min}")
        if self.max is not None and n > self.max:
            return CheckResult(self.name, False, f"{n} characters, more than {self.max}")
        return CheckResult(self.name, True)


Check = Annotated[
    JsonCheck | MatchReferenceCheck | ExactMatchCheck | ContainsCheck | RegexCheck | LengthCheck,
    Field(discriminator="type"),
]
