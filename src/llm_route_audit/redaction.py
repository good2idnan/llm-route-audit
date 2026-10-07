"""Hide private data in logs before any request leaves your machine.

Pattern-based and offline. Each value is replaced with a numbered placeholder, such as
[EMAIL_1], and the same value gets the same placeholder in the request and in the original
answer, so cheaper models' answers can still be compared fairly with the original.
"""

import re
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator

from llm_route_audit.records import LogRecord, Message


def _digits(text: str) -> str:
    return re.sub(r"\D", "", text)


def luhn_ok(number: str) -> bool:
    """Payment-card checksum, so random long numbers are not taken for cards."""
    digits = [int(d) for d in _digits(number)]
    if not 13 <= len(digits) <= 19:
        return False
    total = 0
    for i, d in enumerate(reversed(digits)):
        if i % 2 == 1:
            d *= 2
            if d > 9:
                d -= 9
        total += d
    return total % 10 == 0


def iban_ok(value: str) -> bool:
    """IBAN mod-97 checksum."""
    compact = re.sub(r"\s", "", value).upper()
    if not 15 <= len(compact) <= 34:
        return False
    rearranged = compact[4:] + compact[:4]
    try:
        return int("".join(str(int(ch, 36)) for ch in rearranged)) % 97 == 1
    except ValueError:
        return False


def phone_ok(value: str) -> bool:
    return 8 <= len(_digits(value)) <= 15


@dataclass
class Rule:
    name: str  # placeholder label, e.g. EMAIL
    pattern: re.Pattern[str]
    check: Callable[[str], bool] | None = None
    group: int = 0  # which part of the match to hide (0 = all of it)


def _rx(pattern: str, flags: int = 0) -> re.Pattern[str]:
    return re.compile(pattern, flags)


# Order matters: secrets and emails first, so later rules don't take pieces of them.
BUILT_IN: dict[str, list[Rule]] = {
    "secret": [
        Rule("SECRET", _rx(r"\b(?:sk|pk|rk)-[A-Za-z0-9_-]{16,}")),
        Rule("SECRET", _rx(r"\bBearer\s+[A-Za-z0-9._~+/=-]{16,}")),
        Rule("SECRET", _rx(r"\bgh[pousr]_[A-Za-z0-9]{20,}\b")),
        Rule("SECRET", _rx(r"\bAKIA[0-9A-Z]{16}\b")),
        Rule("SECRET", _rx(r"\beyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{4,}")),
        Rule(
            "SECRET",
            _rx(
                r"\b(?:password|passwd|pwd|secret|api[_-]?key|access[_-]?token)\s*[:=]\s*(\S+)",
                re.I,
            ),
            group=1,
        ),
    ],
    "email": [Rule("EMAIL", _rx(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"))],
    "url": [
        # Only links that carry data: query strings or a user name in the link.
        Rule("URL", _rx(r"\bhttps?://[^\s)>\]\"']*\?[^\s)>\]\"']+")),
        Rule("URL", _rx(r"\bhttps?://[^\s/@]+@[^\s)>\]\"']+")),
    ],
    "iban": [Rule("IBAN", _rx(r"\b[A-Z]{2}\d{2}(?: ?[A-Z0-9]){11,30}\b"), check=iban_ok)],
    "card": [Rule("CARD", _rx(r"\b(?:\d[ -]?){12,18}\d\b"), check=luhn_ok)],
    "id_number": [
        Rule("ID_NUMBER", _rx(r"\b(?!000|666|9\d\d)\d{3}-(?!00)\d{2}-(?!0000)\d{4}\b")),  # US SSN
        Rule(
            "ID_NUMBER",  # UK National Insurance number
            _rx(r"\b[A-CEGHJ-PR-TW-Z][A-CEGHJ-NPR-TW-Z] ?\d{2} ?\d{2} ?\d{2} ?[A-D]\b"),
        ),
    ],
    "phone": [
        Rule(
            "PHONE", _rx(r"\+\d{1,3}[\s.-]?(?:\(?\d{1,4}\)?[\s.-]?){2,5}\d{2,4}\b"), check=phone_ok
        ),
        Rule("PHONE", _rx(r"\(\d{3}\)\s?\d{3}[\s.-]\d{4}\b"), check=phone_ok),
        Rule("PHONE", _rx(r"\b\d{3}[.-]\d{3}[.-]\d{4}\b"), check=phone_ok),
        Rule("PHONE", _rx(r"\b0\d{2,4}[\s-]\d{3,4}[\s-]?\d{3,4}\b"), check=phone_ok),
    ],
    "ip": [
        Rule("IP", _rx(r"\b(?:25[0-5]|2[0-4]\d|1?\d?\d)(?:\.(?:25[0-5]|2[0-4]\d|1?\d?\d)){3}\b")),
        Rule("IP", _rx(r"\b(?:[0-9a-fA-F]{1,4}:){7}[0-9a-fA-F]{1,4}\b")),
    ],
    "date_of_birth": [
        Rule(
            "DATE_OF_BIRTH",
            _rx(
                r"\b(?:dob|d\.o\.b\.|date of birth|birth ?date|born(?: on)?)\s*[:\-]?\s*"
                r"(\d{1,4}[./-]\d{1,2}[./-]\d{1,4}|\d{1,2}\s+[A-Za-z]+\s+\d{4}|[A-Za-z]+\s+\d{1,2},?\s+\d{4})",
                re.I,
            ),
            group=1,
        )
    ],
}
ALL_TYPES = list(BUILT_IN)


class CustomRule(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(pattern=r"^[A-Za-z][A-Za-z0-9_]*$")
    pattern: str

    @field_validator("pattern")
    @classmethod
    def _compiles(cls, value: str) -> str:
        try:
            re.compile(value)
        except re.error as e:
            raise ValueError(f"invalid regular expression: {e}") from None
        return value


class RedactionConfig(BaseModel):
    """Which built-in types to hide (all by default) and your own patterns."""

    model_config = ConfigDict(extra="forbid")

    types: list[str] = Field(default_factory=lambda: list(ALL_TYPES))
    custom: list[CustomRule] = Field(default_factory=list)

    @field_validator("types")
    @classmethod
    def _known(cls, value: list[str]) -> list[str]:
        unknown = sorted(set(value) - set(ALL_TYPES))
        if unknown:
            raise ValueError(f"unknown types {unknown}; choose from {ALL_TYPES}")
        return value

    def rules(self) -> list[Rule]:
        own = [Rule(c.name.upper(), re.compile(c.pattern)) for c in self.custom]
        return own + [rule for t in ALL_TYPES if t in self.types for rule in BUILT_IN[t]]


def load_redaction_config(path: str | Path | None) -> RedactionConfig:
    if path is None:
        return RedactionConfig()
    return RedactionConfig.model_validate(yaml.safe_load(Path(path).read_text("utf-8")) or {})


@dataclass
class Redactor:
    """Replaces private values with numbered placeholders, consistently within one record."""

    rules: list[Rule]
    mapping: dict[tuple[str, str], str] = field(default_factory=dict)
    counts: Counter = field(default_factory=Counter)

    def _placeholder(self, name: str, value: str) -> str:
        key = (name, value)
        if key not in self.mapping:
            number = sum(1 for n, _ in self.mapping if n == name) + 1
            self.mapping[key] = f"[{name}_{number}]"
            self.counts[name] += 1
        return self.mapping[key]

    def text(self, text: str) -> str:
        for rule in self.rules:

            def replace(match: re.Match[str], rule: Rule = rule) -> str:
                value = match.group(rule.group)
                if value is None or value.startswith("[") or (rule.check and not rule.check(value)):
                    return match.group(0)
                hidden = self._placeholder(rule.name, value)
                if rule.group == 0:
                    return hidden
                start, end = match.span(rule.group)
                whole_start = match.start(0)
                full = match.group(0)
                return full[: start - whole_start] + hidden + full[end - whole_start :]

            text = rule.pattern.sub(replace, text)
        return text


def redact_record(record: LogRecord, rules: list[Rule]) -> tuple[LogRecord, Counter]:
    """The record with private values hidden in every message and in the original answer."""
    redactor = Redactor(rules)
    messages = [
        Message(role=m.role, content=redactor.text(m.content)) for m in record.conversation()
    ]
    response = redactor.text(record.response)
    updated = record.model_copy(update={"messages": messages, "prompt": None, "response": response})
    return updated, redactor.counts
