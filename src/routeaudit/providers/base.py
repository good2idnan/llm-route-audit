from dataclasses import dataclass
from typing import Protocol

from routeaudit.candidates import Candidate
from routeaudit.records import Message


@dataclass
class Completion:
    text: str
    input_tokens: int
    output_tokens: int
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    # "refusal" and "truncated" are kept apart from "ok" so grading can count them.
    status: str = "ok"


class ProviderError(Exception):
    """A failed request.

    `fatal` stops the whole replay (bad key, provider unreachable). `disable` stops the
    candidate that hit it (unknown model, unsupported setting), since every later request
    for that candidate would fail the same way.
    """

    def __init__(self, message: str, *, fatal: bool = False, disable: bool = False) -> None:
        super().__init__(message)
        self.fatal = fatal
        self.disable = disable


class Provider(Protocol):
    def complete(self, candidate: Candidate, messages: list[Message]) -> Completion: ...
