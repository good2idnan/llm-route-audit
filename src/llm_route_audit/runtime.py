"""Use an exported routing policy inside your app.

Let the router make the call:

    from llm_route_audit.runtime import Router

    router = Router.from_file("routing-policy.yaml", log_path="logs/requests.jsonl")
    reply = router.complete("classify_ticket", [{"role": "user", "content": "..."}])
    print(reply.text)

Or keep your own client and only ask which model to use:

    router.choose("classify_ticket")            # Choice(model=..., effort=..., ...)
    client.messages.create(**router.anthropic_args("classify_ticket"), max_tokens=1024, ...)

With `log_path`, every call made through `complete` is saved in llm-route-audit's log format,
so `monitor` can re-check routed traffic, and feedback saved with `record_outcome` can be
read by `outcomes`.
"""

import json
import threading
import time
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from llm_route_audit.candidates import Candidate, Effort
from llm_route_audit.policy import Policy, Route, litellm_model, load_policy
from llm_route_audit.providers import get_provider
from llm_route_audit.providers.base import Completion, Provider, ProviderError
from llm_route_audit.records import LogRecord, Message, ToolDef

DEFAULT_MAX_TOKENS = 16_000


@dataclass(frozen=True)
class Choice:
    """The model the policy picks for one task type."""

    task: str | None
    model: str
    effort: Effort | None
    provider: str | None
    routed: bool  # False: the task has no route, so the policy's default model is used
    reference: str | None = None  # the model a switched route replaced

    def candidate(self, max_tokens: int = DEFAULT_MAX_TOKENS) -> Candidate:
        return Candidate(
            model=self.model,
            effort=self.effort,
            provider=self.provider,  # type: ignore[arg-type]
            max_tokens=max_tokens,
        )


@dataclass
class RoutedReply:
    completion: Completion
    choice: Choice
    model: str  # the model that answered (the reference model after a fallback)
    record_id: str  # use it with record_outcome
    latency_ms: float
    fell_back: bool = False

    @property
    def text(self) -> str:
        return self.completion.text


class Router:
    def __init__(
        self,
        policy: Policy,
        log_path: str | Path | None = None,
        outcomes_path: str | Path | None = None,
        fallback_to_reference: bool = True,
    ) -> None:
        self.policy = policy
        self.log_path = Path(log_path) if log_path else None
        if outcomes_path:
            self.outcomes_path: Path | None = Path(outcomes_path)
        elif self.log_path:
            self.outcomes_path = self.log_path.with_name(self.log_path.stem + ".outcomes.jsonl")
        else:
            self.outcomes_path = None
        self.fallback_to_reference = fallback_to_reference
        self._providers: dict[str, Provider] = {}
        self._lock = threading.Lock()

    @classmethod
    def from_file(cls, path: str | Path, **options: Any) -> "Router":
        return cls(load_policy(path), **options)

    # --- choosing ---------------------------------------------------------------------------

    def choose(self, task: str | None) -> Choice:
        route: Route | None = self.policy.routes.get(task) if task else None
        picked = route or self.policy.default
        return Choice(
            task=task,
            model=picked.model,
            effort=picked.effort,
            provider=picked.provider,
            routed=route is not None,
            reference=route.reference if route else None,
        )

    def anthropic_args(self, task: str | None) -> dict[str, Any]:
        """Arguments for the Anthropic SDK's messages.create."""
        candidate = self.choose(task).candidate()
        if candidate.provider != "anthropic":
            raise ValueError(f"task {task!r} routes to {candidate.label}, not an Anthropic model")
        args: dict[str, Any] = {"model": candidate.api_model}
        if candidate.effort:
            args["output_config"] = {"effort": candidate.effort}
        return args

    def openai_args(self, task: str | None) -> dict[str, Any]:
        """Arguments for the OpenAI SDK's chat.completions.create (OpenAI or OpenRouter)."""
        candidate = self.choose(task).candidate()
        if candidate.provider not in ("openai", "openrouter"):
            raise ValueError(f"task {task!r} routes to {candidate.label}, not an OpenAI-style API")
        args: dict[str, Any] = {"model": candidate.api_model}
        if candidate.effort and candidate.provider == "openai":
            args["reasoning_effort"] = candidate.effort
        elif candidate.effort:
            args["extra_body"] = {"reasoning": {"effort": candidate.effort}}
        return args

    def litellm_args(self, task: str | None) -> dict[str, Any]:
        """Arguments for litellm.completion."""
        choice = self.choose(task)
        args: dict[str, Any] = {"model": litellm_model(choice.model, choice.provider)[0]}
        if choice.effort:
            args["reasoning_effort"] = choice.effort
        return args

    # --- calling ----------------------------------------------------------------------------

    def _provider(self, name: str) -> Provider:
        with self._lock:
            if name not in self._providers:
                self._providers[name] = get_provider(name)
            return self._providers[name]

    def _call(
        self, candidate: Candidate, messages: list[Message], tools: list[ToolDef] | None
    ) -> Completion:
        provider = self._provider(candidate.provider or "")
        if tools:
            return provider.complete(candidate, messages, tools=tools)
        return provider.complete(candidate, messages)

    def complete(
        self,
        task: str | None,
        messages: list[dict[str, Any] | Message],
        tools: list[dict[str, Any] | ToolDef] | None = None,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        session_id: str | None = None,
    ) -> RoutedReply:
        """Send a request to the model the policy picks for `task`. If that model fails and
        the route replaced another model, the request is retried once on that model."""
        history = [m if isinstance(m, Message) else Message.model_validate(m) for m in messages]
        tool_defs = [
            t if isinstance(t, ToolDef) else ToolDef.model_validate(t) for t in tools or []
        ]
        choice = self.choose(task)
        candidate = choice.candidate(max_tokens)
        start = time.perf_counter()
        fell_back = False
        try:
            completion = self._call(candidate, history, tool_defs or None)
        except ProviderError as e:
            if e.fatal or not (self.fallback_to_reference and choice.reference):
                raise
            candidate = Candidate(model=choice.reference, max_tokens=max_tokens)
            completion = self._call(candidate, history, tool_defs or None)
            fell_back = True
        latency = (time.perf_counter() - start) * 1000
        record_id = uuid.uuid4().hex
        if self.log_path:
            self._log(record_id, task, candidate, history, tool_defs, completion, latency,
                      session_id, fell_back)  # fmt: skip
        return RoutedReply(completion, choice, candidate.model, record_id, latency, fell_back)

    # --- logging ----------------------------------------------------------------------------

    def _append(self, path: Path, line: str) -> None:
        with self._lock:
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8", newline="\n") as f:
                f.write(line + "\n")

    def _log(
        self,
        record_id: str,
        task: str | None,
        candidate: Candidate,
        messages: list[Message],
        tools: list[ToolDef],
        completion: Completion,
        latency_ms: float,
        session_id: str | None,
        fell_back: bool,
    ) -> None:
        record = LogRecord(
            id=record_id,
            timestamp=datetime.now(UTC),
            model=candidate.model,
            messages=messages,
            response=completion.text,
            response_tool_calls=completion.tool_calls,
            tools=tools or None,
            session_id=session_id,
            input_tokens=completion.input_tokens,
            output_tokens=completion.output_tokens,
            cache_read_tokens=completion.cache_read_tokens or None,
            cache_write_tokens=completion.cache_write_tokens or None,
            latency_ms=latency_ms,
            task_type=task,
            metadata={
                "source": "llm-route-audit runtime",
                "effort": candidate.effort,
                "fell_back": fell_back,
            },
        )
        assert self.log_path is not None
        self._append(self.log_path, record.model_dump_json(exclude_none=True))

    def record_outcome(self, record_id: str, outcome: str, note: str | None = None) -> None:
        """Save feedback for a logged request, such as "good" or "bad", for `outcomes`."""
        if self.outcomes_path is None:
            raise ValueError("set log_path (or outcomes_path) to record outcomes")
        entry = {
            "record_id": record_id,
            "outcome": outcome,
            "timestamp": datetime.now(UTC).isoformat(),
        }
        if note:
            entry["note"] = note
        self._append(self.outcomes_path, json.dumps(entry, ensure_ascii=False))
