"""Half-price batch mode: send many requests at once and collect the answers later.

Anthropic and OpenRouter batch APIs charge about half the normal price and answer within 24
hours (usually much sooner). `run_batches` submits the requests a run still needs, records
the batch ids in a small state file, waits a while, and writes finished answers into the
result cache. The normal replay or grade flow then finds them there. If batches are still
running when the wait ends, running the same command again later collects them; nothing is
sent or paid for twice.
"""

import json
import os
import time
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

from pydantic import BaseModel, Field

from llm_route_audit.analyze import estimate_tokens
from llm_route_audit.cache import ResultCache, request_key
from llm_route_audit.candidates import Candidate
from llm_route_audit.costs import PriceTable
from llm_route_audit.providers.base import Completion, ProviderError
from llm_route_audit.records import Message
from llm_route_audit.replay import candidate_cost, worst_case_cost
from llm_route_audit.runner import Job

BATCH_DISCOUNT = 0.5
BATCH_PROVIDERS = frozenset({"anthropic", "openrouter"})
Items = list[tuple[str, list[Message]]]  # (custom id = cache key, request messages)


class PendingBatch(BaseModel):
    id: str
    candidate: Candidate
    keys: list[str]
    input_estimates: dict[str, int] = Field(default_factory=dict)
    submitted_at: str


@dataclass
class BatchCheck:
    done: bool
    results: dict[str, Completion | str] = field(default_factory=dict)  # completion or error
    progress: str = ""


class BatchClient(Protocol):
    def submit(self, candidate: Candidate, items: Items) -> str: ...

    def check(self, batch: PendingBatch) -> BatchCheck: ...


def load_state(path: Path) -> list[PendingBatch]:
    if not path.exists():
        return []
    return [PendingBatch.model_validate(b) for b in json.loads(path.read_text("utf-8"))]


def save_state(path: Path, batches: list[PendingBatch]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps([b.model_dump(mode="json") for b in batches], indent=2), "utf-8")


def batch_cost(prices: PriceTable, candidate: Candidate, completion: Completion) -> float | None:
    """The provider's reported cost, or the price-table cost at the batch discount."""
    if completion.cost is not None:
        return completion.cost
    full = candidate_cost(
        prices,
        candidate,
        input_tokens=completion.input_tokens,
        output_tokens=completion.output_tokens,
        cache_read_tokens=completion.cache_read_tokens,
        cache_write_tokens=completion.cache_write_tokens,
    )
    return None if full is None else full * BATCH_DISCOUNT


@dataclass
class BatchProgress:
    submitted: int = 0
    collected: int = 0
    spent: float = 0.0
    pending: int = 0  # answers still being worked on by the provider
    live: int = 0  # requests for providers without a batch API; they run normally
    held_back: int = 0  # not submitted, to stay under max_spend
    failed: Counter = field(default_factory=Counter)


def run_batches(
    jobs: list[Job],
    cache: ResultCache,
    prices: PriceTable,
    state_path: Path,
    client_for: Callable[[str], BatchClient],
    max_spend: float | None = None,
    wait_seconds: float = 3600,
    poll_seconds: float = 30,
    on_status: Callable[[str], None] | None = None,
    sleep: Callable[[float], None] = time.sleep,
    clock: Callable[[], float] = time.monotonic,
) -> BatchProgress:
    """Submit what isn't cached or already submitted, then collect finished batches."""
    progress = BatchProgress()
    state = load_state(state_path)
    in_flight = {key for batch in state for key in batch.keys}

    groups: dict[str, tuple[Candidate, Items]] = {}
    for job in jobs:
        key = request_key(job.candidate, job.messages)
        if key in in_flight or cache.get(key) is not None:
            continue
        if job.candidate.provider not in BATCH_PROVIDERS:
            progress.live += 1
            continue
        in_flight.add(key)
        groups.setdefault(job.candidate.label, (job.candidate, []))[1].append((key, job.messages))

    budget = max_spend
    for candidate, items in groups.values():
        if budget is not None:
            fitting: Items = []
            for key, messages in items:
                worst = worst_case_cost(prices, Job(candidate, messages))
                if worst is None or worst * BATCH_DISCOUNT > budget:
                    progress.held_back += 1
                    continue
                budget -= worst * BATCH_DISCOUNT
                fitting.append((key, messages))
            items = fitting
        if not items:
            continue
        try:
            batch_id = client_for(candidate.provider).submit(candidate, items)
        except ProviderError as e:
            if e.fatal:
                raise
            progress.failed[f"{candidate.label}: could not submit batch ({e})"] += len(items)
            continue
        state.append(
            PendingBatch(
                id=batch_id,
                candidate=candidate,
                keys=[key for key, _ in items],
                input_estimates={
                    key: sum(estimate_tokens(m.content) for m in messages)
                    for key, messages in items
                },
                submitted_at=datetime.now(UTC).isoformat(),
            )
        )
        save_state(state_path, state)
        progress.submitted += len(items)

    deadline = clock() + wait_seconds
    while state:
        for batch in list(state):
            try:
                check = client_for(batch.candidate.provider).check(batch)
            except ProviderError as e:
                if e.fatal:
                    raise
                # A hiccup while checking: the batch stays saved and is checked again.
                if on_status:
                    on_status(f"{batch.candidate.label}: could not check yet ({e})")
                continue
            if on_status and check.progress:
                on_status(f"{batch.candidate.label}: {check.progress}")
            if not check.done:
                continue
            for key in batch.keys:
                result = check.results.get(key, "missing from the batch results")
                if isinstance(result, str):
                    progress.failed[result] += 1
                    continue
                result.cost = batch_cost(prices, batch.candidate, result)
                cache.put(key, batch.candidate, result, None)
                progress.collected += 1
                progress.spent += result.cost or 0.0
            state.remove(batch)
            save_state(state_path, state)
        if not state or clock() >= deadline:
            break
        sleep(poll_seconds)

    progress.pending = sum(len(batch.keys) for batch in state)
    return progress


# --- Anthropic ---------------------------------------------------------------------------------


class AnthropicBatches:
    def __init__(self, client: Any = None) -> None:
        self._client = client

    def _get(self) -> Any:
        if self._client is None:
            import anthropic

            try:
                self._client = anthropic.Anthropic(max_retries=5)
            except anthropic.AnthropicError as e:
                raise ProviderError(
                    f"Could not create the Anthropic client: {e}", fatal=True
                ) from e
        return self._client

    def submit(self, candidate: Candidate, items: Items) -> str:
        import anthropic

        from llm_route_audit.providers.anthropic import build_request

        try:
            batch = self._get().messages.batches.create(
                requests=[
                    {"custom_id": key, "params": build_request(candidate, messages)}
                    for key, messages in items
                ]
            )
        except anthropic.AuthenticationError as e:
            raise ProviderError("Anthropic rejected the API key.", fatal=True) from e
        except anthropic.APIStatusError as e:
            raise ProviderError(f"API error {e.status_code}: {e.message}") from e
        return batch.id

    def check(self, batch: PendingBatch) -> BatchCheck:
        from llm_route_audit.providers.anthropic import parse_response

        info = self._get().messages.batches.retrieve(batch.id)
        counts = info.request_counts
        progress = f"{counts.succeeded + counts.errored} of {len(batch.keys)} done"
        if info.processing_status != "ended":
            return BatchCheck(False, progress=progress)
        results: dict[str, Completion | str] = {}
        for entry in self._get().messages.batches.results(batch.id):
            outcome = entry.result
            if outcome.type == "succeeded":
                results[entry.custom_id] = parse_response(outcome.message)
            else:
                results[entry.custom_id] = f"batch request {outcome.type}"
        return BatchCheck(True, results, progress)


# --- OpenRouter --------------------------------------------------------------------------------


class OpenRouterBatches:
    def __init__(self, base_url: str | None = None) -> None:
        from llm_route_audit.providers.openrouter import BASE_URL

        self.base_url = base_url or BASE_URL

    def _headers(self) -> dict[str, str]:
        key = os.environ.get("OPENROUTER_API_KEY")
        if not key:
            raise ProviderError("OPENROUTER_API_KEY is not set.", fatal=True)
        return {"Authorization": f"Bearer {key}", "X-OpenRouter-Title": "llm-route-audit"}

    def _call(self, path: str, payload: dict[str, Any] | None) -> Any:
        from llm_route_audit.providers.http import HTTPFailure, post_json

        try:
            return post_json(f"{self.base_url}{path}", payload, self._headers(), timeout=120)
        except HTTPFailure as e:
            if e.status in (401, 402):
                raise ProviderError(f"OpenRouter: {e}", fatal=True) from e
            raise ProviderError(f"OpenRouter batch error {e.status}: {e}") from e

    def submit(self, candidate: Candidate, items: Items) -> str:
        from llm_route_audit.providers.openrouter import build_request

        requests = []
        for key, messages in items:
            body = build_request(candidate, messages)
            body.pop("model")  # the batch sets the model once
            requests.append({"custom_id": key, "body": body})
        # OpenRouter asks for endpoint and model to come before the requests.
        data = self._call(
            "/batches",
            {
                "endpoint": "/v1/chat/completions",
                "model": candidate.api_model,
                "requests": requests,
            },
        )
        return str(data["id"])

    def check(self, batch: PendingBatch) -> BatchCheck:
        from llm_route_audit.providers.openrouter import parse_response

        data = self._call(f"/batches/{batch.id}", None)
        counts = data.get("request_counts") or {}
        progress = f"{data.get('status')}, {counts.get('completed', 0)} of {len(batch.keys)} done"
        if data.get("status") not in ("completed", "failed", "expired", "cancelled"):
            return BatchCheck(False, progress=progress)

        results: dict[str, Completion | str] = {}
        for item in data.get("results") or []:
            key = item.get("custom_id")
            response = item.get("response") or {}
            if item.get("error") or response.get("status_code", 200) >= 400:
                error = item.get("error") or {}
                results[key] = f"batch request failed: {error.get('message', 'error')}"
                continue
            try:
                completion = parse_response(response.get("body") or {})
            except ProviderError as e:
                results[key] = str(e)
                continue
            if not completion.input_tokens and not completion.output_tokens:
                # Batch results may omit per-request usage; estimate it from the text.
                completion.input_tokens = batch.input_estimates.get(key, 0)
                completion.output_tokens = estimate_tokens(completion.text)
            results[key] = completion
        _share_batch_cost(results, (data.get("usage") or {}).get("cost"))
        if not results:
            reason = f"batch {data.get('status')}"
            results = {key: reason for key in batch.keys}
        return BatchCheck(True, results, progress)


def _share_batch_cost(results: dict[str, Completion | str], total: float | None) -> None:
    """Spread a batch-level cost over its answers by size (output weighs more, as priced)."""
    completions = [r for r in results.values() if isinstance(r, Completion)]
    if total is None or not completions:
        return
    weights = [c.input_tokens + 4 * c.output_tokens for c in completions]
    whole = sum(weights) or 1
    for completion, weight in zip(completions, weights, strict=True):
        completion.cost = total * weight / whole


def batch_client_for(provider: str) -> BatchClient:
    if provider == "anthropic":
        return AnthropicBatches()
    if provider == "openrouter":
        return OpenRouterBatches()
    raise ValueError(f"no batch API for provider '{provider}'")
