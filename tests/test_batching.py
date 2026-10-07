import io
import json
from datetime import date
from types import SimpleNamespace

import pytest

from llm_route_audit.batching import (
    AnthropicBatches,
    BatchCheck,
    OpenRouterBatches,
    PendingBatch,
    _share_batch_cost,
    load_state,
    run_batches,
)
from llm_route_audit.cache import ResultCache, request_key
from llm_route_audit.candidates import Candidate
from llm_route_audit.costs import ModelPrice, PriceTable
from llm_route_audit.providers.base import Completion, ProviderError
from llm_route_audit.records import Message
from llm_route_audit.runner import Job

PRICES = PriceTable(
    updated=date(2026, 10, 1),
    models={
        "claude-x": ModelPrice(input=1, output=10),
        "openrouter/vendor/model-y": ModelPrice(input=1, output=10),
    },
)
CLAUDE = Candidate(model="claude-x", provider="anthropic", max_tokens=1000)
ROUTER = Candidate(model="openrouter/vendor/model-y", max_tokens=1000)
LOCAL = Candidate(model="ollama/llama3.2")


def jobs(candidate, n):
    return [Job(candidate, [Message(role="user", content=f"question {i}")]) for i in range(n)]


class FakeBatches:
    """A batch service that finishes each batch after `ready_after` checks."""

    def __init__(self, ready_after=1, fail_keys=()):
        self.ready_after = ready_after
        self.fail_keys = set(fail_keys)
        self.submitted: dict[str, list[str]] = {}
        self.checks: dict[str, int] = {}

    def submit(self, candidate, items):
        batch_id = f"batch-{len(self.submitted) + 1}"
        self.submitted[batch_id] = [item[0] for item in items]
        return batch_id

    def check(self, batch):
        self.checks[batch.id] = self.checks.get(batch.id, 0) + 1
        if self.checks[batch.id] < self.ready_after:
            return BatchCheck(False, progress="in progress")
        results = {
            key: "batch request errored"
            if key in self.fail_keys
            else Completion(text="answer", input_tokens=100, output_tokens=50)
            for key in batch.keys
        }
        return BatchCheck(True, results, "done")


@pytest.fixture
def cache():
    c = ResultCache(":memory:")
    yield c
    c.close()


def run(cache, tmp_path, work, fake, **kwargs):
    defaults = {"wait_seconds": 0, "poll_seconds": 0, "sleep": lambda s: None}
    defaults.update(kwargs)
    return run_batches(work, cache, PRICES, tmp_path / "batches.json", lambda _: fake, **defaults)


def test_one_batch_per_model_and_answers_land_in_the_cache(cache, tmp_path):
    fake = FakeBatches()
    work = jobs(CLAUDE, 3) + jobs(ROUTER, 2)
    progress = run(cache, tmp_path, work, fake)
    assert sorted(len(keys) for keys in fake.submitted.values()) == [2, 3]
    assert (progress.submitted, progress.collected, progress.pending) == (5, 5, 0)
    hit, _ = cache.get(request_key(CLAUDE, work[0].messages))
    # half the list price: (100 * 1 + 50 * 10) / 1M * 0.5
    assert hit.text == "answer" and hit.cost == pytest.approx(0.0003)
    assert progress.spent == pytest.approx(5 * 0.0003)
    assert load_state(tmp_path / "batches.json") == []


def test_unfinished_batches_are_saved_and_collected_next_time(cache, tmp_path):
    fake = FakeBatches(ready_after=3)
    work = jobs(CLAUDE, 2)
    first = run(cache, tmp_path, work, fake)
    assert (first.submitted, first.pending) == (2, 2)
    assert len(load_state(tmp_path / "batches.json")) == 1

    second = run(cache, tmp_path, work, fake)  # check 2: still running
    assert (second.submitted, second.pending) == (0, 2)  # not sent again
    third = run(cache, tmp_path, work, fake)
    assert (third.collected, third.pending) == (2, 0)
    assert len(fake.submitted) == 1


def test_waiting_polls_until_done(cache, tmp_path):
    fake = FakeBatches(ready_after=3)
    sleeps = []
    clock = iter(range(100)).__next__
    progress = run(
        cache, tmp_path, jobs(CLAUDE, 1), fake, wait_seconds=60, sleep=sleeps.append, clock=clock
    )
    assert progress.collected == 1 and len(sleeps) == 2


def test_cached_answers_are_not_sent(cache, tmp_path):
    work = jobs(CLAUDE, 2)
    cache.put(request_key(CLAUDE, work[0].messages), CLAUDE, Completion("x", 1, 1), None)
    fake = FakeBatches()
    assert run(cache, tmp_path, work, fake).submitted == 1


def test_providers_without_a_batch_api_run_live(cache, tmp_path):
    progress = run(cache, tmp_path, jobs(LOCAL, 3), FakeBatches())
    assert (progress.submitted, progress.live) == (0, 3)


def test_max_spend_holds_back_what_could_break_it(cache, tmp_path):
    # worst case per request: ~(input) + 1000 output tokens at $10/M = ~$0.01, half in batch
    progress = run(cache, tmp_path, jobs(CLAUDE, 10), FakeBatches(), max_spend=0.02)
    assert progress.submitted == 3 and progress.held_back == 7


def test_failed_requests_are_reported_not_cached(cache, tmp_path):
    work = jobs(CLAUDE, 2)
    bad = request_key(CLAUDE, work[1].messages)
    progress = run(cache, tmp_path, work, FakeBatches(fail_keys={bad}))
    assert progress.collected == 1 and progress.failed == {"batch request errored": 1}
    assert cache.get(bad) is None


def test_a_rejected_batch_falls_back_to_live(cache, tmp_path):
    class Refuses(FakeBatches):
        def submit(self, candidate, items):
            raise ProviderError("model has no batch endpoint", disable=True)

    progress = run(cache, tmp_path, jobs(ROUTER, 2), Refuses())
    assert progress.submitted == 0 and sum(progress.failed.values()) == 2


def test_share_batch_cost_by_size():
    small, big = Completion("a", 10, 0), Completion("b", 10, 100)
    _share_batch_cost({"1": small, "2": big, "3": "failed"}, 0.42)  # weights 10 and 410
    assert small.cost == pytest.approx(0.01) and big.cost == pytest.approx(0.41)


# --- provider clients, without network access --------------------------------------------------


class FakeResponse(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


def test_openrouter_batch_round_trip(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test")
    calls = []

    def fake_urlopen(request, timeout):
        calls.append(
            (
                request.get_method(),
                request.full_url,
                json.loads(request.data) if request.data else None,
            )
        )
        if request.get_method() == "POST":
            return FakeResponse(json.dumps({"id": "batch_9", "status": "validating"}).encode())
        body = {
            "status": "completed",
            "request_counts": {"total": 2, "completed": 2},
            "usage": {"cost": 0.002},
            "results": [
                {
                    "custom_id": "k1",
                    "response": {
                        "status_code": 200,
                        "body": {
                            "choices": [{"message": {"content": "hi"}, "finish_reason": "stop"}]
                        },
                    },
                    "error": None,
                },
                {"custom_id": "k2", "response": None, "error": {"message": "overloaded"}},
            ],
        }
        return FakeResponse(json.dumps(body).encode())

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    client = OpenRouterBatches(base_url="https://or.test/api/v1")
    batch_id = client.submit(ROUTER, [("k1", [Message(role="user", content="q")], None)])
    method, url, payload = calls[0]
    assert (method, url, batch_id) == ("POST", "https://or.test/api/v1/batches", "batch_9")
    assert list(payload)[:3] == ["endpoint", "model", "requests"]
    assert payload["model"] == "vendor/model-y" and "model" not in payload["requests"][0]["body"]

    pending = PendingBatch(
        id="batch_9",
        candidate=ROUTER,
        keys=["k1", "k2"],
        input_estimates={"k1": 30},
        submitted_at="t",
    )
    check = client.check(pending)
    assert check.done
    answer = check.results["k1"]
    assert answer.text == "hi" and answer.input_tokens == 30  # estimated: usage was missing
    assert answer.cost == pytest.approx(0.002)
    assert "overloaded" in check.results["k2"]


def test_anthropic_batch_round_trip():
    created = {}
    succeeded = SimpleNamespace(
        type="succeeded",
        message=SimpleNamespace(
            content=[SimpleNamespace(type="text", text="hello")],
            usage=SimpleNamespace(input_tokens=5, output_tokens=2),
            stop_reason="end_turn",
        ),
    )
    batches = SimpleNamespace(
        create=lambda requests: (
            created.update(requests=requests) or SimpleNamespace(id="msgbatch_1")
        ),
        retrieve=lambda batch_id: SimpleNamespace(
            processing_status="ended", request_counts=SimpleNamespace(succeeded=1, errored=1)
        ),
        results=lambda batch_id: [
            SimpleNamespace(custom_id="k1", result=succeeded),
            SimpleNamespace(custom_id="k2", result=SimpleNamespace(type="expired")),
        ],
    )
    client = AnthropicBatches(client=SimpleNamespace(messages=SimpleNamespace(batches=batches)))
    assert (
        client.submit(CLAUDE, [("k1", [Message(role="user", content="q")], None)]) == "msgbatch_1"
    )
    assert created["requests"][0]["custom_id"] == "k1"
    assert created["requests"][0]["params"]["model"] == "claude-x"

    check = client.check(
        PendingBatch(id="msgbatch_1", candidate=CLAUDE, keys=["k1", "k2"], submitted_at="t")
    )
    assert check.done and check.results["k1"].text == "hello"
    assert check.results["k2"] == "batch request expired"


def test_a_hiccup_while_checking_keeps_the_batch(cache, tmp_path):
    class Flaky(FakeBatches):
        def check(self, batch):
            raise ProviderError("timed out")

    progress = run(cache, tmp_path, jobs(CLAUDE, 2), Flaky())
    assert progress.pending == 2
    assert len(load_state(tmp_path / "batches.json")) == 1


def test_bad_key_in_batch_mode_gives_a_clear_message(tmp_path, monkeypatch):
    from pathlib import Path

    from typer.testing import CliRunner

    from llm_route_audit.cli import app

    class BadKey(FakeBatches):
        def submit(self, candidate, items):
            raise ProviderError("OpenRouter: API key expired.", fatal=True)

    monkeypatch.setattr("llm_route_audit.cli.batch_client_for", lambda provider: BadKey())
    sample = Path(__file__).resolve().parent.parent / "examples" / "sample_logs.jsonl"
    candidates = tmp_path / "c.yaml"
    candidates.write_text("candidates:\n  - model: claude-haiku-4-5\n", "utf-8")
    result = CliRunner().invoke(
        app,
        [
            "replay",
            str(sample),
            "-c",
            str(candidates),
            "--sample",
            "3",
            "--batch",
            "--yes",
            "--cache",
            str(tmp_path / "cache.sqlite"),
            "--out",
            str(tmp_path / "r.jsonl"),
        ],
    )
    assert result.exit_code == 1
    assert "Batch mode stopped: OpenRouter: API key expired." in result.output
    assert "Traceback" not in result.output


def test_batch_mode_never_falls_back_to_full_price(tmp_path, monkeypatch):
    from pathlib import Path

    from typer.testing import CliRunner

    from llm_route_audit.cli import app

    class NoBatchEndpoint(FakeBatches):
        def submit(self, candidate, items):
            raise ProviderError("Model does not have a :batch endpoint.", disable=True)

    monkeypatch.setattr("llm_route_audit.cli.batch_client_for", lambda provider: NoBatchEndpoint())
    monkeypatch.setattr(
        "llm_route_audit.cli.get_provider", lambda name: pytest.fail("no live calls")
    )
    sample = Path(__file__).resolve().parent.parent / "examples" / "sample_logs.jsonl"
    candidates = tmp_path / "c.yaml"
    candidates.write_text("candidates:\n  - model: claude-haiku-4-5\n", "utf-8")
    result = CliRunner().invoke(
        app,
        [
            "replay",
            str(sample),
            "-c",
            str(candidates),
            "--sample",
            "3",
            "--batch",
            "--yes",
            "--cache",
            str(tmp_path / "cache.sqlite"),
            "--out",
            str(tmp_path / "r.jsonl"),
        ],
    )
    assert result.exit_code == 1
    assert "nothing was sent for them at full price" in result.output
