# routeaudit

**Find out whether LLM model routing saves money without hurting quality, on your own traffic.**

[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](pyproject.toml)
[![Status: pre-alpha](https://img.shields.io/badge/status-pre--alpha-orange.svg)](ROADMAP.md)

Model routers send easy prompts to cheap models and hard prompts to strong ones. Whether that works for *your* workload is usually a guess. routeaudit measures it.

It replays a sample of your real requests on cheaper model options, grades the answers and compares cost against quality. You get a clear report and a routing policy you can use in the gateway you already run.

## Features

- **Uses your own data.** Works on your real request logs, not public benchmarks.
- **Honest baselines.** Every result is compared with "always use the strongest model" and "strongest model at low effort". Sometimes the right answer is not to route.
- **Model and effort.** Tests (model, reasoning effort) pairs, not just models.
- **Cache-aware costs.** Prices include prompt-cache reads and writes.
- **No surprise spend.** Shows a cost estimate and asks before any paid API call.
- **Readable output.** Exports a plain YAML policy, not a black box.
- **Local first.** Your logs stay on your machine.

## How it works

```
your logs ─▶ group by task ─▶ replay a sample ─▶ grade answers ─▶ price ─▶ report + policy
```

1. **Import** request logs (JSONL; LiteLLM and OpenRouter planned).
2. **Group** requests by task type.
3. **Replay** a sample on candidate models and effort levels.
4. **Grade** answers with exact checks first, then an AI judge.
5. **Price** every option with current, cache-aware rates.
6. **Report** cost against quality, and **export** the winning policy.

## Status

routeaudit is in early development. The project setup and log validation are done; the first full audit (v0.1) is in progress. See the [roadmap](ROADMAP.md).

| Command | Status |
|---|---|
| `routeaudit validate` | ✅ Available |
| `routeaudit analyze` | 🚧 v0.1 |
| `routeaudit replay` | 🚧 v0.1 |
| `routeaudit report` | 🚧 v0.1 |
| `routeaudit export` | 🚧 v0.1 |

## Installation

Requires Python 3.11+ and [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/good2idnan/routeaudit.git
cd routeaudit
uv sync
```

## Quick start

Check a log file against the routeaudit format using the bundled sample:

```bash
uv run routeaudit validate examples/sample_logs.jsonl
```

```
OK: 200 records
Models:     claude-opus-5-5 (200)
Task types: draft_reply (45), classify_ticket (45), extract_invoice (40), summarize_call (40), review_contract (30)
```

The sample is a synthetic log from a fictional hosting company. All names and data are made up.

## Log format

One JSON object per line:

```json
{"id": "req_0001", "timestamp": "2026-09-28T09:14:03Z", "model": "claude-opus-5-5",
 "task_type": "classify_ticket",
 "messages": [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}],
 "response": "{\"category\": \"billing\", \"priority\": \"high\"}",
 "input_tokens": 180, "output_tokens": 32, "latency_ms": 1610}
```

| Field | Required | Notes |
|---|---|---|
| `id`, `timestamp`, `model`, `response` | Yes | `id` must be unique |
| `messages` or `prompt` | Yes | Use either one |
| `input_tokens`, `output_tokens` | No | `input_tokens` counts uncached input only |
| `cache_read_tokens`, `cache_write_tokens` | No | For cache-aware pricing |
| `latency_ms`, `task_type`, `outcome`, `metadata` | No | Unknown fields are ignored |

## Pricing

Default prices are in [`src/routeaudit/data/prices.yaml`](src/routeaudit/data/prices.yaml), in USD per 1M tokens. Provider prices change, so check them before relying on any numbers.

## Development

```bash
uv sync
uv run pytest
uv run ruff check . && uv run ruff format --check .
```

Issues and pull requests are welcome.

## License

[MIT](LICENSE) © 2026 Muhammad Idnan
