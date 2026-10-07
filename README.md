# routeaudit

**Find out whether LLM model routing saves money without hurting quality, on your own traffic.**

[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](pyproject.toml)
[![Status: pre-alpha](https://img.shields.io/badge/status-pre--alpha-orange.svg)](#status)

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

routeaudit is in early development. Validation, traffic analysis and replay work today; grading and reporting are in progress.

| Command | Status |
|---|---|
| `routeaudit validate` | ✅ Available |
| `routeaudit analyze` | ✅ Available |
| `routeaudit replay` | ✅ Available (Anthropic, Ollama) |
| `routeaudit grade` | 🚧 v0.1 |
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

The examples use the bundled sample: a synthetic log from a fictional hosting company. All names and data in it are made up.

**1. Check that a log file is valid**

```bash
uv run routeaudit validate examples/sample_logs.jsonl
```

**2. See what your traffic costs today**

```bash
uv run routeaudit analyze examples/sample_logs.jsonl
```

```
By task type (most expensive first)
  Task             Requests  Share  Avg in  Avg out     Cost  Cost share  Per 1K    p50
  review_contract        30    15%     209    1,126  $0.7005         55%  $23.35  24.1s
  summarize_call         40    20%     199      235  $0.2197         17%   $5.49   6.3s
  draft_reply            45    22%     113      159  $0.1639         13%   $3.64   4.5s
  extract_invoice        40    20%     149      158  $0.1502         12%   $3.76   4.6s
  classify_ticket        45    22%     121       31  $0.0499          4%   $1.11   1.6s
```

In this sample, contract review is 15% of requests but 55% of the cost. Add `--json` for machine-readable output.

**3. Replay a sample on cheaper models**

List the models to test in a candidates file (see [`examples/candidates.yaml`](examples/candidates.yaml)), then preview the plan:

```bash
uv run routeaudit replay examples/sample_logs.jsonl -c examples/candidates.yaml --dry-run
```

```
Replay plan: 50 requests x 3 candidates = 150 answers

  Candidate                Cached  To run  Est. cost
  claude-haiku-4-5              0      50    $0.0827
  claude-sonnet-5-5 @ low       0      50    $0.1653
  claude-opus-5-5 @ low         0      50    $0.3307
  Total                               150    $0.5787
```

Remove `--dry-run` to run it. routeaudit asks before spending anything (`--yes` skips the question, `--budget 1.00` sets a hard limit). Answers are cached, so re-running the same replay costs nothing, and an interrupted run picks up where it stopped.

Set `ANTHROPIC_API_KEY` in your environment or in a `.env` file. To try it for free, use a local [Ollama](https://ollama.com) model such as `ollama/llama3.2` as a candidate.

## Prices

routeaudit works with any provider. It ships with a starter price table, and you can pass your own with `--prices`:

```yaml
# my-prices.yaml (USD per 1M tokens)
updated: 2026-10-01
models:
  my-model-large: { input: 3.00, output: 15.00, cache_read: 0.30 }
  my-model-local: { input: 0, output: 0 }
```

```bash
uv run routeaudit analyze logs.jsonl --prices my-prices.yaml
```

Requests whose model has no price are reported, not silently dropped.

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

## Development

```bash
uv sync
uv run pytest
uv run ruff check . && uv run ruff format --check .
```

Issues and pull requests are welcome.

## License

[MIT](LICENSE) © 2026 Muhammad Idnan
