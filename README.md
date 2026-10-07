# llm-route-audit

**Find out whether LLM model routing saves money without hurting quality, on your own traffic.**

[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](pyproject.toml)
[![Status: pre-alpha](https://img.shields.io/badge/status-pre--alpha-orange.svg)](#status)

Model routers send easy prompts to cheap models and hard prompts to strong ones. Whether that works for *your* workload is usually a guess. llm-route-audit measures it.

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

llm-route-audit is in early development. The full audit (v0.1) works end to end: validate, analyze, replay, grade, report and export.

| Command | Status |
|---|---|
| `routeaudit validate` | ✅ Available |
| `routeaudit analyze` | ✅ Available |
| `routeaudit replay` | ✅ Available (Anthropic, OpenRouter, Ollama) |
| `routeaudit grade` | ✅ Available |
| `routeaudit report` | ✅ Available |
| `routeaudit export` | ✅ Available (YAML, LiteLLM config) |
| `routeaudit import` | ✅ Available (LiteLLM logs) |

## Installation

Requires Python 3.11+ and [uv](https://docs.astral.sh/uv/). The command-line tool is called `routeaudit`.

```bash
git clone https://github.com/good2idnan/llm-route-audit.git
cd llm-route-audit
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

Remove `--dry-run` to run it. Spending is under your control:

- routeaudit shows the estimate and asks before spending anything (`--yes` skips the question).
- `--budget 1.00` refuses to start if the *estimate* is above $1.00.
- `--max-spend 1.00` is a hard limit on *actual* spend. Before each call it reserves that call's worst-case cost (all of `max_tokens`), so the run can never go over, even with parallel calls. Calls that don't fit are held back and reported.

Answers are cached, so re-running the same replay costs nothing, and an interrupted run picks up where it stopped.

**Providers.** Name each candidate after where it runs:

| Provider | Candidate name | Key in `.env` |
|---|---|---|
| Anthropic | `claude-haiku-4-5` | `ANTHROPIC_API_KEY` |
| [OpenRouter](https://openrouter.ai) | `openrouter/anthropic/claude-haiku-4.5` | `OPENROUTER_API_KEY` |
| [Ollama](https://ollama.com) (local, free) | `ollama/llama3.2` | none |

OpenRouter gives one key for many providers' models, and routeaudit looks up their prices and records what each call actually cost. See [`examples/candidates-openrouter.yaml`](examples/candidates-openrouter.yaml).

**4. Grade the answers**

```bash
uv run routeaudit grade examples/sample_logs.jsonl --config examples/grading.yaml
```

Each replayed answer is compared with the original answer from your log:

1. **Exact checks** run first and cost nothing: valid JSON, fields that must match the original, required text, regular expressions, length limits.
2. **An AI judge** then compares the answers that passed. It reads each pair twice, with the order swapped, because judges tend to favour whichever answer comes first. If the two verdicts disagree, the result counts as a tie.

An answer passes when it clears every check and the judge rates it at least as good as the original. Rules are set per task type in a YAML file (see [`examples/grading.yaml`](examples/grading.yaml)); tasks without rules go straight to the judge. Use `--judge-model` to pick the judge, `--dry-run` to see the plan, and the same cost confirmation, `--budget`, `--max-spend` and cache as replay.

**5. Get the report**

```bash
uv run routeaudit report examples/sample_logs.jsonl
```

For each task type, routeaudit recommends the **cheapest option that keeps at least 95% of the original's pass rate**. It only does so once that option has **at least 10 graded answers**; until then it keeps your current model and says why. It also compares whole-workload strategies, such as "always use model X", against the per-task policy. Example from a small test run (explored with `--min-samples 1`):

```
Whole workload (weighted by traffic)
  Strategy                     Quality  Cost vs now  Coverage
  Current setup (as logged)       100%         100%      100%
  Always claude-haiku-4.5          58%          19%      100%
  Always claude-sonnet-5.5 @ low   78%          61%      100%
  routeaudit policy               100%          26%      100%
```

The same report is saved as a self-contained HTML page (`.routeaudit/report.html`) with a cost/quality chart. It loads nothing from the internet. Change the rules with `--target 0.9` and `--min-samples 30`, or get JSON with `--json`.

**6. Export the policy**

```bash
uv run routeaudit export examples/sample_logs.jsonl --out routing-policy.yaml
```

```yaml
# example
version: 1
default: {model: "claude-opus-5-5"}
routes:
  classify_ticket: {model: "claude-opus-5-5"}  # no cheaper option kept 95% of the original's pass rate
  draft_reply: {model: "claude-haiku-4-5"}  # cheapest option within 95% of the original's pass rate; pass 97% on 40, cost 28%
```

Each route says which model to use for that task type, and why.

Using [LiteLLM](https://github.com/BerriAI/litellm)? Export a ready-to-use proxy config instead:

```bash
uv run routeaudit export examples/sample_logs.jsonl --format litellm --out litellm-config.yaml
```

It creates one model alias per task type (`route/classify_ticket`, `route/draft_reply`, ...). Your app sends each request to the alias for its task, and LiteLLM forwards it to the model the audit chose, with the right effort setting.

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

**Using LiteLLM?** Import its logs instead of writing this format yourself. Point routeaudit at the JSON or JSONL files from LiteLLM's logging callbacks (one file or a folder):

```bash
uv run routeaudit import litellm-logs/ --out logs.jsonl
```

Tag requests in LiteLLM with `task:<name>` (for example `task:classify_ticket`) to get a per-task report. Requests that can't be replayed faithfully yet are skipped and counted: failed calls, LiteLLM cache hits, images and tool calls. API keys and other metadata are not copied.

**Writing your own logs?** Use one JSON object per line:

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
