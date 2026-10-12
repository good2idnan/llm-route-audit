# llm-route-audit

### Prove it before you route.

**Test cheaper AI models on your own requests, and switch only where the answers stay just as good.**

[![PyPI](https://img.shields.io/pypi/v/llm-route-audit.svg)](https://pypi.org/project/llm-route-audit/)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://pypi.org/project/llm-route-audit/)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/good2idnan/llm-route-audit/blob/main/LICENSE)

![Per-task routing compared with using one cheaper model everywhere](https://raw.githubusercontent.com/good2idnan/llm-route-audit/main/docs/images/demo-chart.svg)

<sub>Demo run: the cheapest model everywhere cut cost to 19% but only 58% of answers passed; per-task routing kept every answer passing at 26% of the cost. Only 5 requests, so run it on your own logs.</sub>

## The problem

Most teams send every request to their most expensive model, just to be safe. Model routers promise savings by guessing which requests are "easy", but nobody checks the guesses, so quality can drop without anyone noticing.

llm-route-audit replaces the guess with a test. It replays your real requests on cheaper models, grades each answer against the original, and tells you, task by task, where switching is safe:

| Task | Use | Why |
|---|---|---|
| Sorting tickets | `claude-haiku-4-5` | as good as the original on 49 of 50, 75% cheaper |
| Drafting replies | `claude-sonnet-5-5` at low effort | as good on 48 of 50, 55% cheaper |
| Reviewing contracts | keep `claude-opus-5-5` | no cheaper model was good enough |

<sub>Example output. Your numbers come from your own traffic.</sub>

## What you get

- **Lower bills, same quality.** Switch only the tasks where a cheaper model matched the original.
- **Proof, not guesses.** Every pass rate and cost comes with a 95% range, measured on your requests.
- **No surprise spend.** A cost estimate before anything runs, a hard spending limit, and cached answers.
- **Safe after the switch.** Spot checks and user feedback catch quality drops, and the route can be sent back.
- **Private.** It runs on your machine. Nothing is sent anywhere except the model calls you approve.

## How it works

```
your logs ─▶ test cheaper models ─▶ grade vs the original ─▶ report ─▶ switch ─▶ keep watching
```

1. **Test.** Replay a sample of your logged requests on the models you choose to try.
2. **Grade.** Compare every answer with the original: exact checks first, then an AI judge. Your own labels always win.
3. **Report.** For each task, the cheapest model that kept at least 95% of the original's quality.
4. **Switch and watch.** Export a routing table for your gateway or app, then monitor routed traffic over time.

## Try it in one minute

No API key needed. The repository includes the results of a real run.

```bash
git clone https://github.com/good2idnan/llm-route-audit.git && cd llm-route-audit
uv sync
uv run llm-route-audit report examples/sample_logs.jsonl \
  --replay examples/demo/replay.jsonl --grades examples/demo/grades.jsonl --min-samples 1
```

## Use it on your traffic

```bash
pip install llm-route-audit
```

List the models to try in `candidates.yaml`:

```yaml
candidates:
  - model: claude-haiku-4-5
  - model: claude-sonnet-5-5
    effort: low
```

Then run:

```bash
llm-route-audit analyze logs.jsonl                                  # what you spend today
llm-route-audit replay  logs.jsonl -c candidates.yaml --max-spend 1 # try the cheaper models
llm-route-audit grade   logs.jsonl --config grading.yaml            # mark every answer (rules per task)
llm-route-audit report  logs.jsonl                                  # where switching is safe
llm-route-audit export  logs.jsonl --out routing-policy.yaml        # your routing table
```

After you switch, `monitor`, `outcomes` and `status` keep an eye on quality. Every command shows its cost first and asks before spending.

## Works with

| | |
|---|---|
| **Models** | Anthropic, OpenAI, Google Gemini, OpenRouter, Ollama, any OpenAI-compatible server |
| **Logs** | LiteLLM, Langfuse, OpenTelemetry, or simple JSONL |
| **Routing** | LiteLLM config export, or the built-in runtime router for Python apps |
| **Also audits** | tool-using agents (step by step and whole sessions) and auto-routers such as OpenRouter's |

## Learn more

- [Full guide](https://github.com/good2idnan/llm-route-audit/blob/main/docs/guide.md): every command and option
- [Agents](https://github.com/good2idnan/llm-route-audit/blob/main/docs/guide.md#agents) · [Audit a router](https://github.com/good2idnan/llm-route-audit/blob/main/docs/guide.md#audit-a-router) · [Runtime router](https://github.com/good2idnan/llm-route-audit/blob/main/docs/guide.md#use-the-policy-in-your-app) · [Log format](https://github.com/good2idnan/llm-route-audit/blob/main/docs/guide.md#logs) · [FAQ](https://github.com/good2idnan/llm-route-audit/blob/main/docs/guide.md#faq)
- [Changelog](https://github.com/good2idnan/llm-route-audit/blob/main/CHANGELOG.md) · [Contributing](https://github.com/good2idnan/llm-route-audit/blob/main/CONTRIBUTING.md)

## License

[MIT](https://github.com/good2idnan/llm-route-audit/blob/main/LICENSE)
