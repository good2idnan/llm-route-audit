# Changelog

## Unreleased

- `routeaudit import --format langfuse`: import Langfuse GENERATION observations from a UI export or the API, with task types from trace tags or generation names.
- `routeaudit import --format otel`: import OpenTelemetry GenAI spans from OTLP JSON, using the `gen_ai.*` message and usage attributes, with task types from a span attribute.
- `routeaudit check-model`: test a new model (and effort) on the same sample as the last audit. It adds the results to the audit files and shows which tasks it would take over and how savings change.
- `routeaudit monitor`: shadow checks after you adopt a policy. It samples production requests on each switched route and replays them on the reference model. It grades the production answer against the reference and reports OK, WAIT, WARN or ALERT for each task. Exits with code 2 on an alert.
- `routeaudit export` now writes each switched route's reference model and audited pass rate (`reference`, `expected_pass_rate`), which `monitor` checks against.

## 0.1.0 (2026-10-07)

First public release. It runs the full audit end to end on single-request workloads.

- `routeaudit import`: convert LiteLLM logs (StandardLoggingPayload JSON/JSONL). Task types come from `task:` request tags. Failed calls, cache hits, images and tool calls are skipped and counted.
- `routeaudit validate`: check a log file, with errors reported by line number.
- `routeaudit analyze`: cost by task type and model, a monthly estimate and latency. Accepts `--prices` and `--json`.
- `routeaudit replay`: re-run a sample of logged requests, spread across task types, on candidate (model, effort) pairs.
  - Providers: Anthropic, OpenRouter and Ollama.
  - Spending: cost estimate with confirmation, `--dry-run`, `--budget`, and a hard `--max-spend` limit.
  - Every answer is cached, so the same request is never paid for twice.
- `routeaudit grade`: compare each answer with the original.
  - Exact checks run first: JSON, matching fields, exact match, contains, regex, length.
  - An AI judge then reads each pair in both orders, and its consistency is reported.
- `routeaudit report`: per-task recommendation (the cheapest option within 95% of the original's pass rate, with at least 10 graded answers), a comparison of whole-workload strategies, 95% ranges, and an offline HTML report.
- `routeaudit export`: the routing policy as YAML or as a LiteLLM proxy config with one alias per task type.
