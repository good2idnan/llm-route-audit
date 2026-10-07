# Changelog

## 0.2.1 (2026-10-07)

- New check `json_schema`: the answer must be JSON that fits a JSON Schema, inline or from a file next to the grading file. Errors name the first problem, such as `$.priority: 'critical' is not one of [...]`.
- `grade --labels labels.csv`: your own pass/fail verdicts (CSV or JSONL) override the checks and the judge, and labelled answers are not sent to the judge. Unknown record ids or candidates are listed.
- Cost ratios get a 95% range (bootstrap over the sampled requests) in the report, the HTML report, the JSON output and the exported policy's comments.
- CI and publish workflows run only when started by hand.

## 0.2.0 (2026-10-07)

Agent audits: tool-using agents are audited step by step.

- The log format has agent steps: `tool` messages, `tool_calls` on assistant messages, `response_tool_calls`, `tools` (definitions) and `session_id`.
- `import` keeps agent steps instead of skipping them, for LiteLLM, Langfuse (OpenAI and Anthropic message formats) and OpenTelemetry (`tool_call` and `tool_call_response` parts, `gen_ai.tool.definitions`). Steps are grouped into sessions by session or conversation id, or else by trace.
- `replay` sends each step's tool definitions and history to Anthropic, OpenAI, OpenRouter, Ollama and OpenAI-compatible servers, in batch mode too, and saves the tool calls the candidate makes. Logs with sessions are sampled as whole sessions.
- `grade` compares tool steps call by call (same tools, same arguments, any order). Per task: `agent: {ignore_arguments: [...], judge_alternatives: true}` to leave out free-text arguments and to let the judge accept a different but reasonable step. The judge now sees tool calls, tool results and tool definitions.
- `report` routes agent logs per session type and adds an "Agent sessions, step by step" section: same tool calls, answers passed, sessions that matched at every step, the typical first split, and cost per session.
- Costs assume a candidate gets the same share of prompt-cache hits as the original when the log shows caching, since a replay never hits the cache.
- `monitor` checks agent steps by comparing production tool calls with the reference model's.
- New example: `examples/agent_logs.jsonl` (a synthetic support agent, 40 sessions) with `examples/grading-agent.yaml`.

## 0.1.3 (2026-10-07)

- `llm-route-audit redact`: hide private data in a copy of your logs, offline. Covers emails, phones, checksum-verified cards, IBANs, SSN and UK NI numbers, IPs, links with tokens, API keys and passwords, and dates of birth, plus your own patterns. Placeholders stay consistent between the request and the original answer.
- Batch mode stops with a clear message instead of paying full price when a model doesn't accept batches.
- `--batch` for `replay` and `grade`: half-price batch APIs from Anthropic and OpenRouter. It sends what the run still needs, waits (`--wait-minutes`, `--poll-seconds`), and saves answers to the cache. Unfinished batches are tracked in `.llm-route-audit/batches.json`, and running the command again collects them without resending. `--max-spend` is checked before anything is sent.

## 0.1.2 (2026-10-07)

- `llm-route-audit label`: give requests a task type. `--by system-prompt` (default) groups requests that share system instructions, with no extra install. `--by laya` sorts requests into task types you describe with the open laya model, installed with `pip install "llm-route-audit[laya]"`.
- OpenAI provider: candidates named `openai/<model>`, with `reasoning_effort`, refusal and truncation detection, and automatic price lookup.
- Any OpenAI-compatible server through `base_url` and `api_key_env` (Groq, Together, vLLM, LM Studio, Ollama's OpenAI API). Servers on your own machine count as free.
- Effort levels `none` and `minimal` for models that support them.

## 0.1.1 (2026-10-07)

- Renamed everything to match the project name. The command is now `llm-route-audit` (was `routeaudit`), the Python package is `llm_route_audit`, and results go to `.llm-route-audit/`.
- The per-task recommendation is called "Per-task policy" in reports.

## 0.1.0 (2026-10-07)

First public release. It runs the full audit end to end on single-request workloads.

- `llm-route-audit import`: convert LiteLLM logs (StandardLoggingPayload JSON/JSONL). Task types come from `task:` request tags. Failed calls, cache hits, images and tool calls are skipped and counted.
- `llm-route-audit validate`: check a log file, with errors reported by line number.
- `llm-route-audit analyze`: cost by task type and model, a monthly estimate and latency. Accepts `--prices` and `--json`.
- `llm-route-audit replay`: re-run a sample of logged requests, spread across task types, on candidate (model, effort) pairs.
  - Providers: Anthropic, OpenRouter and Ollama.
  - Spending: cost estimate with confirmation, `--dry-run`, `--budget`, and a hard `--max-spend` limit.
  - Every answer is cached, so the same request is never paid for twice.
- `llm-route-audit grade`: compare each answer with the original.
  - Exact checks run first: JSON, matching fields, exact match, contains, regex, length.
  - An AI judge then reads each pair in both orders, and its consistency is reported.
- `llm-route-audit report`: per-task recommendation (the cheapest option within 95% of the original's pass rate, with at least 10 graded answers), a comparison of whole-workload strategies, 95% ranges, and an offline HTML report.
- `llm-route-audit export`: the routing policy as YAML or as a LiteLLM proxy config with one alias per task type.
- `llm-route-audit import --format langfuse`: import Langfuse GENERATION observations from a UI export or the API, with task types from trace tags or generation names.
- `llm-route-audit import --format otel`: import OpenTelemetry GenAI spans from OTLP JSON, using the `gen_ai.*` message and usage attributes, with task types from a span attribute.
- `llm-route-audit check-model`: test a new model (and effort) on the same sample as the last audit. It adds the results to the audit files and shows which tasks it would take over and how savings change.
- `llm-route-audit monitor`: shadow checks after you adopt a policy. It samples production requests on each switched route and replays them on the reference model. It grades the production answer against the reference and reports OK, WAIT, WARN or ALERT for each task. Exits with code 2 on an alert.
- `llm-route-audit export` now writes each switched route's reference model and audited pass rate (`reference`, `expected_pass_rate`), which `monitor` checks against.
