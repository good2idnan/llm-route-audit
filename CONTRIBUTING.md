# Contributing

Thanks for helping improve llm-route-audit.

## Set up

```bash
git clone https://github.com/good2idnan/llm-route-audit.git
cd llm-route-audit
uv sync
```

## Before you open a pull request

```bash
uv run pytest
uv run ruff check .
uv run ruff format --check .
```

- Add or update tests for any change in behaviour. Tests must not call real model APIs; use the fake providers in `tests/` as examples.
- Keep user-facing output plain and specific: say what happened and what to do next.
- Never commit API keys. Keys belong in `.env`, which is git-ignored.

## Good first contributions

- New log importers, for example Langfuse or OpenTelemetry traces.
- New exact checks for grading.
- Provider adapters for other OpenAI-compatible APIs.

## Reporting a bug

Open an issue with the command you ran, the output, and your Python version. Please remove any private data from logs before sharing them.
