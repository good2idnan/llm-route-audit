"""Command-line entry point: `llm-route-audit ...`."""

import json
from collections import Counter
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from enum import StrEnum
from pathlib import Path
from typing import Annotated, Any

import typer
import yaml
from dotenv import find_dotenv, load_dotenv
from pydantic import ValidationError

from llm_route_audit import __version__
from llm_route_audit import outcomes as feedback
from llm_route_audit.analyze import UNLABELLED, build_profile
from llm_route_audit.batching import BATCH_DISCOUNT, BATCH_PROVIDERS, batch_client_for, run_batches
from llm_route_audit.cache import ResultCache
from llm_route_audit.candidates import Candidate, Effort, load_candidates
from llm_route_audit.checkmodel import (
    compare,
    merge_grades,
    merge_results,
    render_check,
    sample_records,
    write_grade_dicts,
)
from llm_route_audit.costs import PriceTable, load_prices
from llm_route_audit.display import (
    INDENT,
    render_estimate,
    render_grade_plan,
    render_grades,
    render_profile,
    render_replay,
    table,
    usd,
)
from llm_route_audit.grading.grade import (
    judge_upper_bound,
    load_config,
    plan_grades,
    run_judges,
    write_grades,
)
from llm_route_audit.grading.labels import LabelError, apply_human_labels, load_human_labels
from llm_route_audit.ingest.common import DEFAULT_TASK_TAG_PREFIX, write_records
from llm_route_audit.ingest.jsonl import LoadResult, load_jsonl
from llm_route_audit.ingest.langfuse import import_langfuse
from llm_route_audit.ingest.litellm import import_litellm
from llm_route_audit.ingest.otel import DEFAULT_TASK_ATTRIBUTE, import_otel
from llm_route_audit.labeling import (
    DecisionLabeler,
    JevLabeler,
    Label,
    LabelerUnavailable,
    LayaLabeler,
    apply_labels,
    label_by_system_prompt,
    load_tasks,
)
from llm_route_audit.monitor import (
    DEFAULT_MIN_CHECKS,
    DEFAULT_PER_TASK,
    DEFAULT_TOLERANCE,
    assess,
    grading_inputs,
    plan_monitor,
    render_monitor,
)
from llm_route_audit.policy import load_policy
from llm_route_audit.providers import ProviderError, get_provider
from llm_route_audit.providers.http import HTTPFailure
from llm_route_audit.providers.openrouter import fetch_prices as fetch_openrouter_prices
from llm_route_audit.records import LogRecord
from llm_route_audit.redaction import load_redaction_config, redact_record
from llm_route_audit.replay import (
    completion_cost,
    estimate,
    load_results,
    logged_cost,
    run_replay,
    worst_case_cost,
    write_results,
)
from llm_route_audit.report import (
    DEFAULT_MIN_SAMPLES,
    DEFAULT_TARGET,
    Report,
    build_report,
    load_grades,
    policy_litellm,
    policy_yaml,
)
from llm_route_audit.report_view import render_html, render_text
from llm_route_audit.rerun import (
    Budget,
    FunctionTools,
    MCPTools,
    RecordedTools,
    ToolSource,
    ToolsUnavailable,
    render_reruns,
    run_session,
    session_estimate,
    write_runs,
)
from llm_route_audit.rerun import apply_grades as apply_rerun_grades
from llm_route_audit.rerun import grading_inputs as rerun_grading_inputs
from llm_route_audit.runner import Job, execute
from llm_route_audit.sampling import sample_sessions, stratified_sample
from llm_route_audit.status import (
    append_history,
    build_tracks,
    load_history,
    monitor_snapshot,
    outcomes_snapshot,
    render_status_html,
    render_status_text,
)

app = typer.Typer(
    help="Find out whether LLM model routing saves money without hurting quality, "
    "on your own traffic.",
    no_args_is_help=True,
    add_completion=False,
)

MAX_ERRORS_SHOWN = 20
WORK_DIR = Path(".llm-route-audit")

LogsArg = Annotated[
    Path,
    typer.Argument(exists=True, dir_okay=False, readable=True, help="JSONL log file."),
]
PricesOpt = Annotated[
    Path | None,
    typer.Option(
        exists=True,
        dir_okay=False,
        readable=True,
        help="Your own prices file (YAML). Defaults to the built-in table.",
    ),
]

MaxSpendOpt = Annotated[
    float | None,
    typer.Option(
        min=0,
        help="Hard limit in USD on new API spend. Calls that could break it are not sent.",
    ),
]


def _load_records(path: Path) -> list[LogRecord]:
    """Load a log file, or print its errors and exit."""
    result: LoadResult = load_jsonl(path)
    for err in result.errors[:MAX_ERRORS_SHOWN]:
        typer.echo(f"line {err.line}: {err.message}", err=True)
    if len(result.errors) > MAX_ERRORS_SHOWN:
        typer.echo(f"... and {len(result.errors) - MAX_ERRORS_SHOWN} more errors", err=True)
    if not result.ok:
        typer.echo(
            f"{len(result.errors)} invalid lines. Fix them first (see `llm-route-audit validate`).",
            err=True,
        )
        raise typer.Exit(code=1)
    if not result.records:
        typer.echo("The log file has no records.", err=True)
        raise typer.Exit(code=1)
    return result.records


def _price_id(candidate: Candidate) -> str | None:
    """The id OpenRouter's public price list uses for a candidate, if it is listed there."""
    if candidate.provider == "openrouter":
        return candidate.api_model
    if candidate.provider == "openai" and not candidate.base_url:
        return f"openai/{candidate.api_model}"  # OpenAI's own models, at OpenAI's prices
    if candidate.provider == "gemini":
        return f"google/{candidate.api_model}"  # Google's models, at Google's prices
    return None


def _price_as(candidate: Candidate) -> Candidate | None:
    if not candidate.price_as:
        return None
    try:
        return Candidate(model=candidate.price_as)
    except ValidationError:
        return None  # a model name the price table must list itself


def _prices(path: Path | None, candidates: list[Candidate] = ()) -> PriceTable:
    """Load prices, then look up OpenRouter, OpenAI and Gemini models the table doesn't
    list, from OpenRouter's public model list."""
    try:
        table = load_prices(path)
    except (yaml.YAMLError, ValidationError) as e:
        typer.echo(f"Could not read prices file {path}: {e}", err=True)
        raise typer.Exit(code=1) from None
    # Routers have no fixed price; their `price_as` model prices estimates instead.
    priced = list(candidates) + [p for c in candidates if (p := _price_as(c))]
    missing = {
        c.model: price_id
        for c in priced
        if c.model not in table.models and (price_id := _price_id(c))
    }
    if missing:
        try:
            found = fetch_openrouter_prices(sorted(set(missing.values())))
        except (OSError, ValueError, KeyError) as e:
            typer.echo(f"Could not fetch model prices: {e}", err=True)
            found = {}
        for model, price_id in missing.items():
            if price_id in found:
                table.models[model] = found[price_id]
    return table


def _candidates(path: Path) -> list[Candidate]:
    try:
        return load_candidates(path)
    except (yaml.YAMLError, ValidationError) as e:
        typer.echo(f"Could not read candidates file {path}: {e}", err=True)
        raise typer.Exit(code=1) from None


def _confirm_spend(total: float | None, budget: float | None, yes: bool) -> None:
    """Stop if the estimate is over budget; otherwise ask before spending (unless --yes)."""
    if budget is not None and (total is None or total > budget):
        typer.echo(
            f"The estimate ({usd(total) if total is not None else 'unknown'}) is over "
            f"your budget of {usd(budget)}. Nothing was sent.",
            err=True,
        )
        raise typer.Exit(code=1)
    if (total is None or total > 0) and not yes:
        question = (
            f"Spend about {usd(total)} on API calls?"
            if total is not None
            else "Some models have no price, so the cost is unknown. Continue?"
        )
        if not typer.confirm(question, default=False, err=True):
            typer.echo("Cancelled. Nothing was sent.", err=True)
            raise typer.Exit(code=1)


def _progress(noun: str) -> Callable[[int, int], None]:
    def report(done: int, total: int) -> None:
        step = max(1, total // 10)
        if done and (done == total or done % step == 0):
            typer.echo(f"  {done}/{total} {noun}", err=True)

    return report


def _say(to_stderr: bool) -> Callable[..., None]:
    """Echo for human-readable lines: stderr when the command prints JSON on stdout."""

    def say(text: str = "") -> None:
        typer.echo(text, err=to_stderr)

    return say


HistoryOpt = Annotated[
    Path,
    typer.Option(help="Route history file; `llm-route-audit status` charts it."),
]
NoHistoryOpt = Annotated[
    bool, typer.Option("--no-history", help="Don't add this run to the route history.")
]
JsonOpt = Annotated[
    bool,
    typer.Option("--json", help="Print the results as JSON; other messages go to stderr."),
]


BatchOpt = Annotated[
    bool,
    typer.Option(
        "--batch",
        help="Use the provider's batch API (Anthropic, OpenRouter): about half price, answers "
        "within 24 hours. Run the command again to collect answers still in progress.",
    ),
]
WaitOpt = Annotated[
    float, typer.Option(min=0, help="Batch mode: minutes to wait for answers before stopping.")
]
PollOpt = Annotated[float, typer.Option(min=5, help="Batch mode: seconds between progress checks.")]


def _batch_discount(provider: str | None, batch: bool) -> float:
    return BATCH_DISCOUNT if batch and provider in BATCH_PROVIDERS else 1.0


def _collect_batches(
    jobs: list[Job],
    cache: ResultCache,
    prices: PriceTable,
    cache_path: Path,
    max_spend: float | None,
    wait_minutes: float,
    poll_seconds: float,
) -> float | None:
    """Run the batch step. Returns the spending limit left for anything that runs live, or
    exits when answers are still in progress."""
    try:
        progress = run_batches(
            jobs,
            cache,
            prices,
            cache_path.parent / "batches.json",
            batch_client_for,
            max_spend=max_spend,
            wait_seconds=wait_minutes * 60,
            poll_seconds=poll_seconds,
            on_status=lambda line: typer.echo(f"{INDENT}{line}", err=True),
        )
    except ProviderError as e:
        typer.echo(f"Batch mode stopped: {e}", err=True)
        typer.echo("Nothing more was sent. Batches already submitted stay saved.", err=True)
        raise typer.Exit(code=1) from None
    typer.echo(
        f"Batch: sent {progress.submitted}, collected {progress.collected} "
        f"(cost {usd(progress.spent)}), still in progress {progress.pending}."
    )
    if progress.held_back:
        typer.echo(f"{INDENT}{progress.held_back} requests not sent, to stay under --max-spend.")
    if progress.live:
        typer.echo(
            f"{INDENT}{progress.live} requests use a provider without a batch API; they run now."
        )
    for reason, n in progress.failed.most_common(3):
        typer.echo(f"{INDENT}{n} x {reason}")
    if progress.failed:
        # Asked for batch prices: never quietly pay full price instead.
        typer.echo(
            "Some requests could not use the batch API, so nothing was sent for them at full "
            "price. Run the same command without --batch to send them normally, or pick a model "
            "that supports batches. Answers already collected are saved.",
            err=True,
        )
        raise typer.Exit(code=1)
    if progress.pending:
        typer.echo(
            "Answers are still being prepared. Run the same command again later to collect "
            "them; nothing is sent or paid for twice."
        )
        raise typer.Exit(code=0)
    return None if max_spend is None else max(0.0, max_spend - progress.spent)


def _version(value: bool) -> None:
    if value:
        typer.echo(f"llm-route-audit {__version__}")
        raise typer.Exit()


@app.callback()
def main(
    version: Annotated[
        bool,
        typer.Option(
            "--version", callback=_version, is_eager=True, help="Show the version and exit."
        ),
    ] = False,
) -> None:
    # API keys may live in a .env file in the current folder; real env vars win.
    load_dotenv(find_dotenv(usecwd=True))


@app.command()
def validate(path: LogsArg) -> None:
    """Check that a log file matches the llm-route-audit log format."""
    result = load_jsonl(path)
    for err in result.errors[:MAX_ERRORS_SHOWN]:
        typer.echo(f"line {err.line}: {err.message}", err=True)
    if len(result.errors) > MAX_ERRORS_SHOWN:
        typer.echo(f"... and {len(result.errors) - MAX_ERRORS_SHOWN} more errors", err=True)
    if not result.ok:
        typer.echo(
            f"{len(result.errors)} invalid lines, {len(result.records)} valid records.", err=True
        )
        raise typer.Exit(code=1)

    models = Counter(r.model for r in result.records)
    tasks = Counter(r.task_type or UNLABELLED for r in result.records)
    typer.echo(f"OK: {len(result.records)} records")
    typer.echo("Models:     " + ", ".join(f"{m} ({n})" for m, n in models.most_common()))
    typer.echo("Task types: " + ", ".join(f"{t} ({n})" for t, n in tasks.most_common()))


@app.command()
def analyze(
    path: LogsArg,
    prices: PricesOpt = None,
    as_json: Annotated[
        bool, typer.Option("--json", help="Print the profile as JSON instead of tables.")
    ] = False,
) -> None:
    """Show what your logged traffic costs today, by task type and model."""
    profile = build_profile(_load_records(path), _prices(prices))
    if as_json:
        typer.echo(json.dumps(profile.to_dict(), indent=2))
    else:
        typer.echo(render_profile(profile, source=path.as_posix()))


@app.command()
def replay(
    path: LogsArg,
    candidates: Annotated[
        Path,
        typer.Option(
            "--candidates",
            "-c",
            exists=True,
            dir_okay=False,
            readable=True,
            help="YAML file listing the models (and effort levels) to test.",
        ),
    ],
    sample: Annotated[
        int, typer.Option(min=1, help="Requests to replay, spread across task types.")
    ] = 50,
    seed: Annotated[int, typer.Option(help="Picks the sample. Same seed, same sample.")] = 0,
    prices: PricesOpt = None,
    out: Annotated[Path, typer.Option(help="Where to save the answers (JSONL).")] = WORK_DIR
    / "replay.jsonl",
    cache_path: Annotated[
        Path, typer.Option("--cache", help="Cache of answers already paid for (SQLite).")
    ] = WORK_DIR / "cache.sqlite",
    concurrency: Annotated[
        int, typer.Option(min=1, max=32, help="Requests to send at the same time.")
    ] = 4,
    budget: Annotated[
        float | None,
        typer.Option(min=0, help="Don't start if the estimate is above this many USD."),
    ] = None,
    dry_run: Annotated[
        bool, typer.Option("--dry-run", help="Show the plan and cost estimate, then stop.")
    ] = False,
    max_spend: MaxSpendOpt = None,
    batch: BatchOpt = False,
    wait_minutes: WaitOpt = 60,
    poll_seconds: PollOpt = 30,
    yes: Annotated[
        bool, typer.Option("--yes", "-y", help="Skip the confirmation before spending.")
    ] = False,
) -> None:
    """Re-run a sample of your logged requests on cheaper models and save the answers."""
    records = _load_records(path)
    candidate_list = _candidates(candidates)
    price_table = _prices(prices, candidate_list)
    picked = stratified_sample(records, sample, seed=seed)
    cache = ResultCache(cache_path)
    try:
        estimates = estimate(picked, candidate_list, price_table, cache)
        for e in estimates:
            if e.cost is not None:
                e.cost *= _batch_discount(e.candidate.provider, batch)
        typer.echo(render_estimate(picked, estimates))
        if batch:
            typer.echo(
                f"{INDENT}Batch mode: Anthropic and OpenRouter candidates at about half price."
            )
        typer.echo("")
        if dry_run:
            typer.echo("Dry run: nothing was sent.")
            return

        costs = [e.cost for e in estimates]
        _confirm_spend(
            None if None in costs else sum(c for c in costs if c is not None), budget, yes
        )
        if batch:
            jobs = [Job(c, r.conversation(), r.tools) for c in candidate_list for r in picked]
            max_spend = _collect_batches(
                jobs, cache, price_table, cache_path, max_spend, wait_minutes, poll_seconds
            )
        run = run_replay(
            picked,
            candidate_list,
            cache,
            price_table,
            provider_for=get_provider,
            concurrency=concurrency,
            on_progress=_progress("answers"),
            max_spend=max_spend,
        )
    finally:
        cache.close()

    write_results(out, run.results)
    typer.echo("")
    typer.echo(
        render_replay(run, picked, logged_cost(price_table, picked), out_path=out.as_posix())
    )
    if run.stopped_reason:
        raise typer.Exit(code=1)


@app.command()
def grade(
    path: LogsArg,
    replay_path: Annotated[
        Path, typer.Option("--replay", help="Answers saved by `llm-route-audit replay`.")
    ] = WORK_DIR / "replay.jsonl",
    config: Annotated[
        Path | None,
        typer.Option(
            exists=True,
            dir_okay=False,
            readable=True,
            help="Grading rules per task type (YAML). Without it, every answer goes to the judge.",
        ),
    ] = None,
    judge_model: Annotated[
        str | None, typer.Option(help="Judge model, e.g. claude-opus-5-5 or ollama/llama3.2.")
    ] = None,
    judge_effort: Annotated[Effort | None, typer.Option(help="Judge effort level.")] = None,
    labels: Annotated[
        Path | None,
        typer.Option(
            exists=True,
            dir_okay=False,
            readable=True,
            help="Your own pass/fail labels (CSV or JSONL). They override the checks and the "
            "judge, and labelled answers are not sent to the judge.",
        ),
    ] = None,
    judge_max_tokens: Annotated[
        int | None, typer.Option(min=1, help="Most tokens the judge may write per verdict.")
    ] = None,
    prices: PricesOpt = None,
    out: Annotated[Path, typer.Option(help="Where to save the grades (JSONL).")] = WORK_DIR
    / "grades.jsonl",
    cache_path: Annotated[
        Path, typer.Option("--cache", help="Cache of answers already paid for (SQLite).")
    ] = WORK_DIR / "cache.sqlite",
    concurrency: Annotated[
        int, typer.Option(min=1, max=32, help="Judge calls to send at the same time.")
    ] = 4,
    budget: Annotated[
        float | None,
        typer.Option(min=0, help="Don't start if the estimate is above this many USD."),
    ] = None,
    dry_run: Annotated[
        bool, typer.Option("--dry-run", help="Run the free checks, show the plan, then stop.")
    ] = False,
    max_spend: MaxSpendOpt = None,
    batch: BatchOpt = False,
    wait_minutes: WaitOpt = 60,
    poll_seconds: PollOpt = 30,
    yes: Annotated[
        bool, typer.Option("--yes", "-y", help="Skip the confirmation before spending.")
    ] = False,
) -> None:
    """Grade replayed answers against the originals: exact checks first, then an AI judge."""
    records = _load_records(path)
    if not replay_path.exists():
        typer.echo(
            f"No replay answers at {replay_path}. Run `llm-route-audit replay` first.", err=True
        )
        raise typer.Exit(code=1)
    results = load_results(replay_path)
    try:
        rules = load_config(config)
        if judge_model:
            rules.judge.model = judge_model
            rules.judge.provider = None
        if judge_effort:
            rules.judge.effort = judge_effort
        if judge_max_tokens:
            rules.judge.max_tokens = judge_max_tokens
        plan = plan_grades(records, results, rules)
    except (yaml.YAMLError, ValidationError) as e:
        typer.echo(f"Could not use the grading settings: {e}", err=True)
        raise typer.Exit(code=1) from None
    if labels is not None:
        try:
            human = apply_human_labels(plan, load_human_labels(labels))
        except LabelError as e:
            typer.echo(f"Could not read the labels: {e}", err=True)
            raise typer.Exit(code=1) from None
        typer.echo(f"Your labels: {human.applied} applied from {labels.name}.")
        if human.unmatched:
            examples = ", ".join(
                f"{label.record_id} / {label.candidate}" for label in human.unmatched[:3]
            )
            typer.echo(
                f"{INDENT}{len(human.unmatched)} labels match no answer in this run "
                f"(check record_id and candidate): {examples}"
            )
    price_table = _prices(prices, [plan.judge])

    cache = ResultCache(cache_path)
    try:
        cost, calls = plan.estimate_cost(price_table, cache)
        if cost is not None:
            cost *= _batch_discount(plan.judge.provider, batch)
        typer.echo(render_grade_plan(plan, cost, calls))
        if batch:
            typer.echo(f"{INDENT}Batch mode: judge calls at about half price.")
        typer.echo("")
        if dry_run:
            typer.echo("Dry run: no judge calls were made.")
            return
        if calls:
            _confirm_spend(cost, budget, yes)
        if batch and calls:
            jobs = [job for _, a, b in plan.judge_pairs for job in (a, b)]
            max_spend = _collect_batches(
                jobs, cache, price_table, cache_path, max_spend, wait_minutes, poll_seconds
            )
        run = run_judges(
            plan,
            cache,
            price_table,
            provider_for=get_provider,
            concurrency=concurrency,
            on_progress=_progress("judge calls"),
            max_spend=max_spend,
        )
    finally:
        cache.close()

    write_grades(out, run.grades)
    typer.echo("")
    typer.echo(render_grades(run, out_path=out.as_posix()))
    if run.stopped_reason:
        raise typer.Exit(code=1)


class ExportFormat(StrEnum):
    yaml = "yaml"
    litellm = "litellm"


ReplayOpt = Annotated[
    Path, typer.Option("--replay", help="Answers saved by `llm-route-audit replay`.")
]
GradesOpt = Annotated[
    Path, typer.Option("--grades", help="Grades saved by `llm-route-audit grade`.")
]
TargetOpt = Annotated[
    float,
    typer.Option(
        min=0.0, max=1.0, help="Share of the original's pass rate a cheaper option must keep."
    ),
]
MinSamplesOpt = Annotated[
    int, typer.Option(min=1, help="Graded answers needed before an option can be recommended.")
]


def _build_report(
    path: Path,
    replay_path: Path,
    grades_path: Path,
    prices: Path | None,
    target: float,
    min_samples: int,
) -> Report:
    records = _load_records(path)
    for file, step in ((replay_path, "replay"), (grades_path, "grade")):
        if not file.exists():
            typer.echo(
                f"No {step} results at {file}. Run `llm-route-audit {step}` first.", err=True
            )
            raise typer.Exit(code=1)
    return build_report(
        records,
        load_results(replay_path),
        load_grades(grades_path),
        _prices(prices),
        target=target,
        min_samples=min_samples,
    )


@app.command()
def report(
    path: LogsArg,
    replay_path: ReplayOpt = WORK_DIR / "replay.jsonl",
    grades_path: GradesOpt = WORK_DIR / "grades.jsonl",
    prices: PricesOpt = None,
    target: TargetOpt = DEFAULT_TARGET,
    min_samples: MinSamplesOpt = DEFAULT_MIN_SAMPLES,
    html_out: Annotated[
        Path, typer.Option("--html", help="Where to save the HTML report.")
    ] = WORK_DIR / "report.html",
    as_json: Annotated[
        bool, typer.Option("--json", help="Print the report as JSON instead of tables.")
    ] = False,
) -> None:
    """Compare cost and quality per task and recommend which model to use for each."""
    result = _build_report(path, replay_path, grades_path, prices, target, min_samples)
    html_out.parent.mkdir(parents=True, exist_ok=True)
    html_out.write_text(render_html(result, source=path.as_posix()), encoding="utf-8")
    if as_json:
        typer.echo(json.dumps(result.to_dict(), indent=2))
    else:
        typer.echo(render_text(result, html_path=html_out.as_posix()))


@app.command()
def export(
    path: LogsArg,
    replay_path: ReplayOpt = WORK_DIR / "replay.jsonl",
    grades_path: GradesOpt = WORK_DIR / "grades.jsonl",
    prices: PricesOpt = None,
    target: TargetOpt = DEFAULT_TARGET,
    min_samples: MinSamplesOpt = DEFAULT_MIN_SAMPLES,
    out: Annotated[
        Path | None, typer.Option(help="Write the policy to this file instead of the screen.")
    ] = None,
    fmt: Annotated[
        ExportFormat,
        typer.Option(
            "--format",
            help="yaml: llm-route-audit's routing table. litellm: a LiteLLM proxy config with one "
            "model alias per task type.",
        ),
    ] = ExportFormat.yaml,
) -> None:
    """Write the recommended routing policy, as a YAML table or a LiteLLM config."""
    result = _build_report(path, replay_path, grades_path, prices, target, min_samples)
    policy = policy_litellm(result) if fmt is ExportFormat.litellm else policy_yaml(result)
    if out is None:
        typer.echo(policy, nl=False)
    else:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(policy, encoding="utf-8")
        typer.echo(f"Policy saved to {out.as_posix()}")


class ImportFormat(StrEnum):
    litellm = "litellm"
    langfuse = "langfuse"
    otel = "otel"


TASK_TIPS = {
    ImportFormat.litellm: 'Tag requests in LiteLLM with "{prefix}<name>"',
    ImportFormat.langfuse: 'Tag traces in Langfuse with "{prefix}<name>", or use --task-from-name',
    ImportFormat.otel: 'Set a "{attribute}" attribute on each GenAI span',
}


@app.command("import")
def import_logs(
    source: Annotated[
        Path,
        typer.Argument(exists=True, readable=True, help="Log file or folder of log files."),
    ],
    fmt: Annotated[
        ImportFormat,
        typer.Option(
            "--format",
            help="Where the logs come from: litellm (logging callbacks), langfuse "
            "(observations export or API), otel (OTLP JSON with GenAI spans).",
        ),
    ] = ImportFormat.litellm,
    out: Annotated[
        Path, typer.Option(help="Where to write the llm-route-audit log file (JSONL).")
    ] = Path("logs.jsonl"),
    task_tag_prefix: Annotated[
        str,
        typer.Option(help="LiteLLM/Langfuse: tags starting with this set the task type."),
    ] = DEFAULT_TASK_TAG_PREFIX,
    task_from_name: Annotated[
        bool,
        typer.Option(help="Langfuse: use each generation's name as its task type."),
    ] = False,
    task_attribute: Annotated[
        str, typer.Option(help="OpenTelemetry: span attribute that holds the task type.")
    ] = DEFAULT_TASK_ATTRIBUTE,
) -> None:
    """Convert logs from LiteLLM, Langfuse or OpenTelemetry into llm-route-audit's log format."""
    if fmt is ImportFormat.langfuse:
        result = import_langfuse(source, task_tag_prefix, task_from_name)
    elif fmt is ImportFormat.otel:
        result = import_otel(source, task_attribute)
    else:
        result = import_litellm(source, task_tag_prefix=task_tag_prefix)

    tasks: Counter = Counter()
    if not result.records:
        typer.echo("No usable requests found.", err=True)
    else:
        write_records(out, result.records)
        tasks = Counter(r.task_type or UNLABELLED for r in result.records)
        typer.echo(f"Imported {len(result.records)} requests to {out.as_posix()}")
        typer.echo("Task types: " + ", ".join(f"{t} ({n})" for t, n in tasks.most_common()))
    if result.skipped:
        typer.echo(f"Skipped {sum(result.skipped.values())}:")
        for reason, n in result.skipped.most_common():
            typer.echo(f"{INDENT}{n} x {reason}")
    if tasks.get(UNLABELLED):
        tip = TASK_TIPS[fmt].format(prefix=task_tag_prefix, attribute=task_attribute)
        typer.echo(
            f"Tip: {tasks[UNLABELLED]} requests have no task type. {tip} to get a per-task report."
        )
    if not result.records:
        raise typer.Exit(code=1)


@app.command()
def monitor(
    path: Annotated[
        Path,
        typer.Argument(
            exists=True,
            dir_okay=False,
            readable=True,
            help="Production logs recorded after you adopted the policy.",
        ),
    ],
    policy_path: Annotated[
        Path,
        typer.Option(
            "--policy",
            exists=True,
            dir_okay=False,
            readable=True,
            help="Policy written by `llm-route-audit export` (YAML format).",
        ),
    ],
    config: Annotated[
        Path | None,
        typer.Option(exists=True, dir_okay=False, readable=True, help="Grading rules (YAML)."),
    ] = None,
    reference_model: Annotated[
        str | None,
        typer.Option(help="Model to compare against. Defaults to each route's reference."),
    ] = None,
    judge_model: Annotated[str | None, typer.Option(help="Judge model.")] = None,
    judge_effort: Annotated[Effort | None, typer.Option(help="Judge effort level.")] = None,
    judge_max_tokens: Annotated[
        int | None, typer.Option(min=1, help="Most tokens the judge may write per verdict.")
    ] = None,
    per_task: Annotated[
        int, typer.Option(min=1, help="Production requests to check per routed task.")
    ] = DEFAULT_PER_TASK,
    seed: Annotated[int, typer.Option(help="Picks the requests. Same seed, same pick.")] = 0,
    tolerance: Annotated[
        float,
        typer.Option(min=0.0, max=1.0, help="How far below the audited pass rate is acceptable."),
    ] = DEFAULT_TOLERANCE,
    min_checks: Annotated[
        int, typer.Option(min=1, help="Checks needed before a task can be called OK.")
    ] = DEFAULT_MIN_CHECKS,
    prices: PricesOpt = None,
    out: Annotated[Path, typer.Option(help="Where to save the grades (JSONL).")] = WORK_DIR
    / "monitor.jsonl",
    cache_path: Annotated[
        Path, typer.Option("--cache", help="Cache of answers already paid for (SQLite).")
    ] = WORK_DIR / "cache.sqlite",
    concurrency: Annotated[int, typer.Option(min=1, max=32, help="Calls at the same time.")] = 4,
    budget: Annotated[
        float | None,
        typer.Option(min=0, help="Don't start if the estimate is above this many USD."),
    ] = None,
    max_spend: MaxSpendOpt = None,
    dry_run: Annotated[
        bool, typer.Option("--dry-run", help="Show the plan and cost estimate, then stop.")
    ] = False,
    yes: Annotated[
        bool, typer.Option("--yes", "-y", help="Skip the confirmation before spending.")
    ] = False,
    history: HistoryOpt = WORK_DIR / "history.jsonl",
    no_history: NoHistoryOpt = False,
    as_json: JsonOpt = False,
) -> None:
    """Check that routed traffic still meets the quality the audit measured.

    Exits with code 2 when a task's quality has dropped, so it can run from cron or CI.
    """
    say = _say(as_json)
    records = _load_records(path)
    try:
        policy = load_policy(policy_path)
        rules = load_config(config)
        if judge_model:
            rules.judge.model = judge_model
            rules.judge.provider = None
        if judge_effort:
            rules.judge.effort = judge_effort
        if judge_max_tokens:
            rules.judge.max_tokens = judge_max_tokens
        judge = rules.judge.candidate()
        plan = plan_monitor(records, policy, per_task, seed, reference_model)
    except (yaml.YAMLError, ValidationError) as e:
        typer.echo(f"Could not use the policy or settings: {e}", err=True)
        raise typer.Exit(code=1) from None

    if not plan.checks:
        if as_json:
            typer.echo(
                json.dumps({"kind": "monitor", "routes": [], "not_checked": plan.not_checked})
            )
            return
        say("Nothing to check:")
        for task, why in plan.not_checked.items():
            say(f"{INDENT}- {task}: {why}")
        if not any(r.switched for r in policy.routes.values()):
            say(
                "The policy has no switched routes with an audited pass rate. Re-export it "
                "with this version of llm-route-audit."
            )
        return

    price_table = _prices(prices, [c.reference for c in plan.checks] + [judge])
    cache = ResultCache(cache_path)
    try:
        shadow_cost, judge_cost = plan.estimate_cost(price_table, rules, judge)
        tasks = sorted({c.task for c in plan.checks})
        say(
            f"Monitor plan: {len(plan.checks)} production answers from {len(tasks)} routed "
            f"tasks ({', '.join(tasks)})"
        )
        say(
            f"{INDENT}Shadow answers from the reference model: est. "
            f"{usd(shadow_cost) if shadow_cost is not None else 'unknown'}"
        )
        say(
            f"{INDENT}Judge ({judge.label}): up to "
            f"{usd(judge_cost) if judge_cost is not None else 'unknown'}"
        )
        say("")
        if dry_run:
            say("Dry run: nothing was sent.")
            return
        total = None if shadow_cost is None or judge_cost is None else shadow_cost + judge_cost
        _confirm_spend(total, budget, yes)

        shadow_jobs = plan.shadow_jobs()
        execution = execute(
            shadow_jobs,
            cache,
            get_provider,
            concurrency=concurrency,
            on_progress=_progress("shadow answers"),
            max_spend=max_spend,
            worst_case=lambda job: worst_case_cost(price_table, job),
            actual_cost=lambda job, c: completion_cost(price_table, job.candidate, c),
        )
        shadow_spent = sum(
            completion_cost(price_table, job.candidate, o.completion) or 0.0
            for job, o in zip(shadow_jobs, execution.outcomes, strict=True)
            if o.completion is not None and not o.cached
        )
        references, answers, failed = grading_inputs(plan, execution)
        run = run_judges(
            plan_grades(references, answers, rules),
            cache,
            price_table,
            provider_for=get_provider,
            concurrency=concurrency,
            on_progress=_progress("judge calls"),
            max_spend=None if max_spend is None else max(0.0, max_spend - shadow_spent),
        )
    finally:
        cache.close()

    write_grades(out, run.grades)
    health = assess(run.grades, plan, tolerance=tolerance, min_checks=min_checks)
    snapshot = monitor_snapshot(health, tolerance, path.as_posix(), policy_path.as_posix())
    if not no_history:
        append_history(history, snapshot)
    if as_json:
        snapshot.update(
            spent=shadow_spent + run.judge_spent,
            shadow_failed=failed,
            not_checked=plan.not_checked,
            grades_file=out.as_posix(),
        )
        typer.echo(json.dumps(snapshot))
    else:
        say(
            render_monitor(
                health,
                plan,
                source=path.as_posix(),
                spent=shadow_spent + run.judge_spent,
                shadow_failed=failed,
                out_path=out.as_posix(),
            )
        )
    if not no_history:
        say(f"Added to the route history ({history.as_posix()}); see `llm-route-audit status`.")
    if execution.stopped_reason or run.stopped_reason:
        raise typer.Exit(code=1)
    if any(h.status == "ALERT" for h in health):
        raise typer.Exit(code=2)


@app.command("check-model")
def check_model(
    path: LogsArg,
    model: Annotated[
        str,
        typer.Option(
            "--model", "-m", help="The model to test, e.g. openrouter/anthropic/claude-sonnet-5.5."
        ),
    ],
    effort: Annotated[Effort | None, typer.Option(help="Effort level for the model.")] = None,
    max_tokens: Annotated[
        int, typer.Option(min=1, help="Most tokens the model may write per answer.")
    ] = 16_000,
    replay_path: ReplayOpt = WORK_DIR / "replay.jsonl",
    grades_path: GradesOpt = WORK_DIR / "grades.jsonl",
    config: Annotated[
        Path | None,
        typer.Option(exists=True, dir_okay=False, readable=True, help="Grading rules (YAML)."),
    ] = None,
    judge_model: Annotated[str | None, typer.Option(help="Judge model.")] = None,
    judge_effort: Annotated[Effort | None, typer.Option(help="Judge effort level.")] = None,
    judge_max_tokens: Annotated[
        int | None, typer.Option(min=1, help="Most tokens the judge may write per verdict.")
    ] = None,
    prices: PricesOpt = None,
    target: TargetOpt = DEFAULT_TARGET,
    min_samples: MinSamplesOpt = DEFAULT_MIN_SAMPLES,
    cache_path: Annotated[
        Path, typer.Option("--cache", help="Cache of answers already paid for (SQLite).")
    ] = WORK_DIR / "cache.sqlite",
    concurrency: Annotated[int, typer.Option(min=1, max=32, help="Calls at the same time.")] = 4,
    budget: Annotated[
        float | None,
        typer.Option(min=0, help="Don't start if the estimate is above this many USD."),
    ] = None,
    max_spend: MaxSpendOpt = None,
    dry_run: Annotated[
        bool, typer.Option("--dry-run", help="Show the plan and cost estimate, then stop.")
    ] = False,
    yes: Annotated[
        bool, typer.Option("--yes", "-y", help="Skip the confirmation before spending.")
    ] = False,
) -> None:
    """Test a new model on the same sample as your last audit and show what it would change.

    The new answers and grades are added to the replay and grades files, so later `report`
    and `export` runs include the new model.
    """
    records = _load_records(path)
    for file, step in ((replay_path, "replay"), (grades_path, "grade")):
        if not file.exists():
            typer.echo(f"No {step} results at {file}. Run an audit first.", err=True)
            raise typer.Exit(code=1)
    results = load_results(replay_path)
    grades = load_grades(grades_path)
    try:
        candidate = Candidate(model=model, effort=effort, max_tokens=max_tokens)
        rules = load_config(config)
        if judge_model:
            rules.judge.model = judge_model
            rules.judge.provider = None
        if judge_effort:
            rules.judge.effort = judge_effort
        if judge_max_tokens:
            rules.judge.max_tokens = judge_max_tokens
        judge = rules.judge.candidate()
    except (yaml.YAMLError, ValidationError) as e:
        typer.echo(f"Could not use these settings: {e}", err=True)
        raise typer.Exit(code=1) from None

    sample = sample_records(records, results)
    price_table = _prices(prices, [candidate, judge])
    before = build_report(records, results, grades, price_table, target, min_samples)

    cache = ResultCache(cache_path)
    try:
        [replay_estimate] = estimate(sample, [candidate], price_table, cache)
        judge_estimate = judge_upper_bound(price_table, judge, rules, sample)
        typer.echo(f"Model check: {candidate.label} on the {len(sample)} requests of your audit")
        typer.echo(
            f"{INDENT}Answers: est. "
            f"{usd(replay_estimate.cost) if replay_estimate.cost is not None else 'unknown'}"
            f" ({replay_estimate.cached} already cached)"
        )
        typer.echo(
            f"{INDENT}Judge ({judge.label}): up to "
            f"{usd(judge_estimate) if judge_estimate is not None else 'unknown'}"
        )
        typer.echo("")
        if dry_run:
            typer.echo("Dry run: nothing was sent.")
            return
        total = (
            None
            if replay_estimate.cost is None or judge_estimate is None
            else replay_estimate.cost + judge_estimate
        )
        _confirm_spend(total, budget, yes)
        run = run_replay(
            sample,
            [candidate],
            cache,
            price_table,
            provider_for=get_provider,
            concurrency=concurrency,
            on_progress=_progress("answers"),
            max_spend=max_spend,
        )
        graded = run_judges(
            plan_grades(records, run.results, rules),
            cache,
            price_table,
            provider_for=get_provider,
            concurrency=concurrency,
            on_progress=_progress("judge calls"),
            max_spend=None if max_spend is None else max(0.0, max_spend - run.spent),
        )
    finally:
        cache.close()

    merged_results = merge_results(results, run.results)
    merged_grades = merge_grades(grades, graded.grades)
    write_results(replay_path, merged_results)
    write_grade_dicts(grades_path, merged_grades)
    after = build_report(records, merged_results, merged_grades, price_table, target, min_samples)
    typer.echo(
        render_check(
            candidate.label,
            compare(before, after, candidate.label),
            before,
            after,
            spent=run.spent + graded.judge_spent,
        )
    )
    if run.stopped_reason or graded.stopped_reason:
        raise typer.Exit(code=1)


@app.command()
def outcomes(
    path: Annotated[
        Path,
        typer.Argument(
            exists=True,
            dir_okay=False,
            readable=True,
            help="Production logs with an outcome per request (from before and after the switch).",
        ),
    ],
    policy_path: Annotated[
        Path,
        typer.Option(
            "--policy",
            exists=True,
            dir_okay=False,
            readable=True,
            help="Policy written by `llm-route-audit export` (YAML format).",
        ),
    ],
    outcome_file: Annotated[
        Path | None,
        typer.Option(
            "--outcomes",
            exists=True,
            dir_okay=False,
            readable=True,
            help="Extra outcomes (CSV or JSONL with record_id and outcome), such as the file "
            "the runtime router writes. They override the logs' own outcome field.",
        ),
    ] = None,
    good: Annotated[
        list[str] | None,
        typer.Option(help="Another outcome value that means good. Repeat for more."),
    ] = None,
    bad: Annotated[
        list[str] | None,
        typer.Option(help="Another outcome value that means bad. Repeat for more."),
    ] = None,
    min_outcomes: Annotated[
        int, typer.Option(min=1, help="Outcomes needed on each model before judging.")
    ] = feedback.DEFAULT_MIN_OUTCOMES,
    tolerance: Annotated[
        float,
        typer.Option(min=0.0, max=1.0, help="How much lower a routed model may score."),
    ] = feedback.DEFAULT_TOLERANCE,
    out: Annotated[
        Path | None,
        typer.Option(help="Write an updated policy here, with REVERT routes sent back."),
    ] = None,
    history: HistoryOpt = WORK_DIR / "history.jsonl",
    no_history: NoHistoryOpt = False,
    as_json: JsonOpt = False,
) -> None:
    """Learn from real-world feedback: are routed tasks getting as many good outcomes as before?

    Exits with code 2 when a route should be reverted, so it can run from cron or CI.
    """
    say = _say(as_json)
    records = _load_records(path)
    try:
        policy = load_policy(policy_path)
        extra = feedback.load_outcome_file(outcome_file) if outcome_file else {}
    except (yaml.YAMLError, ValidationError, ValueError, OSError) as e:
        typer.echo(f"Could not read the policy or outcomes: {e}", err=True)
        raise typer.Exit(code=1) from None
    report = feedback.assess_outcomes(
        records,
        policy,
        extra,
        good=feedback.GOOD | {feedback.outcome_key(v) for v in good or []},
        bad=feedback.BAD | {feedback.outcome_key(v) for v in bad or []},
        min_outcomes=min_outcomes,
        tolerance=tolerance,
    )
    snapshot = outcomes_snapshot(report, path.as_posix(), policy_path.as_posix())
    if not no_history:
        append_history(history, snapshot)
    if as_json:
        snapshot.update(
            with_outcome=report.with_outcome,
            unrecognised=dict(report.unrecognised),
            unmatched_ids=report.unmatched_ids,
            not_switched=report.not_switched,
        )
        typer.echo(json.dumps(snapshot))
    else:
        say(feedback.render_outcomes(report, path.as_posix()))
    if not no_history:
        say(f"Added to the route history ({history.as_posix()}); see `llm-route-audit status`.")
    if out is not None:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(feedback.updated_policy_yaml(policy, report), encoding="utf-8")
        say(f"Updated policy saved to {out.as_posix()}")
    if report.reverts:
        raise typer.Exit(code=2)


@app.command()
def rerun(
    path: LogsArg,
    candidates_path: Annotated[
        Path,
        typer.Option(
            "--candidates",
            "-c",
            exists=True,
            dir_okay=False,
            readable=True,
            help="YAML file listing the models to test.",
        ),
    ],
    sessions_wanted: Annotated[
        int, typer.Option("--sessions", min=1, help="Agent sessions to re-run, spread by type.")
    ] = 10,
    seed: Annotated[int, typer.Option(help="Picks the sessions. Same seed, same pick.")] = 0,
    config: Annotated[
        Path | None,
        typer.Option(
            exists=True,
            dir_okay=False,
            readable=True,
            help="Grading rules (YAML) for the final answers, and agent.ignore_arguments.",
        ),
    ] = None,
    tool_handler: Annotated[
        str | None,
        typer.Option(
            help="Your tools as a Python function, FILE.py:FUNCTION or MODULE:FUNCTION, called "
            "as FUNCTION(name, arguments). Answers calls the log has no result for. Runs for real."
        ),
    ] = None,
    mcp: Annotated[
        str | None,
        typer.Option(
            help='Your tools from an MCP server: a command such as "python server.py", or a URL. '
            "Answers calls the log has no result for. Runs for real."
        ),
    ] = None,
    max_turns: Annotated[
        int | None,
        typer.Option(min=1, help="Most model calls per session. Default: twice the original."),
    ] = None,
    prices: PricesOpt = None,
    out: Annotated[Path, typer.Option(help="Where to save the results (JSONL).")] = WORK_DIR
    / "reruns.jsonl",
    cache_path: Annotated[
        Path, typer.Option("--cache", help="Cache of answers already paid for (SQLite).")
    ] = WORK_DIR / "cache.sqlite",
    budget: Annotated[
        float | None,
        typer.Option(min=0, help="Don't start if the estimate is above this many USD."),
    ] = None,
    max_spend: MaxSpendOpt = None,
    dry_run: Annotated[
        bool, typer.Option("--dry-run", help="Show the plan and cost estimate, then stop.")
    ] = False,
    yes: Annotated[
        bool, typer.Option("--yes", "-y", help="Skip the confirmation before spending.")
    ] = False,
    as_json: JsonOpt = False,
) -> None:
    """Re-run whole agent sessions on cheaper models and see if they still finish the job."""
    say = _say(as_json)
    records = _load_records(path)
    picked = sample_sessions(records, sessions_wanted, seed=seed)
    if not picked:
        typer.echo("No agent sessions found: re-runs need records with a session_id.", err=True)
        raise typer.Exit(code=1)
    candidates = _candidates(candidates_path)
    try:
        rules = load_config(config)
    except (yaml.YAMLError, ValidationError) as e:
        typer.echo(f"Could not use the grading settings: {e}", err=True)
        raise typer.Exit(code=1) from None
    judge = rules.judge.candidate()
    price_table = _prices(prices, [*candidates, judge])

    estimates = {
        c.label: [session_estimate(price_table, c, steps) for _, steps in picked]
        for c in candidates
    }
    finals = [steps[-1] for _, steps in picked]
    judge_cost = judge_upper_bound(price_table, judge, rules, finals)
    total: float | None = 0.0
    say(f"Re-run plan: {len(picked)} sessions x {len(candidates)} candidates")
    counts = Counter(task for task, _ in picked)
    say("Sessions: " + ", ".join(f"{t} {n}" for t, n in sorted(counts.items())))
    rows = []
    for c in candidates:
        values = estimates[c.label]
        cost = None if None in values else sum(v for v in values if v is not None)
        total = None if cost is None or total is None else total + cost
        rows.append([c.label, usd(cost) if cost is not None else "unknown"])
    judge_total = None if judge_cost is None else judge_cost * len(candidates)
    total = None if total is None or judge_total is None else total + judge_total
    rows.append(
        [
            f"judge ({judge.label}), at most",
            usd(judge_total) if judge_total is not None else "unknown",
        ]
    )
    say("")
    for line in table(["Model", "Est. cost"], rows):
        say(line)
    say(
        f"{INDENT}Estimates assume each session takes as many turns as the original. "
        "Use --max-spend for a hard limit."
    )
    tools_note = "the log"
    if tool_handler or mcp:
        tools_note += " first, then " + (
            "your function " + tool_handler if tool_handler else f"MCP server {mcp}"
        )
        say("")
        say(
            "Warning: tool calls the log has no result for will run for real through your tools. "
            "Use test accounts or a sandbox."
        )
    say("")
    if dry_run:
        say("Dry run: nothing was sent.")
        return
    _confirm_spend(total, budget, yes)
    if (tool_handler or mcp) and not yes:
        if not typer.confirm("Run your tools for real when needed?", default=False):
            raise typer.Exit(code=1)

    live: list[ToolSource] = []
    try:
        if tool_handler:
            live.append(FunctionTools.load(tool_handler))
        if mcp:
            live.append(MCPTools(mcp))
    except ToolsUnavailable as e:
        typer.echo(str(e), err=True)
        raise typer.Exit(code=1) from None

    cache = ResultCache(cache_path)
    spend = Budget(limit=max_spend)
    runs = []
    providers: dict[str, Any] = {}
    try:
        done = 0
        progress = _progress("sessions")
        for candidate in candidates:
            provider = providers.setdefault(candidate.provider, get_provider(candidate.provider))
            for task, steps in picked:
                ignore = rules.rule_for(task).agent.ignore_arguments
                sources = [RecordedTools(steps, ignore), *live]
                try:
                    runs.append(
                        run_session(candidate, steps, task, provider, sources, cache,
                                    price_table, spend, max_turns, ignore)
                    )  # fmt: skip
                except ProviderError as e:
                    typer.echo(f"Stopped: {e}", err=True)
                    raise typer.Exit(code=1) from None
                done += 1
                progress(done, len(picked) * len(candidates))
        steps_of = {steps[0].session_id or steps[0].id: steps for _, steps in picked}
        judge_records, answers = rerun_grading_inputs(runs, steps_of)
        plan = plan_grades(judge_records, answers, rules)
        remaining = None if max_spend is None else max(0.0, max_spend - spend.spent)
        graded = run_judges(
            plan, cache, price_table, provider_for=get_provider, max_spend=remaining
        )
        spend.spent += graded.judge_spent
        apply_rerun_grades(runs, graded.grades)
    finally:
        cache.close()
        for source in live:
            if isinstance(source, MCPTools):
                source.close()

    write_runs(out, runs)
    if as_json:
        typer.echo(
            json.dumps(
                {"spent": spend.spent, "tools": tools_note, "runs": [r.to_dict() for r in runs]}
            )
        )
    else:
        say("")
        say(render_reruns(runs, path.as_posix(), tools_note, spend.spent))
    say(f"\nResults saved to {out.as_posix()}")


@app.command()
def status(
    history: HistoryOpt = WORK_DIR / "history.jsonl",
    html_path: Annotated[
        Path, typer.Option("--html", help="Where to save the status page.")
    ] = WORK_DIR / "status.html",
    as_json: JsonOpt = False,
) -> None:
    """Route health over time, from every `monitor` and `outcomes` run, with a status page.

    Exits with code 2 when a route's latest check is ALERT or REVERT.
    """
    tracks = build_tracks(load_history(history))
    html_path.parent.mkdir(parents=True, exist_ok=True)
    html_path.write_text(render_status_html(tracks, history.as_posix()), encoding="utf-8")
    if as_json:
        typer.echo(json.dumps({"routes": [t.to_dict() for t in tracks]}))
    else:
        typer.echo(render_status_text(tracks, history.as_posix(), html_path.as_posix()))
    if any(t.needs_attention for t in tracks):
        raise typer.Exit(code=2)


class LabelMethod(StrEnum):
    system_prompt = "system-prompt"
    laya = "laya"
    jev = "jev"


@app.command()
def label(
    path: LogsArg,
    out: Annotated[Path, typer.Option(help="Where to write the labelled log file (JSONL).")],
    by: Annotated[
        LabelMethod,
        typer.Option(
            help="system-prompt: group requests that share system instructions (free, instant). "
            "laya: sort into --tasks with the local laya model. "
            "jev: sort into --tasks with TypeSafe's Jev API (paid, sends request text)."
        ),
    ] = LabelMethod.system_prompt,
    tasks: Annotated[
        Path | None,
        typer.Option(
            exists=True,
            dir_okay=False,
            readable=True,
            help="laya/jev: YAML file listing your task types and a one-line description of each.",
        ),
    ] = None,
    min_confidence: Annotated[
        float,
        typer.Option(min=0.0, max=1.0, help="laya/jev: leave a request unlabelled below this."),
    ] = 0.6,
    concurrency: Annotated[
        int, typer.Option(min=1, max=32, help="jev: requests to send at the same time.")
    ] = 8,
    yes: Annotated[
        bool, typer.Option("--yes", "-y", help="jev: skip the confirmation before spending.")
    ] = False,
    overwrite: Annotated[
        bool, typer.Option(help="Replace task types the log already has.")
    ] = False,
) -> None:
    """Give each request a task type, so reports can recommend a model per kind of request."""
    records = _load_records(path)
    failed = 0
    if by is LabelMethod.system_prompt:
        labels = label_by_system_prompt(records)
    else:
        if tasks is None:
            typer.echo(f"--by {by} needs --tasks, a YAML file listing your task types.", err=True)
            raise typer.Exit(code=1)
        labeler_class = LayaLabeler if by is LabelMethod.laya else JevLabeler
        try:
            labeler = labeler_class(load_tasks(tasks), min_confidence=min_confidence)
        except (yaml.YAMLError, ValidationError) as e:
            typer.echo(f"Could not read tasks file {tasks}: {e}", err=True)
            raise typer.Exit(code=1) from None
        except LabelerUnavailable as e:
            typer.echo(str(e), err=True)
            raise typer.Exit(code=1) from None
        todo = [r for r in records if overwrite or not r.task_type]
        if isinstance(labeler, JevLabeler) and todo:
            typer.echo(
                f"Jev will read {len(todo)} requests (about {labeler.input_tokens(todo):,} "
                f"tokens). The request text is sent to TypeSafe."
            )
            typer.echo(
                f"{INDENT}Estimated cost: {usd(labeler.cost(todo))} at the launch price "
                "of $0.042 per 1M input tokens."
            )
            _confirm_spend(labeler.cost(todo), None, yes)
        try:
            labels, failed = _label_all(labeler, todo, concurrency if by is LabelMethod.jev else 1)
        except LabelerUnavailable as e:
            typer.echo(f"Labelling stopped: {e}", err=True)
            raise typer.Exit(code=1) from None

    labelled = apply_labels(records, labels, overwrite=overwrite)
    write_records(out, labelled)
    counts = Counter(r.task_type or UNLABELLED for r in labelled)
    kept = sum(1 for r in records if r.task_type and not overwrite)
    typer.echo(f"Labelled {len(labelled)} requests, saved to {out.as_posix()}")
    typer.echo("Task types: " + ", ".join(f"{t} ({n})" for t, n in counts.most_common()))
    if kept:
        typer.echo(f"{kept} requests kept the task type they already had (use --overwrite).")
    if counts.get(UNLABELLED):
        hint = (
            "they have no system instructions to group by"
            if by is LabelMethod.system_prompt
            else f"{by} was less than {min_confidence:.0%} sure or chose none of your tasks"
        )
        typer.echo(f"{counts[UNLABELLED]} requests stayed unlabelled: {hint}.")
    if failed:
        typer.echo(f"{failed} requests could not be labelled (API errors); run again to retry.")


def _label_all(
    labeler: DecisionLabeler, records: list[LogRecord], concurrency: int
) -> tuple[dict[str, Label], int]:
    """Label records, several at a time. Returns the labels and how many failed."""
    labels: dict[str, Label] = {}
    failed = 0
    progress = _progress("requests")
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = {pool.submit(labeler.label, r): r for r in records}
        for done, future in enumerate(as_completed(futures), start=1):
            try:
                labels[futures[future].id] = future.result()
            except HTTPFailure:
                failed += 1
            progress(done, len(records))
    return labels, failed


@app.command()
def redact(
    path: LogsArg,
    out: Annotated[Path, typer.Option(help="Where to write the cleaned log file (JSONL).")],
    rules: Annotated[
        Path | None,
        typer.Option(
            exists=True,
            dir_okay=False,
            readable=True,
            help="YAML file choosing which types to hide and adding your own patterns.",
        ),
    ] = None,
) -> None:
    """Hide private data (emails, phone numbers, cards, secrets, ...) in a copy of your logs.

    Run the audit on the cleaned copy, so no private values are sent to any model. The
    private values themselves are never printed.
    """
    records = _load_records(path)
    try:
        config = load_redaction_config(rules)
    except (yaml.YAMLError, ValidationError) as e:
        typer.echo(f"Could not read redaction rules {rules}: {e}", err=True)
        raise typer.Exit(code=1) from None

    active = config.rules()
    cleaned, totals, changed = [], Counter(), 0
    for record in records:
        updated, counts = redact_record(record, active)
        cleaned.append(updated)
        totals.update(counts)
        changed += bool(counts)
    write_records(out, cleaned)

    typer.echo(f"Checked {len(records)} requests; {changed} contained private data.")
    if totals:
        typer.echo("Hidden: " + ", ".join(f"{n} {t.lower()}" for t, n in totals.most_common()))
    else:
        typer.echo("Nothing to hide was found.")
    typer.echo(f"Cleaned copy saved to {out.as_posix()}. Run the audit on that file.")
    typer.echo(
        f"{INDENT}Checked types: {', '.join(config.types)}"
        + (f"; your rules: {', '.join(c.name for c in config.custom)}" if config.custom else "")
    )
