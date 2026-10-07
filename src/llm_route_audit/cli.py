"""Command-line entry point: `llm-route-audit ...`."""

import json
from collections import Counter
from collections.abc import Callable
from enum import StrEnum
from pathlib import Path
from typing import Annotated

import typer
import yaml
from dotenv import find_dotenv, load_dotenv
from pydantic import ValidationError

from llm_route_audit import __version__
from llm_route_audit.analyze import UNLABELLED, build_profile
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
    usd,
)
from llm_route_audit.grading.grade import (
    judge_upper_bound,
    load_config,
    plan_grades,
    run_judges,
    write_grades,
)
from llm_route_audit.ingest.common import DEFAULT_TASK_TAG_PREFIX, write_records
from llm_route_audit.ingest.jsonl import LoadResult, load_jsonl
from llm_route_audit.ingest.langfuse import import_langfuse
from llm_route_audit.ingest.litellm import import_litellm
from llm_route_audit.ingest.otel import DEFAULT_TASK_ATTRIBUTE, import_otel
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
from llm_route_audit.providers import get_provider
from llm_route_audit.providers.openrouter import fetch_prices as fetch_openrouter_prices
from llm_route_audit.records import LogRecord
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
from llm_route_audit.runner import execute
from llm_route_audit.sampling import stratified_sample

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
    return None


def _prices(path: Path | None, candidates: list[Candidate] = ()) -> PriceTable:
    """Load prices, then look up OpenRouter and OpenAI models the table doesn't list."""
    try:
        table = load_prices(path)
    except (yaml.YAMLError, ValidationError) as e:
        typer.echo(f"Could not read prices file {path}: {e}", err=True)
        raise typer.Exit(code=1) from None
    missing = {
        c.model: price_id
        for c in candidates
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
        if not typer.confirm(question, default=False):
            typer.echo("Cancelled. Nothing was sent.")
            raise typer.Exit(code=1)


def _progress(noun: str) -> Callable[[int, int], None]:
    def report(done: int, total: int) -> None:
        step = max(1, total // 10)
        if done and (done == total or done % step == 0):
            typer.echo(f"  {done}/{total} {noun}", err=True)

    return report


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
        typer.echo(render_estimate(picked, estimates))
        typer.echo("")
        if dry_run:
            typer.echo("Dry run: nothing was sent.")
            return

        costs = [e.cost for e in estimates]
        _confirm_spend(
            None if None in costs else sum(c for c in costs if c is not None), budget, yes
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
    price_table = _prices(prices, [plan.judge])

    cache = ResultCache(cache_path)
    try:
        cost, calls = plan.estimate_cost(price_table, cache)
        typer.echo(render_grade_plan(plan, cost, calls))
        typer.echo("")
        if dry_run:
            typer.echo("Dry run: no judge calls were made.")
            return
        if calls:
            _confirm_spend(cost, budget, yes)
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
) -> None:
    """Check that routed traffic still meets the quality the audit measured.

    Exits with code 2 when a task's quality has dropped, so it can run from cron or CI.
    """
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
        typer.echo("Nothing to check:")
        for task, why in plan.not_checked.items():
            typer.echo(f"{INDENT}- {task}: {why}")
        if not any(r.switched for r in policy.routes.values()):
            typer.echo(
                "The policy has no switched routes with an audited pass rate. Re-export it "
                "with this version of llm-route-audit."
            )
        return

    price_table = _prices(prices, [c.reference for c in plan.checks] + [judge])
    cache = ResultCache(cache_path)
    try:
        shadow_cost, judge_cost = plan.estimate_cost(price_table, rules, judge)
        tasks = sorted({c.task for c in plan.checks})
        typer.echo(
            f"Monitor plan: {len(plan.checks)} production answers from {len(tasks)} routed "
            f"tasks ({', '.join(tasks)})"
        )
        typer.echo(
            f"{INDENT}Shadow answers from the reference model: est. "
            f"{usd(shadow_cost) if shadow_cost is not None else 'unknown'}"
        )
        typer.echo(
            f"{INDENT}Judge ({judge.label}): up to "
            f"{usd(judge_cost) if judge_cost is not None else 'unknown'}"
        )
        typer.echo("")
        if dry_run:
            typer.echo("Dry run: nothing was sent.")
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
    typer.echo(
        render_monitor(
            health,
            plan,
            source=path.as_posix(),
            spent=shadow_spent + run.judge_spent,
            shadow_failed=failed,
            out_path=out.as_posix(),
        )
    )
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
