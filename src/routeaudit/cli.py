"""Command-line entry point: `routeaudit ...`."""

import json
from collections import Counter
from pathlib import Path
from typing import Annotated

import typer
import yaml
from dotenv import find_dotenv, load_dotenv
from pydantic import ValidationError

from routeaudit import __version__
from routeaudit.analyze import UNLABELLED, build_profile
from routeaudit.cache import ResultCache
from routeaudit.candidates import Candidate, load_candidates
from routeaudit.costs import PriceTable, load_prices
from routeaudit.display import render_estimate, render_profile, render_replay, usd
from routeaudit.ingest.jsonl import LoadResult, load_jsonl
from routeaudit.providers import get_provider
from routeaudit.records import LogRecord
from routeaudit.replay import estimate, logged_cost, run_replay, write_results
from routeaudit.sampling import stratified_sample

app = typer.Typer(
    help="Find out whether LLM model routing saves money without hurting quality, "
    "on your own traffic.",
    no_args_is_help=True,
    add_completion=False,
)

MAX_ERRORS_SHOWN = 20
WORK_DIR = Path(".routeaudit")

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


def _load_records(path: Path) -> list[LogRecord]:
    """Load a log file, or print its errors and exit."""
    result: LoadResult = load_jsonl(path)
    for err in result.errors[:MAX_ERRORS_SHOWN]:
        typer.echo(f"line {err.line}: {err.message}", err=True)
    if len(result.errors) > MAX_ERRORS_SHOWN:
        typer.echo(f"... and {len(result.errors) - MAX_ERRORS_SHOWN} more errors", err=True)
    if not result.ok:
        typer.echo(
            f"{len(result.errors)} invalid lines. Fix them first (see `routeaudit validate`).",
            err=True,
        )
        raise typer.Exit(code=1)
    if not result.records:
        typer.echo("The log file has no records.", err=True)
        raise typer.Exit(code=1)
    return result.records


def _prices(path: Path | None) -> PriceTable:
    try:
        return load_prices(path)
    except (yaml.YAMLError, ValidationError) as e:
        typer.echo(f"Could not read prices file {path}: {e}", err=True)
        raise typer.Exit(code=1) from None


def _candidates(path: Path) -> list[Candidate]:
    try:
        return load_candidates(path)
    except (yaml.YAMLError, ValidationError) as e:
        typer.echo(f"Could not read candidates file {path}: {e}", err=True)
        raise typer.Exit(code=1) from None


def _version(value: bool) -> None:
    if value:
        typer.echo(f"routeaudit {__version__}")
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
    """Check that a log file matches the routeaudit log format."""
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
    yes: Annotated[
        bool, typer.Option("--yes", "-y", help="Skip the confirmation before spending.")
    ] = False,
) -> None:
    """Re-run a sample of your logged requests on cheaper models and save the answers."""
    records = _load_records(path)
    candidate_list = _candidates(candidates)
    price_table = _prices(prices)
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
        total = None if None in costs else sum(c for c in costs if c is not None)
        if budget is not None and (total is None or total > budget):
            typer.echo(
                f"The estimate ({usd(total) if total is not None else 'unknown'}) is over "
                f"your budget of {usd(budget)}. Nothing was sent.",
                err=True,
            )
            raise typer.Exit(code=1)
        needs_spend = total is None or total > 0
        if needs_spend and not yes:
            question = (
                f"Spend about {usd(total)} on API calls?"
                if total is not None
                else "Some models have no price, so the cost is unknown. Continue?"
            )
            if not typer.confirm(question, default=False):
                typer.echo("Cancelled. Nothing was sent.")
                raise typer.Exit(code=1)

        def progress(done: int, total_jobs: int) -> None:
            step = max(1, total_jobs // 10)
            if done and (done == total_jobs or done % step == 0):
                typer.echo(f"  {done}/{total_jobs} answers", err=True)

        run = run_replay(
            picked,
            candidate_list,
            cache,
            price_table,
            provider_for=get_provider,
            concurrency=concurrency,
            on_progress=progress,
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
