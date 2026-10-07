"""Command-line entry point: `routeaudit ...`."""

from collections import Counter
from pathlib import Path
from typing import Annotated

import typer

from routeaudit import __version__
from routeaudit.ingest.jsonl import load_jsonl

app = typer.Typer(
    help="Find out whether LLM model routing saves money without hurting quality, "
    "on your own traffic.",
    no_args_is_help=True,
    add_completion=False,
)

MAX_ERRORS_SHOWN = 20


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
    pass


@app.command()
def validate(
    path: Annotated[
        Path,
        typer.Argument(exists=True, dir_okay=False, readable=True, help="JSONL log file to check."),
    ],
) -> None:
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
    tasks = Counter(r.task_type or "(unlabelled)" for r in result.records)
    typer.echo(f"OK: {len(result.records)} records")
    typer.echo("Models:     " + ", ".join(f"{m} ({n})" for m, n in models.most_common()))
    typer.echo("Task types: " + ", ".join(f"{t} ({n})" for t, n in tasks.most_common()))
