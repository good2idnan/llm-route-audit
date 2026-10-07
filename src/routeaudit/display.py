"""Plain-text rendering for terminal output. ASCII only, so it prints on any console."""

from collections import Counter

from routeaudit.analyze import UNLABELLED, GroupStats, TrafficProfile, percentile
from routeaudit.records import LogRecord
from routeaudit.replay import CandidateEstimate, ReplayRun

INDENT = "  "


def usd(amount: float | None) -> str:
    if amount is None:
        return "-"
    return f"${amount:,.2f}" if amount >= 1 else f"${amount:.4f}"


def pct(fraction: float) -> str:
    return f"{fraction:.0%}"


def seconds(ms: float | None) -> str:
    return "-" if ms is None else f"{ms / 1000:.1f}s"


def table(headers: list[str], rows: list[list[str]]) -> list[str]:
    """First column left-aligned, the rest right-aligned."""
    widths = [max([len(h), *(len(r[i]) for r in rows)]) for i, h in enumerate(headers)]

    def line(cells: list[str]) -> str:
        parts = [
            c.ljust(w) if i == 0 else c.rjust(w)
            for i, (c, w) in enumerate(zip(cells, widths, strict=True))
        ]
        return (INDENT + "  ".join(parts)).rstrip()

    return [line(headers), *(line(r) for r in rows)]


def _per_thousand(group: GroupStats) -> str:
    per_request = group.cost_per_request
    return usd(None if per_request is None else per_request * 1000)


def render_profile(profile: TrafficProfile, source: str) -> str:
    total = profile.total
    out = [
        f"Traffic profile: {source}",
        "",
        f"{INDENT}Requests      {total.requests:,} over {profile.span_days:.1f} days "
        f"({profile.first_seen:%Y-%m-%d} to {profile.last_seen:%Y-%m-%d})",
        f"{INDENT}Total cost    {usd(total.cost)}  ({_per_thousand(total)} per 1,000 requests)",
    ]
    if profile.monthly_cost is not None:
        out.append(f"{INDENT}Monthly est.  {usd(profile.monthly_cost)} at this volume")
    out += [
        f"{INDENT}Tokens        {total.input_tokens:,} input, {total.output_tokens:,} output, "
        f"{total.cache_read_tokens:,} cache read",
        f"{INDENT}Prices as of  {profile.prices_updated.isoformat()}",
        "",
        "By task type (most expensive first)",
    ]
    out += table(
        ["Task", "Requests", "Share", "Avg in", "Avg out", "Cost", "Cost share", "Per 1K", "p50"],
        [
            [
                g.name,
                f"{g.requests:,}",
                pct(g.requests / total.requests),
                f"{g.avg_input_tokens:,.0f}",
                f"{g.avg_output_tokens:,.0f}",
                usd(g.cost),
                pct(g.cost / total.cost) if total.cost else "-",
                _per_thousand(g),
                seconds(percentile(g.latencies_ms, 50)),
            ]
            for g in profile.by_task
        ],
    )
    out += ["", "By model"]
    out += table(
        ["Model", "Requests", "Cost", "Cost share"],
        [
            [
                g.name,
                f"{g.requests:,}",
                usd(g.cost) if g.priced_requests else "no price",
                pct(g.cost / total.cost) if total.cost else "-",
            ]
            for g in profile.by_model
        ],
    )

    notes = []
    if profile.unpriced_models:
        names = ", ".join(f"{m} ({n})" for m, n in profile.unpriced_models.items())
        notes.append(
            f"No price for: {names}. These requests are left out of all costs. "
            "Add the models to a prices file and pass --prices."
        )
    if profile.estimated_records:
        notes.append(
            f"{profile.estimated_records:,} requests had no token counts; "
            "they were estimated from text length (about 4 characters per token)."
        )
    if profile.unlabelled_records:
        notes.append(
            f"{profile.unlabelled_records:,} requests have no task_type and are grouped as "
            "(unlabelled). Add task_type to your logs for a per-task breakdown."
        )
    if notes:
        out += ["", "Notes"] + [f"{INDENT}- {n}" for n in notes]
    return "\n".join(out)


def render_estimate(sample: list[LogRecord], estimates: list[CandidateEstimate]) -> str:
    mix = Counter(r.task_type or UNLABELLED for r in sample)
    to_run = sum(e.requests - e.cached for e in estimates)
    costs = [e.cost for e in estimates]
    total = None if None in costs else sum(c for c in costs if c is not None)
    out = [
        f"Replay plan: {len(sample)} requests x {len(estimates)} candidates "
        f"= {len(sample) * len(estimates)} answers",
        "Sample: " + ", ".join(f"{t} {n}" for t, n in sorted(mix.items())),
        "",
    ]
    out += table(
        ["Candidate", "Cached", "To run", "Est. cost"],
        [
            [e.candidate.label, f"{e.cached}", f"{e.requests - e.cached}", usd(e.cost)]
            for e in estimates
        ]
        + [["Total", "", f"{to_run}", usd(total) if total is not None else "unknown"]],
    )
    out += [
        "",
        f"{INDENT}The estimate assumes answers as long as the originals. "
        "Thinking can make real costs higher.",
    ]
    unpriced = [e.candidate.label for e in estimates if e.cost is None]
    if unpriced:
        out.append(f"{INDENT}No price for: {', '.join(unpriced)}. Add them with --prices.")
    return "\n".join(out)


def render_replay(
    run: ReplayRun, sample: list[LogRecord], original_cost: float | None, out_path: str
) -> str:
    fresh = sum(1 for r in run.results if not r.cached and r.status not in ("error", "skipped"))
    cached = sum(1 for r in run.results if r.cached)
    lines = [
        f"Replay finished: {fresh} new answers, {cached} from cache. Spent {usd(run.spent)}.",
        f"Answers saved to {out_path}",
        "",
    ]
    rows = [
        [
            "original (as logged)",
            f"{len(sample)}",
            "-",
            "-",
            "-",
            "-",
            usd(original_cost),
            seconds(percentile([r.latency_ms for r in sample if r.latency_ms is not None], 50)),
        ]
    ]
    labels = list(dict.fromkeys(_label(r.model, r.effort) for r in run.results))
    for label in labels:
        results = [r for r in run.results if _label(r.model, r.effort) == label]
        status = Counter(r.status for r in results)
        priced = [r.cost for r in results if r.cost is not None]
        latencies = [r.latency_ms for r in results if r.latency_ms is not None]
        rows.append(
            [
                label,
                f"{status['ok']}",
                f"{status['refusal']}",
                f"{status['truncated']}",
                f"{status['error']}",
                f"{status['skipped']}",
                usd(sum(priced)) if priced else "-",
                seconds(percentile(latencies, 50)),
            ]
        )
    lines += table(
        ["Candidate", "OK", "Refused", "Truncated", "Errors", "Skipped", "Sample cost", "p50"],
        rows,
    )

    notes = [f"Stopped early: {run.stopped_reason}"] if run.stopped_reason else []
    notes += [f"{label} was skipped after: {reason}" for label, reason in run.disabled.items()]
    errors = Counter(r.error for r in run.results if r.status == "error" and r.error)
    notes += [f"{n} x {message}" for message, n in errors.most_common(3)]
    if notes:
        lines += ["", "Notes"] + [f"{INDENT}- {n}" for n in notes]
    if any(r.status in ("error", "skipped") for r in run.results):
        lines += ["", f"{INDENT}Run the same command again to retry; finished answers are cached."]
    return "\n".join(lines)


def _label(model: str, effort: str | None) -> str:
    return f"{model} @ {effort}" if effort else model
