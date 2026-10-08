"""Route health over time: every `monitor` and `outcomes` run is saved, and `status` shows
each route's history as a table and an offline HTML page with one chart per route.

History is a JSONL file, one line per run:
    {"kind": "monitor" | "outcomes", "at": ISO time, "logs": ..., "policy": ...,
     "routes": [{"task", "model", "status", "rate", "low", "high", "count", "target", ...}]}
"""

import html
import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ATTENTION = {"ALERT", "REVERT"}
CAUTION = {"WARN", "WATCH"}
KIND_NAMES = {"monitor": "quality vs reference", "outcomes": "good outcomes"}


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def monitor_snapshot(
    health: list[Any], tolerance: float, logs: str, policy: str, at: str | None = None
) -> dict[str, Any]:
    """One `monitor` run: each route's pass rate against the reference model."""
    routes = []
    for h in health:
        interval = h.interval
        routes.append(
            {
                "task": h.task,
                "model": h.route.label,
                "status": h.status,
                "rate": h.pass_rate,
                "low": interval[0] if interval else None,
                "high": interval[1] if interval else None,
                "count": h.checked,
                "target": max(0.0, (h.route.expected_pass_rate or 0.0) - tolerance),
                "note": h.note,
            }
        )
    return {"kind": "monitor", "at": at or _now(), "logs": logs, "policy": policy, "routes": routes}


def outcomes_snapshot(report: Any, logs: str, policy: str, at: str | None = None) -> dict[str, Any]:
    """One `outcomes` run: each switched route's good-outcome rate and the replaced model's."""
    routes = []
    for r in report.routes:
        interval = r.routed.interval
        routes.append(
            {
                "task": r.task,
                "model": r.route.label,
                "status": r.status,
                "rate": r.routed.rate,
                "low": interval[0] if interval else None,
                "high": interval[1] if interval else None,
                "count": r.routed.total,
                "target": r.original.rate,
                "target_count": r.original.total,
                "note": r.note,
            }
        )
    return {
        "kind": "outcomes",
        "at": at or _now(),
        "logs": logs,
        "policy": policy,
        "routes": routes,
    }


def append_history(path: str | Path, snapshot: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(snapshot, ensure_ascii=False) + "\n")


def load_history(path: str | Path) -> list[dict[str, Any]]:
    """Every saved run, oldest first. Lines that can't be read are skipped."""
    path = Path(path)
    if not path.exists():
        return []
    runs = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            run = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(run, dict) and run.get("kind") in KIND_NAMES and run.get("at"):
            runs.append(run)
    return sorted(runs, key=lambda r: r["at"])


@dataclass
class Point:
    at: datetime
    status: str
    rate: float | None
    low: float | None
    high: float | None
    count: int
    target: float | None
    model: str
    note: str = ""


@dataclass
class Track:
    """One route's history for one kind of check."""

    task: str
    kind: str
    points: list[Point] = field(default_factory=list)

    @property
    def latest(self) -> Point:
        return self.points[-1]

    @property
    def change(self) -> float | None:
        """Change in rate since the previous run, in percentage points."""
        rated = [p for p in self.points if p.rate is not None]
        if len(rated) < 2:
            return None
        return (rated[-1].rate - rated[-2].rate) * 100  # type: ignore[operator]

    @property
    def needs_attention(self) -> bool:
        return self.latest.status in ATTENTION

    def to_dict(self) -> dict[str, Any]:
        return {
            "task": self.task,
            "check": self.kind,
            "latest_status": self.latest.status,
            "latest_rate": self.latest.rate,
            "change_points": self.change,
            "points": [
                {
                    "at": p.at.isoformat(),
                    "status": p.status,
                    "rate": p.rate,
                    "low": p.low,
                    "high": p.high,
                    "count": p.count,
                    "target": p.target,
                    "model": p.model,
                }
                for p in self.points
            ],
        }


def build_tracks(history: list[dict[str, Any]]) -> list[Track]:
    tracks: dict[tuple[str, str], Track] = {}
    for run in history:
        at = datetime.fromisoformat(run["at"])
        if at.tzinfo is None:
            at = at.replace(tzinfo=UTC)
        for r in run.get("routes") or []:
            key = (r["task"], run["kind"])
            track = tracks.setdefault(key, Track(r["task"], run["kind"]))
            track.points.append(
                Point(
                    at=at,
                    status=r.get("status", "?"),
                    rate=r.get("rate"),
                    low=r.get("low"),
                    high=r.get("high"),
                    count=r.get("count", 0),
                    target=r.get("target"),
                    model=r.get("model", ""),
                    note=r.get("note", ""),
                )
            )
    order = {kind: i for i, kind in enumerate(KIND_NAMES)}
    return sorted(
        tracks.values(),
        key=lambda t: (not t.needs_attention, t.task, order.get(t.kind, 9)),
    )


def _pct(value: float | None) -> str:
    return "-" if value is None else f"{value:.0%}"


def _range(p: Point) -> str:
    return "" if p.low is None or p.high is None else f" ({p.low:.0%}-{p.high:.0%})"


def _change(track: Track) -> str:
    change = track.change
    return "-" if change is None else f"{change:+.0f} pts"


def render_status_text(tracks: list[Track], source: str, html_path: str | None = None) -> str:
    from llm_route_audit.display import INDENT, table

    if not tracks:
        return (
            f"No route history yet in {source}. Run `monitor` or `outcomes`; each run is saved "
            "there automatically."
        )
    times = [p.at for t in tracks for p in t.points]
    runs = len({(t.kind, p.at) for t in tracks for p in t.points})
    lines = [
        f"Route health over time: {source} ({runs} runs, {min(times):%Y-%m-%d} to "
        f"{max(times):%Y-%m-%d})",
        "",
    ]
    lines += table(
        ["Route", "Check", "Runs", "Latest", "Rate (95% range)", "Target", "Change"],
        [
            [
                t.task,
                KIND_NAMES[t.kind],
                str(len(t.points)),
                t.latest.status,
                _pct(t.latest.rate) + _range(t.latest),
                _pct(t.latest.target),
                _change(t),
            ]
            for t in tracks
        ],
        numeric_columns={2, 4, 5, 6},
    )
    attention = [
        f"{t.task} ({KIND_NAMES[t.kind]}: {t.latest.status})" for t in tracks if t.needs_attention
    ]
    lines.append("")
    lines.append(
        f"Needs attention: {', '.join(attention)}." if attention else "No route needs attention."
    )
    if html_path:
        lines.append(f"{INDENT}Status page saved to {html_path}")
    return "\n".join(lines)


# --- HTML ------------------------------------------------------------------------------------

EXTRA_CSS = """
:root{--bad:#b42318;--bad-soft:#fdecea;--band:rgba(29,92,150,.14)}
@media (prefers-color-scheme:dark){:root{--bad:#f97066;--bad-soft:#3a1714;--band:rgba(125,179,230,.18)}}
.pill.ok{background:var(--good-soft);color:var(--good)}.pill.wait{background:var(--sunk);color:var(--muted)}
.pill.caution{background:var(--warn-soft);color:var(--warn)}.pill.bad{background:var(--bad-soft);color:var(--bad)}
.track{display:flex;flex-direction:column;gap:10px}
.track h3{font-size:1rem;margin:0;display:flex;gap:10px;align-items:center;flex-wrap:wrap}
.track .meta{color:var(--muted);font-size:.85rem;margin:0}
.chart{max-width:680px}
.chart text{font-size:11.5px;fill:var(--muted)}
.chart .grid{stroke:var(--line);stroke-width:1}
.chart .axis{stroke:var(--line);stroke-width:1}
.chart .band{fill:var(--band)}
.chart .series{fill:none;stroke:var(--accent);stroke-width:2}
.chart .series.base{stroke:var(--muted)}
.chart .target{stroke:var(--muted);stroke-width:1}
.chart .dot{fill:var(--accent);stroke:var(--surface);stroke-width:2}
.chart .dot.base{fill:var(--muted)}
.chart .hit{fill:transparent;cursor:default}
.chart .label{fill:var(--fg);font-size:11.5px}
.legend{display:flex;gap:16px;font-size:.85rem;color:var(--muted);flex-wrap:wrap}
.legend span::before{content:"";display:inline-block;width:14px;height:2px;margin-right:6px;
vertical-align:middle;background:var(--accent)}
.legend span.base::before{background:var(--muted)}
"""


def _pill(status: str) -> str:
    kind = (
        "bad" if status in ATTENTION else "caution" if status in CAUTION
        else "ok" if status == "OK" else "wait"
    )  # fmt: skip
    return f"<span class='pill {kind}'>{html.escape(status)}</span>"


def _ticks(low: float) -> list[float]:
    step = 0.1 if 1 - low <= 0.5 else 0.25
    ticks, value = [], 1.0
    while value >= low - 1e-9:
        ticks.append(round(value, 4))
        value -= step
    return sorted(ticks)


def _chart(track: Track) -> str:
    """Rate over time on one axis: the 95% range as a band, the target (the audited floor,
    or the replaced model's good-outcome rate) as a line or a second series."""
    points = [p for p in track.points if p.rate is not None]
    if not points:
        return "<p class='meta'>No rated runs yet.</p>"
    outcomes = track.kind == "outcomes"
    values = [p.rate for p in points] + [p.low for p in points if p.low is not None]
    values += [p.target for p in points if p.target is not None]
    low = max(0.0, (int(min(values) * 10) / 10) - 0.05)  # type: ignore[arg-type]
    low = min(low, 0.8)
    width, height = 680, 230
    left, right, top, bottom = 44, 120, 14, 196

    first, last = points[0].at, points[-1].at
    span = (last - first).total_seconds()

    def x(at: datetime) -> float:
        if span == 0:
            return (left + width - right) / 2
        return left + (width - right - left) * (at - first).total_seconds() / span

    def y(v: float) -> float:
        return bottom - (bottom - top) * (v - low) / (1 - low)

    parts = [
        f"<svg class='chart' viewBox='0 0 {width} {height}' width='100%' role='img' "
        f"aria-label='{html.escape(track.task)}: {KIND_NAMES[track.kind]} over time'>"
    ]
    for tick in _ticks(low):
        parts.append(
            f"<line class='grid' x1='{left}' x2='{width - right}' y1='{y(tick):.1f}' y2='{y(tick):.1f}'/>"
        )
        parts.append(
            f"<text x='{left - 6}' y='{y(tick) + 4:.1f}' text-anchor='end'>{tick:.0%}</text>"
        )
    parts.append(
        f"<line class='axis' x1='{left}' x2='{width - right}' y1='{bottom}' y2='{bottom}'/>"
    )
    date_ticks = [points[0], points[-1]] if span else [points[0]]
    if len(points) >= 3 and span:
        date_ticks.insert(1, points[len(points) // 2])
    for p in date_ticks:
        parts.append(
            f"<text x='{x(p.at):.1f}' y='{bottom + 18}' text-anchor='middle'>{p.at:%b %d}</text>"
        )

    banded = [p for p in points if p.low is not None and p.high is not None]
    if len(banded) >= 2:
        upper = " ".join(f"{x(p.at):.1f},{y(p.high):.1f}" for p in banded)  # type: ignore[arg-type]
        lower = " ".join(f"{x(p.at):.1f},{y(p.low):.1f}" for p in reversed(banded))  # type: ignore[arg-type]
        parts.append(f"<polygon class='band' points='{upper} {lower}'/>")

    targets = [p for p in points if p.target is not None]
    if outcomes and targets:
        if len(targets) >= 2:
            path = " ".join(f"{x(p.at):.1f},{y(p.target):.1f}" for p in targets)  # type: ignore[arg-type]
            parts.append(f"<polyline class='series base' points='{path}'/>")
        for p in targets:
            parts.append(
                f"<circle class='dot base' cx='{x(p.at):.1f}' cy='{y(p.target):.1f}' r='4'/>"
            )  # type: ignore[arg-type]
        end = targets[-1]
        parts.append(
            f"<text class='label' x='{width - right + 8}' y='{y(end.target) + 4:.1f}'>"  # type: ignore[arg-type]
            f"replaced model {end.target:.0%}</text>"
        )
    elif targets:
        level = targets[-1].target
        parts.append(
            f"<line class='target' x1='{left}' x2='{width - right}' y1='{y(level):.1f}' y2='{y(level):.1f}'/>"
        )  # type: ignore[arg-type]
        parts.append(
            f"<text class='label' x='{width - right + 8}' y='{y(level) + 4:.1f}'>"  # type: ignore[arg-type]
            f"audited floor {level:.0%}</text>"
        )

    if len(points) >= 2:
        path = " ".join(f"{x(p.at):.1f},{y(p.rate):.1f}" for p in points)  # type: ignore[arg-type]
        parts.append(f"<polyline class='series' points='{path}'/>")
    for p in points:
        cx, cy = x(p.at), y(p.rate)  # type: ignore[arg-type]
        parts.append(f"<circle class='dot' cx='{cx:.1f}' cy='{cy:.1f}' r='4.5'/>")
    end = points[-1]
    label_y = y(end.rate)  # type: ignore[arg-type]
    if targets and abs(label_y - y(targets[-1].target)) < 14:  # type: ignore[arg-type]
        label_y += -14 if end.rate >= (targets[-1].target or 0) else 14  # type: ignore[operator]
    parts.append(
        f"<text class='label' x='{width - right + 8}' y='{label_y + 4:.1f}'>"
        f"{'routed' if outcomes else 'now'} {end.rate:.0%}</text>"
    )
    for p in points:  # generous hover targets, drawn last so they sit on top
        tip = f"{p.at:%Y-%m-%d %H:%M} UTC - {p.status}: {_pct(p.rate)}{_range(p)} on {p.count}" + (
            f"; target {_pct(p.target)}" if p.target is not None else ""
        )
        parts.append(
            f"<circle class='hit' cx='{x(p.at):.1f}' cy='{y(p.rate):.1f}' r='12'>"  # type: ignore[arg-type]
            f"<title>{html.escape(tip)}</title></circle>"
        )
    parts.append("</svg>")
    return "".join(parts)


def render_status_html(tracks: list[Track], source: str) -> str:
    from llm_route_audit.report_view import CSS

    e = html.escape
    attention = [t for t in tracks if t.needs_attention]
    caution = [t for t in tracks if t.latest.status in CAUTION]
    times = [p.at for t in tracks for p in t.points]
    cards = [
        (
            "Routes watched",
            str(len({t.task for t in tracks})),
            f"{len(tracks)} checks with history",
        ),
        ("Need attention", str(len(attention)), "ALERT or REVERT on the latest run"),
        ("Worth a look", str(len(caution)), "WARN or WATCH on the latest run"),
        (
            "Last check",
            f"{max(times):%b %d}" if times else "-",
            f"{max(times):%H:%M} UTC" if times else "",
        ),
    ]
    out = [
        "<!doctype html><html lang='en'><head><meta charset='utf-8'>",
        "<meta name='viewport' content='width=device-width, initial-scale=1'>",
        "<title>Route health</title>",
        f"<style>{CSS}{EXTRA_CSS}</style></head><body><main>",
        "<header><h1>Route health</h1>",
        f"<p class='sub'>{e(source)} &middot; generated {datetime.now(UTC):%Y-%m-%d %H:%M} UTC</p></header>",
        "<div class='cards'>",
    ]
    for label, value, note in cards:
        out.append(
            f"<div class='card'><span class='label'>{e(label)}</span>"
            f"<span class='value'>{e(value)}</span><span class='note'>{e(note)}</span></div>"
        )
    out.append("</div>")
    if not tracks:
        out.append(
            "<p class='callout'>No history yet. Run <code>llm-route-audit monitor</code> or "
            "<code>llm-route-audit outcomes</code>; each run is saved automatically.</p>"
        )

    out.append("<section><h2>Now</h2><div class='panel'><table>")
    out.append(
        "<tr><th>Route</th><th>Check</th><th>Latest</th><th>Rate</th><th>95% range</th>"
        "<th>Target</th><th>Change</th><th>Why</th></tr>"
    )
    for t in tracks:
        p = t.latest
        range_text = "-" if p.low is None else f"{p.low:.0%}-{p.high:.0%}"
        out.append(
            f"<tr><td>{e(t.task)}</td><td>{e(KIND_NAMES[t.kind])}</td><td>{_pill(p.status)}</td>"
            f"<td class='num'>{_pct(p.rate)}</td><td class='num'>{range_text}</td>"
            f"<td class='num'>{_pct(p.target)}</td><td class='num'>{_change(t)}</td>"
            f"<td>{e(p.note)}</td></tr>"
        )
    out.append("</table></div></section>")

    for t in tracks:
        p = t.latest
        out.append("<section class='track'>")
        out.append(f"<h3>{e(t.task)} &middot; {e(KIND_NAMES[t.kind])} {_pill(p.status)}</h3>")
        out.append(
            f"<p class='meta'>Routed to {e(p.model)}. "
            + (
                "Pass rate of production answers graded against the model the route replaced; "
                "the line is the audited floor."
                if t.kind == "monitor"
                else "Share of good outcomes on the routed model and on the model it replaced."
            )
            + " The shaded band is the 95% range.</p>"
        )
        if t.kind == "outcomes":
            out.append(
                "<div class='legend'><span>routed model</span>"
                "<span class='base'>model it replaced</span></div>"
            )
        out.append(f"<div class='panel'>{_chart(t)}<table>")
        out.append(
            "<tr><th>Run</th><th>Status</th><th>Rate</th><th>95% range</th><th>On</th>"
            "<th>Target</th></tr>"
        )
        for q in reversed(t.points):
            range_text = "-" if q.low is None else f"{q.low:.0%}-{q.high:.0%}"
            out.append(
                f"<tr><td>{q.at:%Y-%m-%d %H:%M}</td><td>{_pill(q.status)}</td>"
                f"<td class='num'>{_pct(q.rate)}</td><td class='num'>{range_text}</td>"
                f"<td class='num'>{q.count}</td><td class='num'>{_pct(q.target)}</td></tr>"
            )
        out.append("</table></div></section>")
    out.append(
        "<footer>Each <code>monitor</code> and <code>outcomes</code> run adds a point. Statuses: "
        "OK, WAIT (too little data), WARN or WATCH (lower, may be chance), ALERT or REVERT "
        "(95% sure it got worse). Hover a point for its numbers.</footer>"
    )
    out.append("</main></body></html>")
    return "\n".join(out)
