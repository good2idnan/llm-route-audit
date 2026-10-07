"""Render a routing Report as terminal text or as a self-contained HTML page."""

import html

from routeaudit.display import INDENT, pct, table, usd
from routeaudit.grading.grade import ORIGINAL
from routeaudit.report import OptionStats, Report, Strategy


def _rate(option: OptionStats) -> str:
    return "-" if option.pass_rate is None else pct(option.pass_rate)


def _range(option: OptionStats) -> str:
    interval = option.interval
    return "-" if interval is None else f"{interval[0]:.0%}-{interval[1]:.0%}"


def _ratio(value: float | None) -> str:
    return "-" if value is None else pct(value)


def _use(task_choice: OptionStats, original_model: str) -> str:
    if task_choice.label == ORIGINAL:
        return f"keep {original_model}"
    return task_choice.label


def _savings_line(report: Report) -> str:
    line = f"Projected savings with the policy: {pct(report.savings_share)} of current spend"
    if report.monthly_cost is not None:
        line += f" (about {usd(report.monthly_cost * report.savings_share)} a month)"
    return line + "."


def _needs_more_data(report: Report) -> bool:
    return any(t.reason.startswith("not enough data") for t in report.tasks)


def _agreement_line(report: Report) -> str:
    return (
        f"Judge consistency: same verdict in both orders for {report.judge_agreed} of "
        f"{report.judged} answers ({pct(report.judge_agreed / report.judged)}). "
        "Low consistency means the judge's verdicts are noisy."
    )


def render_text(report: Report, html_path: str | None = None) -> str:
    out = [
        "Routing report",
        f"{INDENT}Rule: use the cheapest option that keeps {report.target:.0%} of the original's "
        f"pass rate, with at least {report.min_samples} graded answers per option.",
        "",
        "Recommendation by task type (most expensive first)",
    ]
    out += table(
        ["Task", "Traffic", "Use", "Why"],
        [
            [t.task, pct(t.request_share), _use(t.choice, t.original.model), t.reason]
            for t in report.tasks
        ],
        text_columns=4,
    )
    out += ["", "Details (pass rate with its 95% range, cost against the original)"]
    rows = []
    for t in report.tasks:
        for option in [t.original, *t.options]:
            rows.append(
                [
                    t.task,
                    option.label,
                    f"{option.graded}",
                    _rate(option),
                    _range(option),
                    _ratio(option.cost_ratio),
                ]
            )
    out += table(["Task", "Option", "Graded", "Pass", "95% range", "Cost"], rows, text_columns=2)
    out += ["", "Whole workload (weighted by traffic)"]
    out += table(
        ["Strategy", "Quality", "Cost vs now", "Coverage"],
        [
            [s.name, _ratio(s.quality), _ratio(s.cost_ratio), pct(s.coverage)]
            for s in report.strategies
        ],
    )
    out += ["", _savings_line(report)]
    if report.judged:
        out.append(_agreement_line(report))
    if _needs_more_data(report):
        out.append(
            "Next: replay and grade more requests (for example --sample 100) so each task "
            f"reaches {report.min_samples} graded answers."
        )
    if html_path:
        out.append(f"HTML report saved to {html_path}")
    return "\n".join(out)


# --- HTML -------------------------------------------------------------------------------------

CSS = """
:root{--bg:#f5f7f9;--surface:#fff;--sunk:#eaeef2;--fg:#15202b;--muted:#56636f;--line:#d5dde5;
--accent:#1d5c96;--accent-soft:#e2edf7;--good:#2c7a4b;--good-soft:#e1f2e8;--warn:#a05f00;
--warn-soft:#fbefd9;color-scheme:light}
@media (prefers-color-scheme:dark){:root{--bg:#0e141a;--surface:#151d25;--sunk:#1b252f;--fg:#e3e9ef;
--muted:#94a2af;--line:#2a3743;--accent:#7db3e6;--accent-soft:#18293a;--good:#62be88;
--good-soft:#14281c;--warn:#e2a24a;--warn-soft:#2e2312;color-scheme:dark}}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--fg);font:15px/1.6 system-ui,"Segoe UI",Roboto,sans-serif}
main{max-width:1040px;margin:0 auto;padding:32px 20px 64px;display:flex;flex-direction:column;gap:28px}
h1{font-size:1.9rem;margin:0;line-height:1.2}h2{font-size:1.2rem;margin:0}
.sub{color:var(--muted);margin:4px 0 0}
.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:12px}
.card{background:var(--surface);border:1px solid var(--line);border-radius:10px;padding:16px;
display:flex;flex-direction:column;gap:4px}
.card .label{font-size:.75rem;letter-spacing:.06em;text-transform:uppercase;color:var(--muted)}
.card .value{font-size:1.6rem;font-weight:650;font-variant-numeric:tabular-nums}
.card .note{font-size:.85rem;color:var(--muted)}
section{display:flex;flex-direction:column;gap:12px}
.panel{background:var(--surface);border:1px solid var(--line);border-radius:10px;padding:16px;overflow-x:auto}
table{width:100%;border-collapse:collapse;font-size:.9rem}
th{text-align:left;font-size:.72rem;letter-spacing:.06em;text-transform:uppercase;color:var(--muted);
padding:8px 10px;border-bottom:1px solid var(--line);white-space:nowrap}
td{padding:8px 10px;border-bottom:1px solid var(--line);vertical-align:top}
tr:last-child td{border-bottom:0}
td.num{text-align:right;font-variant-numeric:tabular-nums;white-space:nowrap}
.pill{display:inline-block;padding:1px 8px;border-radius:999px;font-size:.78rem;white-space:nowrap}
.pill.keep{background:var(--sunk);color:var(--muted)}.pill.switch{background:var(--good-soft);color:var(--good)}
.callout{border-left:4px solid var(--warn);background:var(--warn-soft);padding:12px 16px;border-radius:6px}
svg text{font-family:system-ui,"Segoe UI",sans-serif;fill:var(--fg)}
.axis{stroke:var(--muted)}.grid{stroke:var(--line)}
.dot{fill:var(--surface);stroke:var(--muted);stroke-width:2.5}.dot.policy{fill:var(--accent);stroke:var(--accent)}
.dot.current{fill:var(--muted)}.tick{fill:var(--muted);font-size:12px}
footer{color:var(--muted);font-size:.82rem}
"""


def _e(text: object) -> str:
    return html.escape(str(text))


def _chart(strategies: list[Strategy]) -> str:
    """Cost (x) against quality (y), one numbered marker per strategy."""
    points = [s for s in strategies if s.quality is not None and s.cost_ratio is not None]
    x_max = max([1.2, *(s.cost_ratio * 1.1 for s in points)])
    left, right, top, bottom = 60, 600, 24, 290

    def x(v: float) -> float:
        return left + (right - left) * v / x_max

    def y(v: float) -> float:
        return bottom - (bottom - top) * v

    parts = [
        '<svg viewBox="0 0 640 340" width="100%" style="max-width:640px;min-width:420px" '
        'role="img" aria-label="Cost against quality for each strategy">'
    ]
    for q in (0, 0.25, 0.5, 0.75, 1):
        parts.append(
            f'<line class="grid" x1="{left}" x2="{right}" y1="{y(q):.1f}" y2="{y(q):.1f}"/>'
        )
        parts.append(
            f'<text class="tick" x="{left - 8}" y="{y(q) + 4:.1f}" text-anchor="end">{q:.0%}</text>'
        )
    step = 0.25 if x_max <= 1.5 else 0.5
    tick = 0.0
    while tick <= x_max + 1e-9:
        parts.append(
            f'<text class="tick" x="{x(tick):.1f}" y="{bottom + 18}" text-anchor="middle">{tick:.0%}</text>'
        )
        tick += step
    parts.append(f'<line class="axis" x1="{left}" x2="{right}" y1="{bottom}" y2="{bottom}"/>')
    parts.append(f'<line class="axis" x1="{left}" x2="{left}" y1="{top}" y2="{bottom}"/>')
    parts.append(
        f'<text class="tick" x="{(left + right) / 2}" y="{bottom + 40}" text-anchor="middle">'
        "Cost compared with today</text>"
    )
    parts.append(
        f'<text class="tick" x="16" y="{(top + bottom) / 2}" text-anchor="middle" '
        f'transform="rotate(-90 16 {(top + bottom) / 2})">Quality (pass rate)</text>'
    )
    for number, s in enumerate(strategies, start=1):
        if s.quality is None or s.cost_ratio is None:
            continue
        kind = "policy" if s is strategies[-1] else "current" if number == 1 else ""
        cx, cy = x(s.cost_ratio), y(s.quality)
        parts.append(f'<circle class="dot {kind}" cx="{cx:.1f}" cy="{cy:.1f}" r="9"/>')
        colour = "var(--surface)" if kind else "var(--fg)"
        parts.append(
            f'<text x="{cx:.1f}" y="{cy + 4:.1f}" text-anchor="middle" font-size="11" '
            f'font-weight="600" style="fill:{colour}">{number}</text>'
        )
    parts.append("</svg>")
    return "".join(parts)


def render_html(report: Report, source: str) -> str:
    policy = report.policy
    switched = sum(t.choice.label != ORIGINAL for t in report.tasks)
    monthly = (
        f"about {usd(report.monthly_cost * report.savings_share)} a month"
        if report.monthly_cost is not None
        else "of current spend"
    )
    cards = [
        ("Projected savings", pct(report.savings_share), monthly),
        (
            "Quality with the policy",
            _ratio(policy.quality),
            f"today: {_ratio(report.strategies[0].quality)}",
        ),
        (
            "Tasks switched",
            f"{switched} of {len(report.tasks)}",
            "the rest keep their current model",
        ),
    ]
    if report.judged:
        cards.append(
            (
                "Judge consistency",
                pct(report.judge_agreed / report.judged),
                f"same verdict both ways on {report.judge_agreed} of {report.judged}",
            )
        )
    out = [
        "<!doctype html><html lang='en'><head><meta charset='utf-8'>",
        "<meta name='viewport' content='width=device-width, initial-scale=1'>",
        "<title>LLM route audit report</title>",
        f"<style>{CSS}</style></head><body><main>",
        "<header><h1>LLM route audit report</h1>",
        f"<p class='sub'>{_e(source)} &middot; generated "
        f"{report.generated_at:%Y-%m-%d %H:%M} UTC</p></header>",
        "<div class='cards'>",
    ]
    for label, value, note in cards:
        out.append(
            f"<div class='card'><span class='label'>{_e(label)}</span>"
            f"<span class='value'>{_e(value)}</span><span class='note'>{_e(note)}</span></div>"
        )
    out.append("</div>")
    if _needs_more_data(report):
        out.append(
            "<p class='callout'><strong>Not enough data for some tasks.</strong> A cheaper model "
            f"is only recommended after at least {report.min_samples} graded answers for that "
            "task. Replay and grade more requests (for example <code>--sample 100</code>) "
            "for firm recommendations.</p>"
        )

    out.append("<section><h2>Recommendation by task type</h2><div class='panel'><table>")
    out.append("<tr><th>Task</th><th>Traffic</th><th>Use</th><th>Why</th></tr>")
    for t in report.tasks:
        kind = "keep" if t.choice.label == ORIGINAL else "switch"
        out.append(
            f"<tr><td>{_e(t.task)}</td><td class='num'>{pct(t.request_share)}</td>"
            f"<td><span class='pill {kind}'>{_e(_use(t.choice, t.original.model))}</span></td>"
            f"<td>{_e(t.reason)}</td></tr>"
        )
    out.append("</table></div></section>")

    out.append("<section><h2>Cost and quality of each strategy</h2><div class='panel'>")
    out.append(_chart(report.strategies))
    out.append(
        "<table><tr><th>#</th><th>Strategy</th><th>Quality</th><th>Cost vs today</th>"
        "<th>Coverage</th></tr>"
    )
    for number, s in enumerate(report.strategies, start=1):
        out.append(
            f"<tr><td class='num'>{number}</td><td>{_e(s.name)}</td>"
            f"<td class='num'>{_ratio(s.quality)}</td><td class='num'>{_ratio(s.cost_ratio)}</td>"
            f"<td class='num'>{pct(s.coverage)}</td></tr>"
        )
    out.append("</table></div></section>")

    out.append("<section><h2>Details</h2><div class='panel'><table>")
    out.append(
        "<tr><th>Task</th><th>Option</th><th>Graded</th><th>Pass</th><th>95% range</th>"
        "<th>Cost vs original</th></tr>"
    )
    for t in report.tasks:
        for option in [t.original, *t.options]:
            out.append(
                f"<tr><td>{_e(t.task)}</td><td>{_e(option.label)}</td>"
                f"<td class='num'>{option.graded}</td><td class='num'>{_rate(option)}</td>"
                f"<td class='num'>{_range(option)}</td>"
                f"<td class='num'>{_ratio(option.cost_ratio)}</td></tr>"
            )
    out.append("</table></div></section>")
    out.append(
        "<footer>Quality is the share of answers that passed every exact check and that the AI "
        "judge rated at least as good as the original. Cost compares each option with the "
        "original on the same sampled requests. Small samples give wide 95% ranges; treat them "
        "as early signals.</footer>"
    )
    out.append("</main></body></html>")
    return "\n".join(out)
