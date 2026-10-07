"""Generate examples/agent_logs.jsonl: synthetic steps from a tool-using support agent.

Scenario: "Brightpath Hosting" (the same invented company as sample_logs.jsonl) runs a
support agent on claude-opus-5-5. Each session is one customer case; each logged record is
one step of it: the history so far (messages, tool calls, tool results) and what the model
did next (call tools, or answer). Two session types:

  refund_request  look up the customer and invoice, refund (or pass large refunds on), reply
  site_down       check the site, look at recent deploys, restart or open an incident, reply

All people, companies, domains and amounts are invented.

Run from the repo root:  uv run python scripts/make_agent_logs.py
"""

import json
import random
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

SEED = 7
MODEL = "claude-opus-5-5"
OUT = Path(__file__).resolve().parent.parent / "examples" / "agent_logs.jsonl"
START = datetime(2026, 9, 29, tzinfo=UTC)
COMPANY = "Brightpath Hosting"
SESSIONS_PER_TYPE = 20

rng = random.Random(SEED)

FIRST = ["Amara", "Lucas", "Priya", "Diego", "Hana", "Tom", "Fatima", "Oliver", "Mei", "Kofi"]
LAST = ["Okafor", "Brandt", "Nair", "Ramos", "Sato", "Becker", "Zahra", "Grant", "Lin", "Costa"]
WORDS = ["lumen", "harbor", "copper", "maple", "orbit", "fable", "quill", "tundra", "pixel"]
TLDS = ["shop", "io", "co", "store", "app"]


def tool(name: str, description: str, **properties: str) -> dict[str, Any]:
    return {
        "name": name,
        "description": description,
        "parameters": {
            "type": "object",
            "properties": {key: {"type": kind} for key, kind in properties.items()},
            "required": list(properties),
        },
    }


REFUND_TOOLS = [
    tool("find_customer", "Look up a customer by email.", email="string"),
    tool("get_invoices", "List a customer's recent invoices.", customer_id="string"),
    tool(
        "issue_refund",
        "Refund an invoice, fully or in part. Only for amounts up to $100.",
        invoice_id="string",
        amount="number",
        reason="string",
    ),
    tool(
        "escalate_to_billing",
        "Hand a refund over $100 to the billing team.",
        invoice_id="string",
        amount="number",
        note="string",
    ),
]
SITE_TOOLS = [
    tool("check_status", "HTTP status and response time of a customer's site.", domain="string"),
    tool("recent_deploys", "The customer's deploys in the last 24 hours.", domain="string"),
    tool(
        "restart_service",
        "Restart one service on the customer's server.",
        domain="string",
        service="string",
    ),
    tool(
        "open_incident",
        "Open an incident for the on-call engineer.",
        domain="string",
        severity="string",
        summary="string",
    ),
]

REFUND_SYSTEM = (
    f"You are the support agent for {COMPANY}. Use the tools to look up the customer and "
    "their invoices before acting. You may refund up to $100 yourself with issue_refund; "
    "larger refunds go to escalate_to_billing. Finish with a short, friendly reply to the "
    "customer, signed 'Brightpath Support'."
)
SITE_SYSTEM = (
    f"You are the on-call support agent for {COMPANY}. When a customer reports their site is "
    "down, check its status first, then recent deploys. Restart the web service if a restart "
    "is likely to fix it; otherwise open an incident (severity sev1 if the site is fully "
    "down, sev2 if degraded). Finish with a short reply to the customer, signed "
    "'Brightpath Support'."
)


def tokens(value: Any) -> int:
    text = value if isinstance(value, str) else json.dumps(value)
    return max(1, round(len(text) / 4))


class Session:
    """Builds the logged steps of one agent session."""

    def __init__(self, session_id: str, task: str, system: str, tools: list, user: str, when):
        self.id = session_id
        self.task = task
        self.tools = tools
        self.history: list[dict[str, Any]] = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        self.when = when
        self.records: list[dict[str, Any]] = []
        self.calls = 0
        self.cached = 0  # prompt tokens already in the cache from the previous step

    def _record(self, response: str, calls: list[dict[str, Any]] | None) -> None:
        step = len(self.records) + 1
        prompt = tokens(self.history) + tokens(self.tools)
        record: dict[str, Any] = {
            "id": f"{self.id}_s{step}",
            "timestamp": self.when.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "model": MODEL,
            "task_type": self.task,
            "session_id": self.id,
            "messages": json.loads(json.dumps(self.history)),
            "tools": self.tools,
            "response": response,
            # Prompt caching as an agent loop uses it: each step reads the history so far
            # from cache and writes the new part for the next step.
            "input_tokens": 3,
            "cache_read_tokens": self.cached,
            "cache_write_tokens": prompt - self.cached - 3,
            "output_tokens": tokens(response) + (tokens(calls) if calls else 0) + 20,
            "latency_ms": rng.randint(1800, 5200),
        }
        if calls:
            record["response_tool_calls"] = calls
        self.records.append(record)
        self.cached = prompt - 3
        self.when += timedelta(seconds=rng.randint(3, 9))

    def act(self, text: str, *calls: tuple[str, dict[str, Any], Any]) -> None:
        """One step that calls tools; each call is (name, arguments, result)."""
        made = []
        for name, arguments, _ in calls:
            self.calls += 1
            made.append(
                {"id": f"toolu_{self.id}_{self.calls}", "name": name, "arguments": arguments}
            )
        self._record(text, made)
        self.history.append({"role": "assistant", "content": text, "tool_calls": made})
        for call, (name, _, result) in zip(made, calls, strict=True):
            self.history.append(
                {
                    "role": "tool",
                    "tool_call_id": call["id"],
                    "name": name,
                    "content": json.dumps(result),
                }
            )

    def answer(self, text: str) -> None:
        self._record(text, None)


def refund_session(n: int, when: datetime) -> Session:
    first, last = rng.choice(FIRST), rng.choice(LAST)
    email = f"{first.lower()}.{last.lower()}@{rng.choice(WORDS)}mail.com"
    amount = rng.choice([12.0, 19.0, 29.0, 49.0, 49.0, 79.0, 180.0, 240.0])
    plan = "annual Business plan" if amount > 100 else "Pro plan"
    invoice = f"INV-{rng.randint(20000, 29999)}"
    customer_id = f"cus_{rng.randint(1000, 9999)}"
    reason = rng.choice(
        ["charged twice", "cancelled within the refund window", "billed after downgrade"]
    )
    user = (
        f"From: {first} {last} <{email}>\n"
        f"Hi, I was {reason} for the {plan} (${amount:.0f}, invoice {invoice}). "
        "Can you refund it please?"
    )
    s = Session(f"refund_{n:03d}", "refund_request", REFUND_SYSTEM, REFUND_TOOLS, user, when)
    s.act(
        "",
        (
            "find_customer",
            {"email": email},
            {"customer_id": customer_id, "name": f"{first} {last}", "plan": plan},
        ),
    )
    s.act(
        "",
        (
            "get_invoices",
            {"customer_id": customer_id},
            [
                {"invoice_id": invoice, "amount": amount, "status": "paid"},
                {
                    "invoice_id": f"INV-{rng.randint(10000, 19999)}",
                    "amount": amount,
                    "status": "paid",
                },
            ],
        ),
    )
    if amount <= 100:
        s.act(
            f"The invoice is ${amount:.0f}, within my limit, so I'll refund it.",
            (
                "issue_refund",
                {"invoice_id": invoice, "amount": amount, "reason": reason},
                {"refund_id": f"re_{rng.randint(100000, 999999)}", "status": "succeeded"},
            ),
        )
        s.answer(
            f"Hi {first},\n\nSorry about that. I've refunded ${amount:.0f} for invoice {invoice}; "
            "it should reach your card within 5-10 business days.\n\nBrightpath Support"
        )
    else:
        s.act(
            f"${amount:.0f} is over my $100 limit, so this goes to billing.",
            (
                "escalate_to_billing",
                {
                    "invoice_id": invoice,
                    "amount": amount,
                    "note": f"Customer reports being {reason}.",
                },
                {"ticket": f"BIL-{rng.randint(1000, 9999)}", "eta_hours": 24},
            ),
        )
        s.answer(
            f"Hi {first},\n\nThanks for flagging this. I've passed your ${amount:.0f} refund for "
            f"invoice {invoice} to our billing team, who will confirm within one business day."
            "\n\nBrightpath Support"
        )
    return s


def site_session(n: int, when: datetime) -> Session:
    first = rng.choice(FIRST)
    domain = f"{rng.choice(WORDS)}{rng.choice(WORDS)}.{rng.choice(TLDS)}"
    fixed_by_restart = rng.random() < 0.6
    status = 502 if fixed_by_restart else rng.choice([500, 503])
    user = f"From: {first}\nOur site {domain} is down since {rng.randint(1, 11)}am, customers see errors. Help!"
    s = Session(f"site_{n:03d}", "site_down", SITE_SYSTEM, SITE_TOOLS, user, when)
    s.act("", ("check_status", {"domain": domain}, {"status": status, "response_ms": None}))
    deploys = (
        []
        if fixed_by_restart
        else [{"at": "09:12", "by": "ci", "message": "upgrade php 8.4 -> 8.5"}]
    )
    s.act("", ("recent_deploys", {"domain": domain}, deploys))
    if fixed_by_restart:
        s.act(
            "No recent deploys, and a 502 usually means the web service hung. Restarting it.",
            (
                "restart_service",
                {"domain": domain, "service": "web"},
                {"ok": True, "status_after": 200},
            ),
        )
        s.answer(
            f"Hi {first},\n\n{domain} is back up. The web service had stopped responding, so "
            "we restarted it. We'll keep an eye on it.\n\nBrightpath Support"
        )
    else:
        s.act(
            "A deploy just before the outage changed the PHP version; a restart won't fix that.",
            (
                "open_incident",
                {
                    "domain": domain,
                    "severity": "sev1",
                    "summary": f"{domain} returns {status} after PHP upgrade deploy at 09:12.",
                },
                {"incident": f"INC-{rng.randint(100, 999)}", "on_call": "paged"},
            ),
        )
        s.answer(
            f"Hi {first},\n\nYour site is failing after this morning's PHP upgrade. Our on-call "
            "engineer has been paged and will roll it back or fix it shortly. We'll update you "
            "here.\n\nBrightpath Support"
        )
    return s


def main() -> None:
    records = []
    when = START
    for n in range(1, SESSIONS_PER_TYPE + 1):
        for make in (refund_session, site_session):
            when += timedelta(minutes=rng.randint(4, 40))
            records += make(n, when).records
    records.sort(key=lambda r: r["timestamp"])
    with OUT.open("w", encoding="utf-8", newline="\n") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    sessions = len({r["session_id"] for r in records})
    print(f"wrote {len(records)} steps from {sessions} sessions to {OUT}")


if __name__ == "__main__":
    main()
