"""Generate examples/sample_logs.jsonl: 200 synthetic requests from one SaaS support team.

Scenario: "Brightpath Hosting" sends every AI request to claude-opus-5-5. That is the
setup llm-route-audit is built to audit. All people, companies, domains and amounts are invented.

Run from the repo root:  uv run python scripts/make_sample_logs.py
"""

import json
import random
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

SEED = 42
MODEL = "claude-opus-5-5"
OUT = Path(__file__).resolve().parent.parent / "examples" / "sample_logs.jsonl"
START = datetime(2026, 9, 28, tzinfo=UTC)
COMPANY = "Brightpath Hosting"

rng = random.Random(SEED)

FIRST = [
    "Amara",
    "Lucas",
    "Priya",
    "Diego",
    "Hana",
    "Tom",
    "Fatima",
    "Oliver",
    "Mei",
    "Samuel",
    "Ines",
    "Kofi",
    "Elena",
    "Rahul",
    "Zoe",
    "Mateo",
]
LAST = [
    "Okafor",
    "Brandt",
    "Nair",
    "Ramos",
    "Sato",
    "Becker",
    "Zahra",
    "Grant",
    "Lin",
    "Adeyemi",
    "Costa",
    "Mensah",
    "Petrova",
    "Iyer",
    "Fischer",
    "Silva",
]
WORDS = [
    "lumen",
    "harbor",
    "copper",
    "maple",
    "orbit",
    "fable",
    "quill",
    "tundra",
    "pixel",
    "ember",
]
TLDS = ["shop", "io", "co", "store", "app"]


def person() -> tuple[str, str]:
    return rng.choice(FIRST), rng.choice(LAST)


def domain() -> str:
    return f"{rng.choice(WORDS)}{rng.choice(WORDS)}.{rng.choice(TLDS)}"


def day_in_september() -> date:
    return date(2026, 9, rng.randint(1, 26))


def tokens(text: str) -> int:
    return max(1, round(len(text) / 4))


# --- classify_ticket -----------------------------------------------------------------------

CLASSIFY_SYSTEM = (
    f"You are the support triage assistant for {COMPANY}, a web hosting company. "
    "Classify the ticket. Reply with JSON only, in the form "
    '{"category": "...", "priority": "..."}. '
    "category is one of: billing, technical, account, sales, abuse. "
    "priority is one of: low, normal, high, urgent."
)

TICKETS = [
    (
        "Charged twice this month",
        "Hi, my card was charged $49 twice on {date} for the Pro plan. Please refund the duplicate.",
        "billing",
        "high",
    ),
    (
        "Site down - 502 errors",
        "Our store {domain} has returned 502 Bad Gateway since {hour}:00 UTC. We're losing orders. "
        "Please help ASAP.",
        "technical",
        "urgent",
    ),
    (
        "Can't log in after password reset",
        "I reset my password but the login page keeps saying 'invalid credentials'. "
        "Account email is {email}.",
        "account",
        "high",
    ),
    (
        "Question about team pricing",
        "We're a team of {seats} developers. Do you offer volume discounts or annual billing for "
        "the Business plan?",
        "sales",
        "normal",
    ),
    (
        "Phishing page hosted on your network",
        "A page at {domain}/login imitates our bank's sign-in page. Please take it down.",
        "abuse",
        "urgent",
    ),
    (
        "How do I add an SSH key?",
        "Where in the dashboard can I add a second SSH key for my colleague?",
        "technical",
        "low",
    ),
    (
        "Update billing address",
        "Our company moved. Please update the billing address on future invoices to "
        "{number} Harbour Street, Leeds.",
        "billing",
        "low",
    ),
    (
        "SSL certificate expired",
        "The certificate for {domain} expired this morning and auto-renew didn't run. "
        "Visitors now see a security warning.",
        "technical",
        "high",
    ),
    (
        "Transfer account ownership",
        "Our previous admin left the company. How do we transfer ownership of the account to "
        "{full_name}?",
        "account",
        "normal",
    ),
    (
        "Dedicated server instead of VPS?",
        "Our traffic doubled. Would a dedicated server be cheaper than running three VPS instances?",
        "sales",
        "normal",
    ),
    (
        "Our IP is on a spam blocklist",
        "We got a notice that our server IP is on a blocklist for sending spam. "
        "We haven't sent any bulk email.",
        "abuse",
        "high",
    ),
    (
        "Invoice missing VAT number",
        "Invoice INV-2026-{number} is missing our VAT number GB{vat}. Can you reissue it?",
        "billing",
        "normal",
    ),
    (
        "Database backups failing",
        "Nightly backups for our Postgres instance have failed {number_small} days in a row with "
        "'disk quota exceeded'.",
        "technical",
        "high",
    ),
    (
        "Turn on 2FA for the team",
        "How do I turn on two-factor authentication for all users on our team account?",
        "account",
        "low",
    ),
    (
        "Cancel subscription",
        "Please cancel my Starter plan at the end of this billing cycle. Thanks.",
        "billing",
        "normal",
    ),
]


def classify_ticket(n: int) -> dict:
    subject, body, category, priority = rng.choice(TICKETS)
    first, last = person()
    body = body.format(
        date=day_in_september().strftime("%d %B"),
        domain=domain(),
        hour=rng.randint(0, 23),
        email=f"{first.lower()}@{domain()}",
        seats=rng.choice([12, 25, 40, 80]),
        number=rng.randint(10000, 99999),
        number_small=rng.randint(2, 6),
        full_name=f"{rng.choice(FIRST)} {rng.choice(LAST)}",
        vat=rng.randint(100000000, 999999999),
    )
    user = f"Ticket #{n}\nFrom: {first} {last}\nSubject: {subject}\n\n{body}"
    response = json.dumps({"category": category, "priority": priority})
    return {
        "task_type": "classify_ticket",
        "system": CLASSIFY_SYSTEM,
        "user": user,
        "response": response,
        "thinking": rng.randint(0, 40),
    }


# --- extract_invoice -----------------------------------------------------------------------

EXTRACT_SYSTEM = (
    "Extract the invoice fields. Return JSON only with these keys: vendor, invoice_number, "
    "invoice_date (YYYY-MM-DD), due_date (YYYY-MM-DD), currency (ISO code), total (number), "
    "line_items (list of objects with description, quantity, unit_price)."
)

VENDORS = [
    (
        "Kestrel Office Supply",
        "USD",
        "$",
        0.08,
        [
            ("Printer paper A4, box of 5 reams", 24.50),
            ("Toner cartridge, black", 89.00),
            ("Desk organiser", 15.75),
            ("Whiteboard markers, pack of 12", 11.20),
        ],
    ),
    (
        "Nordlicht Print GmbH",
        "EUR",
        "€",
        0.19,
        [
            ("Business cards, 500 pcs", 39.00),
            ("Roll-up banner 85x200", 129.00),
            ("Flyers A5, 1000 pcs", 74.50),
        ],
    ),
    (
        "Harbor Coffee Co.",
        "USD",
        "$",
        0.0,
        [
            ("Espresso beans, 1 kg", 28.00),
            ("Oat milk, case of 12", 31.40),
            ("Paper cups 8oz, sleeve of 50", 6.90),
        ],
    ),
    (
        "Atlas Cloud Services",
        "USD",
        "$",
        0.0,
        [
            ("Object storage, 2 TB-month", 46.00),
            ("Load balancer hours", 0.03),
            ("Support plan, monthly", 150.00),
        ],
    ),
    (
        "Lumen Electrical Ltd",
        "GBP",
        "£",
        0.20,
        [
            ("Rack PDU, 16A", 212.00),
            ("Cat6 patch cable, 2 m", 3.85),
            ("Server labour, per hour", 65.00),
        ],
    ),
]


def extract_invoice(n: int) -> dict:
    vendor, currency, symbol, tax_rate, catalogue = rng.choice(VENDORS)
    issued = day_in_september()
    due = issued + timedelta(days=30)
    number = f"INV-2026-{rng.randint(10000, 99999)}"
    items = []
    for desc, price in rng.sample(catalogue, k=rng.randint(1, len(catalogue))):
        qty = rng.randint(1, 400) if price < 1 else rng.randint(1, 12)
        items.append({"description": desc, "quantity": qty, "unit_price": price})
    subtotal = round(sum(i["quantity"] * i["unit_price"] for i in items), 2)
    tax = round(subtotal * tax_rate, 2)
    total = round(subtotal + tax, 2)

    if rng.random() < 0.5:
        lines = "\n".join(
            f"{i['quantity']:>4}  {i['description']:<36} {i['unit_price']:>9.2f} "
            f"{i['quantity'] * i['unit_price']:>10.2f}"
            for i in items
        )
        text = (
            f"{vendor.upper()}\nINVOICE No. {number}\n"
            f"Date: {issued:%d %b %Y}    Due: {due:%d %b %Y}\nBill to: {COMPANY}\n\n"
            f" Qty  {'Description':<36} {'Unit':>9} {'Amount':>10}\n{lines}\n\n"
            f"Subtotal {symbol}{subtotal:,.2f}\nTax ({tax_rate:.0%}) {symbol}{tax:,.2f}\n"
            f"TOTAL DUE {currency} {total:,.2f}"
        )
    else:
        lines = "\n".join(
            f"- {i['quantity']} x {i['description']} @ {symbol}{i['unit_price']:,.2f}"
            for i in items
        )
        text = (
            f"From {vendor} to {COMPANY}.\nInvoice #{number}, issued {issued.isoformat()}, "
            f"payable by {due.isoformat()}.\n\nItems:\n{lines}\n\n"
            f"Net {symbol}{subtotal:,.2f}, tax {symbol}{tax:,.2f}. "
            f"Please pay {symbol}{total:,.2f} ({currency})."
        )
    response = json.dumps(
        {
            "vendor": vendor,
            "invoice_number": number,
            "invoice_date": issued.isoformat(),
            "due_date": due.isoformat(),
            "currency": currency,
            "total": total,
            "line_items": items,
        }
    )
    return {
        "task_type": "extract_invoice",
        "system": EXTRACT_SYSTEM,
        "user": f"Invoice text:\n\n{text}",
        "response": response,
        "thinking": rng.randint(20, 120),
    }


# --- draft_reply ---------------------------------------------------------------------------

REPLY_SYSTEM = (
    f"You are a support agent for {COMPANY}. Write a short, friendly email reply to the "
    "customer using the internal notes. Sign as 'Brightpath Support'. Never promise a refund "
    "above $100 yourself; those go to the billing team."
)


def _reply(first: str, body: str) -> str:
    return f"Hi {first},\n\n{body}\n\nBest regards,\nBrightpath Support"


def draft_reply(n: int) -> dict:
    first, last = person()
    kind = rng.randrange(7)
    if kind == 0:
        amt, days = rng.choice([19, 49, 79]), rng.randint(6, 12)
        issued = day_in_september()
        eta = issued + timedelta(days=14)
        msg = (
            f"I asked for a refund of ${amt} {days} days ago and still haven't received it. "
            "This is getting frustrating."
        )
        note = f"Refund of ${amt} issued on {issued:%d %B}. Banks take 5-10 business days."
        body = (
            f"I'm sorry for the wait. I checked your account and the refund of ${amt} was "
            f"issued on {issued:%d %B}. Banks usually take 5-10 business days to show it, so it "
            f"should reach your statement by {eta:%d %B}.\n\nIf it hasn't appeared by then, "
            "reply to this email and I'll chase our payments team directly."
        )
    elif kind == 1:
        plan = rng.choice(["Starter", "Pro", "Business"])
        msg = "We're moving our WordPress site from another host. Can you help migrate it?"
        note = f"Free migration on Pro and Business. Customer is on {plan}. Paid migration is $49."
        if plan == "Starter":
            body = (
                "Happy to help. On the Starter plan our team can migrate the site for a one-off "
                "fee of $49, or you can upgrade to Pro, where migration is free. Let me know "
                "which you prefer and I'll book it in."
            )
        else:
            body = (
                f"Good news: migration is free on your {plan} plan. Reply with a convenient time "
                "and temporary access to your current host, and our team will move the site "
                "with no downtime."
            )
    elif kind == 2:
        hours, credit = rng.choice([2, 3, 5]), rng.choice([10, 25])
        region = rng.choice(["Frankfurt", "Virginia", "Singapore"])
        msg = (
            f"Your outage yesterday took our shop offline for {hours} hours. "
            "What are you doing about it?"
        )
        note = (
            f"Incident INC-{rng.randint(1000, 9999)}: storage controller failure in {region}. "
            f"SLA credit {credit}% of monthly fee, applied automatically."
        )
        body = (
            f"I'm sorry about yesterday's outage. A storage controller failed in our {region} "
            f"data centre, and we know {hours} hours offline hurts a shop. We've replaced the "
            "hardware and added a second controller so a single failure can't take sites down "
            f"again.\n\nA {credit}% credit has been applied to your next invoice under our SLA."
        )
    elif kind == 3:
        amt, days = rng.choice([180, 240, 360]), rng.randint(5, 25)
        msg = f"I want a full refund for the annual plan (${amt}). I've barely used it."
        note = (
            f"Annual plan bought {days} days ago. Full refund allowed within 30 days. "
            "Over $100: escalate to billing."
        )
        body = (
            f"Thanks for letting us know. Your plan was purchased {days} days ago, which is "
            "within our 30-day refund window. Because of the amount, I've passed your request "
            "to our billing team, who will confirm the refund within 2 business days."
        )
    elif kind == 4:
        msg = "Just wanted to say your team fixed our DNS issue really fast. Thanks!"
        note = "No action needed."
        body = (
            "Thank you for the kind words. I'll share them with the team. "
            "We're always here if you need anything else."
        )
    elif kind == 5:
        msg = "The password reset email never arrives. I've tried four times."
        note = "Reset emails bounced: recipient mailbox full."
        body = (
            "Our logs show the reset emails were sent but bounced because your mailbox is "
            "full. Once you free up some space, request the reset again and it should arrive "
            "within a minute."
        )
    else:
        old = rng.choice([29, 49, 99])
        extra = rng.randint(1, 4)
        new = old + extra * 3
        msg = f"Why did my bill go up from ${old} to ${new}?"
        note = f"Customer added {extra} extra IP address(es) at $3/month each on {day_in_september():%d %B}."
        body = (
            f"The difference comes from the {extra} extra IP address"
            f"{'es' if extra > 1 else ''} added to your account this month, at $3 each per "
            f"month. Your plan price itself hasn't changed. If you no longer need them, you can "
            "remove them under Network > IP addresses."
        )
    user = f"Customer: {first} {last}\nMessage: {msg}\n\nInternal notes: {note}"
    return {
        "task_type": "draft_reply",
        "system": REPLY_SYSTEM,
        "user": user,
        "response": _reply(first, body),
        "thinking": rng.randint(30, 160),
    }


# --- summarize_call ------------------------------------------------------------------------

SUMMARY_SYSTEM = (
    "Summarize this sales call transcript for the CRM. Write exactly 3 bullet points, then an "
    "'Action items' list with an owner for each item."
)
REPS = ["Jordan", "Alex", "Sam"]


def summarize_call(n: int) -> dict:
    rep = rng.choice(REPS)
    first, last = person()
    company = f"{rng.choice(WORDS).title()}{rng.choice(['Labs', 'Retail', 'Studio', 'Health'])}"
    sites, budget = rng.choice([4, 12, 30]), rng.choice([800, 1500, 4000])
    deadline = rng.choice(["end of October", "mid November", "before Black Friday"])
    competitor = rng.choice(["AWS", "DigitalOcean", "their current agency host"])
    need = rng.choice(["daily backups", "a 99.95% uptime SLA", "EU-only data storage"])
    transcript = "\n".join(
        [
            f"{rep} (Brightpath): Thanks for joining, {first}. What prompted the call?",
            f"{first} ({company}): We run {sites} client sites on {competitor} and the bills keep "
            "growing. We want something simpler.",
            f"{rep} (Brightpath): Understood. What's your monthly budget?",
            f"{first} ({company}): Around ${budget} a month. We need to move {deadline}.",
            f"{rep} (Brightpath): That fits our Business plan. Anything you can't live without?",
            f"{first} ({company}): {need[0].upper() + need[1:]}. Our biggest client requires it.",
            f"{rep} (Brightpath): We can do that. I'll send a quote and a migration plan.",
            f"{first} ({company}): Great. I'll need to loop in our CTO before signing.",
        ]
    )
    response = (
        f"- {company} hosts {sites} client sites on {competitor} and wants lower, simpler costs.\n"
        f"- Budget is about ${budget}/month; migration must happen {deadline}.\n"
        f"- Hard requirement: {need}, needed by their largest client.\n\n"
        "Action items:\n"
        f"- Send quote for the Business plan and a migration plan (owner: {rep})\n"
        f"- Confirm {need} is covered in the contract (owner: {rep})\n"
        f"- Review with their CTO and reply (owner: {first} {last})"
    )
    return {
        "task_type": "summarize_call",
        "system": SUMMARY_SYSTEM,
        "user": f"Transcript:\n\n{transcript}",
        "response": response,
        "thinking": rng.randint(40, 200),
    }


# --- review_contract -----------------------------------------------------------------------

CONTRACT_SYSTEM = (
    "You are a contracts analyst helping a customer review a hosting agreement. For each "
    "clause: if it is risky for the customer, quote it, explain the risk and propose a "
    "revision; otherwise say it is standard. End with an overall risk rating: low, medium "
    "or high."
)

CLAUSES = [
    (
        "2.5",
        "Unilateral changes",
        "high",
        "Provider may modify these terms at any time by posting the updated terms on its website.",
        "The provider can change prices, limits or liability without telling you or asking.",
        "Provider must give 30 days' written notice of changes; Customer may terminate without "
        "penalty before they take effect.",
    ),
    (
        "4.2",
        "Automatic renewal",
        "medium",
        "This Agreement renews automatically for successive 36-month terms unless either party gives "
        "written notice at least 180 days before the end of the current term.",
        "A long renewal term combined with a six-month notice window is easy to miss and locks you "
        "in for three more years.",
        "Renew for 12-month terms with 30 days' notice; Provider sends a reminder 60 days before "
        "renewal.",
    ),
    (
        "5.4",
        "Price increases",
        "medium",
        "Provider may increase fees at its discretion upon renewal.",
        "There is no cap, so renewal prices can rise by any amount.",
        "Increases capped at 5% per year (or CPI, if lower) with 60 days' notice.",
    ),
    (
        "9.1",
        "Limitation of liability",
        "high",
        "Provider's total liability under this Agreement shall not exceed the fees paid by Customer "
        "in the one (1) month preceding the claim.",
        "One month of fees is far below the likely cost of data loss or a long outage.",
        "Cap at 12 months of fees, with no cap for data breaches, confidentiality breaches or gross "
        "negligence.",
    ),
    (
        "11.3",
        "Data on termination",
        "medium",
        "Upon termination, Provider may delete all Customer Data immediately.",
        "You could lose your data before you have a chance to export it.",
        "Provider keeps Customer Data for 30 days after termination for export, then deletes it and "
        "confirms deletion in writing.",
    ),
    (
        "7.1",
        "Service levels",
        None,
        "Provider will make the Services available 99.9% of each calendar month, excluding scheduled "
        "maintenance announced 48 hours in advance. Service credits apply as set out in Schedule B.",
        "",
        "",
    ),
    (
        "12.1",
        "Confidentiality",
        None,
        "Each party shall protect the other party's Confidential Information with at least "
        "reasonable care and use it only to perform this Agreement.",
        "",
        "",
    ),
    (
        "14.1",
        "Governing law",
        None,
        "This Agreement is governed by the laws of the State of Delaware.",
        "",
        "",
    ),
]


def review_contract(n: int) -> dict:
    picked = sorted(rng.sample(CLAUSES, k=rng.randint(3, 4)), key=lambda c: float(c[0]))
    text = "\n\n".join(f"{num} {title}. {quote}" for num, title, _, quote, _, _ in picked)
    parts = []
    for num, title, risk, quote, why, fix in picked:
        if risk:
            parts.append(
                f"**Clause {num}: {title}** (risk: {risk})\n> {quote}\n\n"
                f"Why it matters: {why}\nSuggested revision: {fix}"
            )
        else:
            parts.append(f"**Clause {num}: {title}**: standard wording, no change needed.")
    levels = {c[2] for c in picked}
    overall = "high" if "high" in levels else "medium" if "medium" in levels else "low"
    response = "\n\n".join(parts) + f"\n\nOverall risk rating: {overall}."
    return {
        "task_type": "review_contract",
        "system": CONTRACT_SYSTEM,
        "user": f"Clauses from the {COMPANY} Master Services Agreement:\n\n{text}",
        "response": response,
        "thinking": rng.randint(400, 1400),
    }


# --- assemble ------------------------------------------------------------------------------

MIX = [
    (classify_ticket, 45),
    (extract_invoice, 40),
    (draft_reply, 45),
    (summarize_call, 40),
    (review_contract, 30),
]


def main() -> None:
    drafts = []
    for make, count in MIX:
        for _ in range(count):
            drafts.append(make(rng.randint(10000, 99999)))

    when = sorted(
        START
        + timedelta(
            days=rng.randint(0, 6),
            hours=rng.randint(8, 18),
            minutes=rng.randint(0, 59),
            seconds=rng.randint(0, 59),
        )
        for _ in drafts
    )
    rng.shuffle(drafts)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", encoding="utf-8", newline="\n") as f:
        for i, (d, ts) in enumerate(zip(drafts, when, strict=True), start=1):
            input_tokens = tokens(d["system"]) + tokens(d["user"]) + 8
            output_tokens = tokens(d["response"]) + d["thinking"]
            latency = (900 + output_tokens * 22) * rng.uniform(0.85, 1.2)
            record = {
                "id": f"req_{i:04d}",
                "timestamp": ts.isoformat().replace("+00:00", "Z"),
                "model": MODEL,
                "task_type": d["task_type"],
                "messages": [
                    {"role": "system", "content": d["system"]},
                    {"role": "user", "content": d["user"]},
                ],
                "response": d["response"],
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "cache_read_tokens": 0,
                "latency_ms": round(latency),
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"Wrote {len(drafts)} records to {OUT}")


if __name__ == "__main__":
    main()
