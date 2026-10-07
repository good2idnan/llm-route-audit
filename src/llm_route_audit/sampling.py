"""Pick a replay sample that keeps every task type represented."""

import math
import random
from collections import Counter, defaultdict

from llm_route_audit.analyze import UNLABELLED
from llm_route_audit.records import LogRecord


def allocate(group_sizes: dict[str, int], size: int) -> dict[str, int]:
    """Split `size` across groups in proportion to their volume, at least one per group.

    Uses largest remainders, so the quotas add up to exactly `size` (or to the number of
    groups, if that is larger, since every group gets one).
    """
    total = sum(group_sizes.values())
    size = max(size, len(group_sizes))
    ideal = {g: size * n / total for g, n in group_sizes.items()}
    quota = {g: min(n, max(1, math.floor(ideal[g]))) for g, n in group_sizes.items()}

    while sum(quota.values()) < size:
        open_groups = [g for g in quota if quota[g] < group_sizes[g]]
        if not open_groups:
            break
        quota[max(open_groups, key=lambda g: (ideal[g] - quota[g], g))] += 1
    while sum(quota.values()) > size:
        shrinkable = [g for g in quota if quota[g] > 1]
        quota[max(shrinkable, key=lambda g: (quota[g] - ideal[g], g))] -= 1
    return quota


def sessions(records: list[LogRecord]) -> list[list[LogRecord]]:
    """Records grouped by session_id, steps in time order. Records without one stand alone."""
    grouped: dict[str, list[LogRecord]] = defaultdict(list)
    units: list[list[LogRecord]] = []
    for record in records:
        if record.session_id:
            grouped[record.session_id].append(record)
        else:
            units.append([record])
    units += [sorted(steps, key=lambda r: (r.timestamp, r.id)) for steps in grouped.values()]
    return units


def session_type(steps: list[LogRecord]) -> str:
    """A session's task type: the one most of its steps carry (the earliest step breaks ties)."""
    counts = Counter(r.task_type or UNLABELLED for r in steps)
    best = max(counts.values())
    return next(t for t in (r.task_type or UNLABELLED for r in steps) if counts[t] == best)


def _session_sample(records: list[LogRecord], size: int, seed: int) -> list[LogRecord]:
    """Whole sessions, spread across session types, until each type's step quota is met."""
    groups: dict[str, list[list[LogRecord]]] = defaultdict(list)
    for steps in sessions(records):
        groups[session_type(steps)].append(steps)

    rng = random.Random(seed)
    quotas = allocate({g: sum(map(len, units)) for g, units in groups.items()}, size)
    picked: list[LogRecord] = []
    for group in sorted(groups):
        units = sorted(groups[group], key=lambda steps: steps[0].id)
        rng.shuffle(units)
        taken = 0
        for steps in units:
            if taken >= quotas[group]:
                break
            picked += steps
            taken += len(steps)
    return sorted(picked, key=lambda r: r.id)


def stratified_sample(records: list[LogRecord], size: int, seed: int = 0) -> list[LogRecord]:
    """About `size` records spread across task types. The same seed gives the same sample.

    Agent logs (records with a session_id) are sampled as whole sessions, so every picked
    session can be followed step by step.
    """
    if size >= len(records):
        return sorted(records, key=lambda r: r.id)
    if any(r.session_id for r in records):
        return _session_sample(records, size, seed)

    groups: dict[str, list[LogRecord]] = defaultdict(list)
    for record in records:
        groups[record.task_type or UNLABELLED].append(record)

    rng = random.Random(seed)
    quotas = allocate({g: len(rs) for g, rs in groups.items()}, size)
    picked: list[LogRecord] = []
    for group in sorted(groups):
        members = sorted(groups[group], key=lambda r: r.id)
        picked += rng.sample(members, quotas[group])
    return sorted(picked, key=lambda r: r.id)
