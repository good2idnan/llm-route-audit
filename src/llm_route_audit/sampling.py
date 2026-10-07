"""Pick a replay sample that keeps every task type represented."""

import math
import random
from collections import defaultdict

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


def stratified_sample(records: list[LogRecord], size: int, seed: int = 0) -> list[LogRecord]:
    """About `size` records spread across task types. The same seed gives the same sample."""
    if size >= len(records):
        return sorted(records, key=lambda r: r.id)

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
