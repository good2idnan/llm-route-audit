from collections import Counter

from llm_route_audit.records import LogRecord
from llm_route_audit.sampling import allocate, stratified_sample


def records(counts: dict[str, int]) -> list[LogRecord]:
    out = []
    for task, n in counts.items():
        for i in range(n):
            out.append(
                LogRecord.model_validate(
                    {
                        "id": f"{task}-{i:03d}",
                        "timestamp": "2026-10-01T00:00:00Z",
                        "model": "m",
                        "task_type": task,
                        "prompt": "p",
                        "response": "r",
                    }
                )
            )
    return out


def test_allocation_is_proportional_and_exact():
    assert allocate({"a": 60, "b": 30, "c": 10}, 10) == {"a": 6, "b": 3, "c": 1}
    assert sum(allocate({"a": 45, "b": 45, "c": 40, "d": 40, "e": 30}, 50).values()) == 50


def test_every_group_gets_at_least_one():
    quotas = allocate({"big": 990, "rare": 10}, 10)
    assert quotas["rare"] == 1
    assert sum(quotas.values()) == 10


def test_sample_is_stratified_and_deterministic():
    logs = records({"a": 60, "b": 30, "c": 10})
    first = stratified_sample(logs, 10, seed=7)
    assert Counter(r.task_type for r in first) == {"a": 6, "b": 3, "c": 1}
    assert [r.id for r in first] == [r.id for r in stratified_sample(logs, 10, seed=7)]
    assert [r.id for r in first] != [r.id for r in stratified_sample(logs, 10, seed=8)]


def test_asking_for_more_than_available_returns_everything():
    logs = records({"a": 3})
    assert len(stratified_sample(logs, 50)) == 3
