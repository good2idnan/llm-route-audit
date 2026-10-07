from collections import Counter

from llm_route_audit.records import LogRecord
from llm_route_audit.sampling import allocate, session_type, sessions, stratified_sample


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


def agent_steps(sessions: dict[str, tuple[str, int]]) -> list[LogRecord]:
    """{session_id: (task, number of steps)}"""
    out = []
    for session, (task, steps) in sessions.items():
        for step in range(steps):
            out.append(
                LogRecord.model_validate(
                    {
                        "id": f"{session}-{step}",
                        "timestamp": f"2026-10-01T00:00:0{step}Z",
                        "model": "m",
                        "task_type": task,
                        "session_id": session,
                        "prompt": "p",
                        "response": "r",
                    }
                )
            )
    return out


def test_agent_logs_are_sampled_as_whole_sessions():
    logs = agent_steps({f"r{i}": ("refund", 4) for i in range(10)})
    logs += agent_steps({f"s{i}": ("site", 2) for i in range(10)})
    picked = stratified_sample(logs, 12, seed=3)
    by_session = Counter(r.session_id for r in picked)
    assert all(by_session[s] == (4 if s.startswith("r") else 2) for s in by_session)
    assert Counter(r.task_type for r in picked) == {"refund": 8, "site": 4}
    assert picked == stratified_sample(logs, 12, seed=3)


def test_session_type_follows_most_steps():
    mixed = agent_steps({"a": ("plan", 1)}) + agent_steps({"a": ("act", 3)})[1:]
    assert session_type(sorted(mixed, key=lambda r: r.id)) == "act"
    assert [len(steps) for steps in sessions(mixed + records({"x": 2}))] == [1, 1, 3]
