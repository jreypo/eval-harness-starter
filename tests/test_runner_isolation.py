"""Trials must not leak state into each other.

If trial 3 sees what trial 2 did, k trials are not k samples, and pass^k is
fiction. These tests run a deliberately destructive "agent" concurrently and
check every trial still started from the pristine seed.
"""

import threading
import time

import pytest

from evalh.core.runner import run_trials
from evalh.core.types import Case, GradeResult, Trial
from evalh.errors import HarnessError

SEED = {"deployments": [{"name": "api", "replicas": 2, "labels": {"tier": "web"}}]}


def make_case(cid="c1"):
    return Case(id=cid, suite="regression", input={"seed": SEED}, expected={})


def vandal(case: Case, trial_index: int) -> Trial:
    seed = case.input["seed"]
    first_seen = repr(seed)
    time.sleep(0.001)  # widen the window for a race if state were shared
    seed["deployments"][0]["replicas"] = 100 + trial_index
    seed["deployments"][0]["labels"]["tier"] = f"vandalised-{trial_index}"
    seed["deployments"].append({"name": f"junk-{trial_index}"})
    return Trial(
        case_id=case.id,
        trial_index=trial_index,
        output=first_seen,
        transcript=[],
        grades=[GradeResult("saw_seed", passed=first_seen == repr(SEED))],
    )


def test_every_trial_starts_from_the_seed():
    cases = [make_case(f"c{i}") for i in range(3)]
    trials = run_trials(cases, vandal, k=5, concurrency=4)
    assert len(trials) == 15
    assert all(t.passed for t in trials), [t.output for t in trials if not t.passed]


def test_seed_is_never_mutated():
    case = make_case()
    run_trials([case], vandal, k=5, concurrency=4)
    assert case.input["seed"] == SEED
    assert SEED["deployments"][0]["replicas"] == 2


def test_order_is_stable_under_concurrency():
    order = []
    lock = threading.Lock()

    def slow_first(case, i):
        if i == 0:
            time.sleep(0.01)
        with lock:
            order.append((case.id, i))
        return Trial(case.id, i, None, [], [GradeResult("g", True)])

    trials = run_trials([make_case("a"), make_case("b")], slow_first, k=3, concurrency=6)
    assert [(t.case_id, t.trial_index) for t in trials] == sorted(order)


def test_ordinary_exceptions_are_recorded_not_raised():
    def boom(case, i):
        raise ValueError("tool blew up")

    (trial,) = run_trials([make_case()], boom, k=1)
    assert not trial.passed
    assert "tool blew up" in trial.error


def test_harness_errors_abort_the_run():
    def broken(case, i):
        raise HarnessError("fixture missing")

    with pytest.raises(HarnessError):
        run_trials([make_case()], broken, k=2)


def test_trial_without_grades_does_not_pass():
    assert not Trial("c", 0, None, [], []).passed
    assert not Trial("c", 0, None, [], [GradeResult("judge", None)]).passed


# ---- the real agent path ---------------------------------------------------------


class GreedyVandalModel:
    """Looks at the cluster, then trashes it differently in every trial."""

    name = "vandal"

    def complete(self, request, *, trial_index=0):
        from evalh.providers.base import Response

        turn = sum(1 for m in request.messages if m["role"] == "assistant")
        if turn == 0:
            calls = [("get_deployments", {})]
        elif turn == 1:
            calls = [
                (
                    "scale",
                    {
                        "namespace": "shop",
                        "deployment": "checkout-api",
                        "replicas": 50 + trial_index,
                    },
                ),
                (
                    "set_memory_limit",
                    {"namespace": "shop", "deployment": "checkout-api", "memory": "4Gi"},
                ),
                ("restart", {"namespace": "billing", "deployment": "payments"}),
            ]
        else:
            return Response([{"type": "text", "text": "done"}], "end_turn")
        blocks = [
            {"type": "tool_use", "id": f"t{turn}{i}", "name": n, "input": a}
            for i, (n, a) in enumerate(calls)
        ]
        return Response(blocks, "tool_use")


def test_agent_trials_each_get_a_fresh_cluster():
    from pathlib import Path

    from evalh.agent.suite import load_tasks, make_run_one

    root = Path(__file__).resolve().parent.parent
    (case,) = [c for c in load_tasks(root / "datasets/agent/tasks.yaml") if c.id == "oomkilled-api"]
    config = {"model": "m", "provider": "stub", "fixtures": "unused"}
    run_one = make_run_one(config, agent="model", llm=GreedyVandalModel())
    trials = run_trials([case], run_one, k=5, concurrency=4)

    # What each trial saw on its first look must be identical: the pristine seed.
    first_looks = {t.transcript[2]["content"][0]["content"] for t in trials}
    assert len(first_looks) == 1
    assert '"memory_limit": "128Mi"' in first_looks.pop()
    # And each trial's own writes are only its own (3 writes, its own replica count).
    for t in trials:
        assert len(t.output["writes"]) == 3
        assert t.output["writes"][0]["args"]["replicas"] == 50 + t.trial_index
