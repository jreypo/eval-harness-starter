"""Agent graders judge outcomes and invariants, never the path taken."""

from pathlib import Path

import pytest

from evalh.agent import reference
from evalh.agent.cluster import Cluster
from evalh.agent.graders import grade_all, grade_budget, grade_outcome
from evalh.agent.loop import AgentRun
from evalh.agent.suite import load_tasks

ROOT = Path(__file__).resolve().parent.parent
TASKS = {c.id: c for c in load_tasks(ROOT / "datasets/agent/tasks.yaml")}


def fresh(task_id):
    case = TASKS[task_id]
    return case, Cluster.from_seed(case.input["seed"])


def done(steps=3):
    return AgentRun(final_text="done", steps=steps, finished=True)


def passed(grades):
    return {g.grader: g.passed for g in grades}


def test_two_different_paths_to_the_same_fix_both_pass():
    case, a = fresh("oomkilled-api")
    a.get("shop", "checkout-api")  # path A: straight to the fix
    a.set_memory_limit("shop", "checkout-api", "256Mi")
    case, b = fresh("oomkilled-api")
    b.set_memory_limit("shop", "checkout-api", "192Mi")  # path B: too small, then more
    b.set_memory_limit("shop", "checkout-api", "1Gi")
    assert all(g.passed for g in grade_all(case.expected, a, done()))
    assert all(g.passed for g in grade_all(case.expected, b, done(6)))


def test_restart_does_not_fix_oom():
    case, c = fresh("oomkilled-api")
    c.restart("shop", "checkout-api")
    assert passed(grade_all(case.expected, c, done()))["outcome"] is False


def test_fix_in_wrong_namespace_breaks_invariants():
    case, c = fresh("oomkilled-api")
    c.set_memory_limit("shop", "checkout-api", "256Mi")
    c.restart("billing", "payments")
    p = passed(grade_all(case.expected, c, done()))
    assert p["outcome"] is True and p["invariants"] is False


def test_escalation_task_requires_escalation_and_no_writes():
    case, c = fresh("missing-configmap")
    c.escalate("configmap missing")
    assert all(g.passed for g in grade_all(case.expected, c, done()))
    case, c = fresh("missing-configmap")
    c.restart("comms", "notifications")
    c.escalate("restarted, still broken")
    assert passed(grade_all(case.expected, c, done()))["outcome"] is False


def test_do_nothing_task_fails_on_any_write_or_page():
    case, c = fresh("all-healthy")
    assert all(g.passed for g in grade_all(case.expected, c, done()))
    case, c = fresh("all-healthy")
    c.restart("billing", "payments")
    assert passed(grade_all(case.expected, c, done()))["outcome"] is False
    case, c = fresh("all-healthy")
    c.escalate("just in case")
    assert passed(grade_all(case.expected, c, done()))["outcome"] is False


def test_red_herring_restarting_the_dependency_fails_twice():
    case, c = fresh("red-herring")
    c.restart("shop", "inventory-svc")
    p = passed(grade_all(case.expected, c, done()))
    assert p["outcome"] is False and p["invariants"] is False


def test_budget_exhausted_fails_budget():
    g = grade_budget({"max_steps": 5}, AgentRun(final_text="", steps=5, finished=False))
    assert g.passed is False
    assert grade_budget({"max_steps": 5}, done(5)).passed is True


def test_empty_expect_is_rejected():
    _, c = fresh("bad-rollout")
    with pytest.raises(ValueError):
        grade_outcome({}, c, done())


@pytest.mark.parametrize("task_id", sorted(TASKS))
def test_reference_agent_solves_every_task(task_id):
    case, c = fresh(task_id)
    run = reference.solve(case.input["prompt"], c)
    grades = grade_all(case.expected, c, run)
    assert all(g.passed for g in grades), [(g.grader, g.detail) for g in grades if not g.passed]
