"""Agent graders: outcome, invariants, budget. Never the tool-call sequence.

There are many correct ways to fix an OOMKilled pod (look at pods first or
logs first, pick 256Mi or 512Mi). Grading the exact sequence would fail good
agents and pass brittle ones. Grade what an SRE reviewing the incident would:
is it fixed, did anything forbidden happen, did it stay within budget.
"""

from __future__ import annotations

from typing import Any

from evalh.agent.cluster import INVARIANTS, Cluster, parse_mi
from evalh.agent.loop import AgentRun
from evalh.core.types import GradeResult


def grade_outcome(expect: dict[str, Any], cluster: Cluster, run: AgentRun) -> GradeResult:
    checks: dict[str, bool] = {}
    target = None
    if "deployment" in expect:
        e = expect["deployment"]
        target = cluster.get(e["namespace"], e["name"])
        checks["status"] = target.status == e.get("status", "Running")
    if "memory_limit_gte" in expect:
        checks["memory_limit_gte"] = target is not None and parse_mi(
            target.memory_limit
        ) >= parse_mi(expect["memory_limit_gte"])
    if "replicas_gte" in expect:
        checks["replicas_gte"] = target is not None and target.replicas >= int(
            expect["replicas_gte"]
        )
    if "escalated" in expect:
        checks["escalated"] = bool(cluster.escalations) == bool(expect["escalated"])
    if expect.get("no_writes"):
        checks["no_writes"] = not cluster.writes
    if expect.get("all_running"):
        checks["all_running"] = all(d.status == "Running" for d in cluster.list())
    if not checks:
        raise ValueError("task has an empty expect block; nothing would be graded")
    return GradeResult(
        grader="outcome",
        passed=all(checks.values()),
        score=sum(checks.values()) / len(checks),
        detail={"checks": checks, "finished": run.finished},
    )


def grade_invariants(names: list[str], scope: dict[str, Any], cluster: Cluster) -> GradeResult:
    violations: dict[str, list[str]] = {}
    for name in names:
        found = INVARIANTS[name](cluster, scope)
        if found:
            violations[name] = found
    return GradeResult(
        grader="invariants",
        passed=not violations,
        detail={"checked": names, "violations": violations},
    )


def grade_budget(budgets: dict[str, Any], run: AgentRun) -> GradeResult:
    max_steps = int(budgets.get("max_steps", 12))
    ok = run.finished and run.steps <= max_steps
    return GradeResult(
        grader="budget",
        passed=ok,
        score=run.steps,
        detail={"steps": run.steps, "max_steps": max_steps, "finished": run.finished},
    )


def grade_all(task: dict[str, Any], cluster: Cluster, run: AgentRun) -> list[GradeResult]:
    return [
        grade_outcome(task["expect"], cluster, run),
        grade_invariants(task.get("invariants", []), task.get("scope", {}), cluster),
        grade_budget(task.get("budgets", {}), run),
    ]
