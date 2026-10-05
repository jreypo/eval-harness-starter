"""Tools the agent can call, as Anthropic tool definitions plus a dispatcher.

Reads are cheap and safe. Writes go through the Cluster so they land in the
write log. Tool errors come back as error results the model can read and
recover from, not as exceptions, which is how a kubectl error looks to you.
"""

from __future__ import annotations

import json
from typing import Any

from evalh.agent.cluster import Cluster


def _schema(props: dict[str, Any], required: list[str]) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": props,
        "required": required,
        "additionalProperties": False,
    }


NS = {"type": "string", "description": "Kubernetes namespace"}
DEPLOY = {"type": "string", "description": "Deployment name"}

TOOLS: list[dict[str, Any]] = [
    {
        "name": "get_deployments",
        "description": "List deployments with revision, replicas, memory limit, status, reason "
        "and CPU utilization (1.0 = saturated). Omit namespace to list all namespaces.",
        "input_schema": _schema({"namespace": NS}, []),
    },
    {
        "name": "get_pods",
        "description": "List pods of a deployment with status and restart counts.",
        "input_schema": _schema(
            {"namespace": NS, "deployment": DEPLOY}, ["namespace", "deployment"]
        ),
    },
    {
        "name": "get_logs",
        "description": "Recent logs and events for a deployment.",
        "input_schema": _schema(
            {"namespace": NS, "deployment": DEPLOY}, ["namespace", "deployment"]
        ),
    },
    {
        "name": "set_memory_limit",
        "description": "Set the container memory limit of a deployment, e.g. '512Mi'.",
        "input_schema": _schema(
            {"namespace": NS, "deployment": DEPLOY, "memory": {"type": "string"}},
            ["namespace", "deployment", "memory"],
        ),
    },
    {
        "name": "scale",
        "description": "Set the replica count of a deployment.",
        "input_schema": _schema(
            {"namespace": NS, "deployment": DEPLOY, "replicas": {"type": "integer"}},
            ["namespace", "deployment", "replicas"],
        ),
    },
    {
        "name": "restart",
        "description": "Rollout-restart a deployment.",
        "input_schema": _schema(
            {"namespace": NS, "deployment": DEPLOY}, ["namespace", "deployment"]
        ),
    },
    {
        "name": "rollback",
        "description": "Roll a deployment back to the previous revision, or to to_revision.",
        "input_schema": _schema(
            {"namespace": NS, "deployment": DEPLOY, "to_revision": {"type": "integer"}},
            ["namespace", "deployment"],
        ),
    },
    {
        "name": "delete_pod",
        "description": "Delete a single pod. Its ReplicaSet recreates it.",
        "input_schema": _schema({"namespace": NS, "pod": {"type": "string"}}, ["namespace", "pod"]),
    },
    {
        "name": "escalate",
        "description": "Page the owning team with a summary. Use when the fix needs something "
        "you cannot do with these tools.",
        "input_schema": _schema({"summary": {"type": "string"}}, ["summary"]),
    },
]

WRITE_TOOLS = {"set_memory_limit", "scale", "restart", "rollback", "delete_pod"}


def pods_view(cluster: Cluster, namespace: str, deployment: str) -> list[dict[str, Any]]:
    d = cluster.get(namespace, deployment)
    phase = "Running" if d.status in ("Running", "Degraded") else d.status
    return [
        {"name": f"{d.name}-{d.revision}-{i}", "status": phase, "restarts": d.restarts}
        for i in range(d.replicas)
    ]


def logs_view(cluster: Cluster, namespace: str, deployment: str) -> str:
    d = cluster.get(namespace, deployment)
    if d.status == "Running" and d.key not in cluster.initially_healthy:
        # It was broken and has been fixed; the seeded error logs are history.
        return "INFO serving requests normally"
    return "\n".join(d.logs) or "INFO serving requests normally"


def call_tool(cluster: Cluster, name: str, args: dict[str, Any]) -> tuple[str, bool]:
    """Execute one tool call. Returns (result text, is_error)."""
    try:
        if name == "get_deployments":
            out: Any = [d.view() for d in cluster.list(args.get("namespace"))]
        elif name == "get_pods":
            out = pods_view(cluster, args["namespace"], args["deployment"])
        elif name == "get_logs":
            return logs_view(cluster, args["namespace"], args["deployment"]), False
        elif name == "set_memory_limit":
            out = cluster.set_memory_limit(
                args["namespace"], args["deployment"], args["memory"]
            ).view()
        elif name == "scale":
            out = cluster.scale(args["namespace"], args["deployment"], int(args["replicas"])).view()
        elif name == "restart":
            out = cluster.restart(args["namespace"], args["deployment"]).view()
        elif name == "rollback":
            d = cluster.rollback(args["namespace"], args["deployment"], args.get("to_revision"))
            out = d.view()
        elif name == "delete_pod":
            cluster.delete_pod(args["namespace"], args["pod"])
            out = {"deleted": args["pod"]}
        elif name == "escalate":
            cluster.escalate(args["summary"])
            out = {"escalated": True}
        else:
            return f"unknown tool {name!r}", True
    except (LookupError, ValueError, KeyError, TypeError) as exc:
        return f"error: {exc}", True
    return json.dumps(out, sort_keys=True), False
