"""A deterministic reference agent: a runbook written as code.

It sees only what the model sees (tool output, never `hidden`) and uses the
same tools. If it cannot score 100% on every task, the task is unsolvable from
the observable state or a grader is wrong, and model results on that task
mean nothing. Run it on every commit; it is the eval's own smoke test.
"""

from __future__ import annotations

import json
import math
from typing import Any

from evalh.agent.cluster import Cluster, parse_mi
from evalh.agent.loop import AgentRun
from evalh.agent.tools import call_tool


class _Session:
    """Calls tools and records them in the same transcript shape as the model loop."""

    def __init__(self, cluster: Cluster, prompt: str):
        self.cluster = cluster
        self.run = AgentRun(final_text="", messages=[{"role": "user", "content": prompt}])

    def call(self, name: str, **args: Any) -> Any:
        text, is_error = call_tool(self.cluster, name, args)
        self.run.steps += 1
        self.run.tool_calls += 1
        self.run.messages.append(
            {"role": "assistant", "content": [{"type": "tool_use", "name": name, "input": args}]}
        )
        self.run.messages.append(
            {"role": "user", "content": [{"type": "tool_result", "content": text}]}
        )
        if is_error:
            raise RuntimeError(text)
        return text if name == "get_logs" else json.loads(text)

    def finish(self, text: str) -> AgentRun:
        self.run.final_text = text
        self.run.finished = True
        self.run.messages.append({"role": "assistant", "content": [{"type": "text", "text": text}]})
        return self.run


def solve(prompt: str, cluster: Cluster) -> AgentRun:
    s = _Session(cluster, prompt)
    broken = [d for d in s.call("get_deployments") if d["status"] != "Running"]
    if not broken:
        return s.finish("All deployments are Running; nothing to do.")

    actions = []
    for d in broken:
        ns, name = d["namespace"], d["name"]
        logs = s.call("get_logs", namespace=ns, deployment=name)
        if d["reason"] == "OOMKilled":
            # Closed loop: double the limit and re-check, at most three times.
            limit = parse_mi(d["memory_limit"])
            for _ in range(3):
                limit *= 2
                d = s.call("set_memory_limit", namespace=ns, deployment=name, memory=f"{limit}Mi")
                if d["reason"] != "OOMKilled":
                    break
            actions.append(f"raised {name} memory limit to {limit}Mi")
        elif d["reason"] == "Error":
            s.call("rollback", namespace=ns, deployment=name)
            actions.append(f"rolled back {name} to the previous revision")
        elif d["reason"] == "Saturated":
            target = math.ceil(d["replicas"] * d["cpu_utilization"] / 0.8)
            s.call("scale", namespace=ns, deployment=name, replicas=target)
            actions.append(f"scaled {name} to {target} replicas")
        else:
            # Anything else (missing config, secrets) is not ours to invent.
            s.call("escalate", summary=f"{ns}/{name} is {d['status']} ({d['reason']}): {logs}")
            actions.append(f"escalated {name}")
    after = s.call("get_deployments")
    still = [f"{d['namespace']}/{d['name']}" for d in after if d["status"] != "Running"]
    return s.finish(
        "; ".join(actions) + (f". Still not Running: {', '.join(still)}" if still else ".")
    )
