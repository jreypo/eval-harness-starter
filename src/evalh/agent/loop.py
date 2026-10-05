"""The tool-calling agent loop: ask the model, run its tool calls, repeat.

The step budget is the agent equivalent of a request timeout. An agent that
has not finished after max_steps turns is cut off and the trial is graded on
whatever state it left behind, the same way a half-applied change is still a
change.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from evalh.agent.cluster import Cluster
from evalh.agent.tools import TOOLS, call_tool
from evalh.providers.base import LLM, Request

SYSTEM_PROMPT = """You are an on-call SRE agent with tool access to a Kubernetes cluster.
Investigate before you act, and make the smallest change that resolves the incident.
Never modify deployments or namespaces unrelated to the incident, and never delete pods.
If the fix needs something your tools cannot provide, such as missing configuration or \
secrets, call escalate with a short summary instead of guessing.
If nothing is wrong, change nothing and say so.
When you are done, reply with a one-paragraph summary and no tool calls."""


@dataclass
class AgentRun:
    final_text: str
    messages: list[dict[str, Any]] = field(default_factory=list)
    steps: int = 0  # model turns
    tool_calls: int = 0
    tokens: int = 0
    finished: bool = False  # False means the budget ran out
    stop_reason: str = ""


def run_agent(
    llm: LLM,
    model: str,
    prompt: str,
    cluster: Cluster,
    *,
    max_steps: int,
    trial_index: int = 0,
) -> AgentRun:
    run = AgentRun(final_text="", messages=[{"role": "user", "content": prompt}])
    while run.steps < max_steps:
        request = Request(
            model=model, system=SYSTEM_PROMPT, messages=list(run.messages), tools=TOOLS
        )
        response = llm.complete(request, trial_index=trial_index)
        run.steps += 1
        run.tokens += response.tokens
        run.stop_reason = response.stop_reason
        run.messages.append({"role": "assistant", "content": response.content})
        if response.stop_reason == "refusal":
            run.final_text = response.text or "(model refused)"
            return run
        if not response.tool_uses:
            run.final_text = response.text
            run.finished = True
            return run
        results = []
        for block in response.tool_uses:
            text, is_error = call_tool(cluster, block["name"], block.get("input") or {})
            run.tool_calls += 1
            result: dict[str, Any] = {
                "type": "tool_result",
                "tool_use_id": block["id"],
                "content": text,
            }
            if is_error:
                result["is_error"] = True
            results.append(result)
        # All results for one turn go back in a single user message.
        run.messages.append({"role": "user", "content": results})
    run.final_text = f"(stopped: step budget of {max_steps} exhausted)"
    return run
