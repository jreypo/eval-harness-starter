"""Wire agent tasks, the cluster, the loop (or the reference agent) and graders."""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path
from typing import Any

import yaml

from evalh.agent import reference
from evalh.agent.cluster import Cluster
from evalh.agent.graders import grade_all
from evalh.agent.loop import SYSTEM_PROMPT, run_agent
from evalh.agent.tools import TOOLS
from evalh.config import fixtures_dir
from evalh.core.report import git_sha, hash_files, hash_text
from evalh.core.types import Case, Trial
from evalh.providers.base import LLM, make_provider


def load_tasks(path: Path | str) -> list[Case]:
    cases = []
    for t in yaml.safe_load(Path(path).read_text()):
        cases.append(
            Case(
                id=t["id"],
                suite=t["suite"],
                input={"prompt": t["prompt"], "seed": t["seed"]},
                expected={k: t.get(k, {}) for k in ("expect", "scope", "invariants", "budgets")},
            )
        )
    return cases


def versions(config: dict[str, Any], agent: str) -> dict[str, str]:
    is_ref = agent == "reference"
    return {
        "dataset": hash_files([Path(config["tasks"])]),
        "prompts": "none"
        if is_ref
        else hash_text(SYSTEM_PROMPT, json.dumps(TOOLS, sort_keys=True)),
        "model": "reference" if is_ref else config["model"],
        "judge_model": "none",  # agent outcomes are graded in code
        "provider": "none" if is_ref else config["provider"],
        "git_sha": git_sha(),
    }


def make_run_one(
    config: dict[str, Any], agent: str = "model", llm: LLM | None = None
) -> Callable[[Case, int], Trial]:
    if agent == "model":
        llm = llm or make_provider(config["provider"], fixtures_dir(config))

    def run_one(case: Case, trial_index: int) -> Trial:
        # A brand-new cluster per trial, built from the seed. Nothing survives
        # between trials: no shared objects, no globals, no reused client state.
        cluster = Cluster.from_seed(case.input["seed"])
        task = case.expected
        if agent == "reference":
            run = reference.solve(case.input["prompt"], cluster)
        else:
            assert llm is not None
            run = run_agent(
                llm,
                config["model"],
                case.input["prompt"],
                cluster,
                max_steps=int(task["budgets"].get("max_steps", 12)),
                trial_index=trial_index,
            )
        return Trial(
            case_id=case.id,
            trial_index=trial_index,
            output={
                "final": run.final_text,
                "writes": [asdict(w) for w in cluster.writes],
                "escalations": cluster.escalations,
            },
            transcript=run.messages,
            grades=grade_all(task, cluster, run),
            steps=run.steps,
            tokens=run.tokens,
        )

    return run_one
