"""Wire the RAG dataset, pipeline and graders into something the runner can run."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

from evalh.config import fixtures_dir
from evalh.core.report import git_sha, hash_files, hash_text
from evalh.core.types import Case, GradeResult, Trial
from evalh.providers.base import LLM, make_provider
from evalh.rag.graders import (
    CORRECTNESS_PROMPT,
    FAITHFULNESS_PROMPT,
    Judge,
    grade_refusal,
    grade_retrieval,
)
from evalh.rag.index import BM25Index, load_corpus
from evalh.rag.pipeline import SYSTEM_PROMPT, RagPipeline, format_context


def load_cases(paths: dict[str, str], only: str | None = None) -> list[Case]:
    cases = []
    for suite, path in paths.items():
        if only and suite != only:
            continue
        for line in Path(path).read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            cases.append(
                Case(
                    id=row["id"],
                    suite=suite,
                    input={"question": row["question"]},
                    expected={
                        "relevant_chunks": row["relevant_chunks"],
                        "reference": row.get("reference", ""),
                        "answerable": row.get("answerable", True),
                    },
                )
            )
    return cases


def versions(config: dict[str, Any], provider: str, *, retrieval_only: bool) -> dict[str, str]:
    ds = config["dataset"]
    paths = [Path(ds["corpus"]), *(Path(p) for p in ds["cases"].values())]
    return {
        "dataset": hash_files(paths),
        "prompts": "none"
        if retrieval_only
        else hash_text(SYSTEM_PROMPT, FAITHFULNESS_PROMPT, CORRECTNESS_PROMPT),
        "model": "none" if retrieval_only else config["model"],
        "judge_model": "none" if retrieval_only else config["judge_model"],
        "provider": "none" if retrieval_only else provider,
        "git_sha": git_sha(),
    }


def build_index(config: dict[str, Any]) -> BM25Index:
    return BM25Index(load_corpus(config["dataset"]["corpus"]))


def retrieval_run_one(config: dict[str, Any]) -> Callable[[Case, int], Trial]:
    """Retrieval only: no model calls, so it is free and runs on every commit."""
    index = build_index(config)
    k = config["retrieval"]["top_k"]
    min_recall = config["retrieval"]["min_recall"]

    def run_one(case: Case, trial_index: int) -> Trial:
        retrieved = [c.id for c, _ in index.search(case.input["question"], k)]
        grade = grade_retrieval(retrieved, case.expected["relevant_chunks"], min_recall)
        return Trial(case.id, trial_index, {"retrieved": retrieved}, [], [grade])

    return run_one


def full_run_one(config: dict[str, Any], llm: LLM | None = None) -> Callable[[Case, int], Trial]:
    llm = llm or make_provider(config["provider"], fixtures_dir(config))
    pipeline = RagPipeline(build_index(config), llm, config["model"], config["retrieval"]["top_k"])
    judge = Judge(llm, config["judge_model"])
    min_recall = config["retrieval"]["min_recall"]

    def run_one(case: Case, trial_index: int) -> Trial:
        question = case.input["question"]
        ans = pipeline.answer(question, trial_index=trial_index)
        grades: list[GradeResult]
        if not case.expected["answerable"]:
            # No relevant chunks exist, so retrieval and correctness are
            # meaningless. The only thing to check is that it refused.
            grades = [grade_refusal(ans.text)]
        else:
            context = format_context([pipeline.index.by_id[i] for i in ans.retrieved])
            grades = [
                grade_retrieval(ans.retrieved, case.expected["relevant_chunks"], min_recall),
                judge.faithfulness(context, ans.text, trial_index=trial_index),
                judge.correctness(
                    question, case.expected["reference"], ans.text, trial_index=trial_index
                ),
            ]
        judge_tokens = sum(int(g.detail.pop("tokens", 0)) for g in grades)
        return Trial(
            case_id=case.id,
            trial_index=trial_index,
            output={"answer": ans.text, "retrieved": ans.retrieved},
            transcript=ans.transcript,
            grades=grades,
            tokens=ans.tokens + judge_tokens,
        )

    return run_one
