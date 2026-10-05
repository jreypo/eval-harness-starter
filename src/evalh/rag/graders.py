"""RAG graders: deterministic code checks first, LLM judge second.

Code graders (recall@k, reciprocal rank, refusal) are cheap, exact and never
flaky; run them on every commit. The judge covers what code cannot (is this
answer supported, is it right) and is itself a model, so it is pinned,
calibrated against human labels, and allowed to say "unknown".
"""

from __future__ import annotations

import json
import re
from typing import Any

from evalh.core.types import GradeResult, Trial
from evalh.providers.base import LLM, Request
from evalh.rag.pipeline import REFUSAL

VERDICTS = {"pass": True, "fail": False, "unknown": None}

_JSON_RULES = """Reply with only a JSON object and nothing else:
{"verdict": "pass" | "fail" | "unknown", "reason": "<one sentence>"}"""

FAITHFULNESS_PROMPT = f"""You grade answers from an on-call assistant. Decide whether every \
factual claim and command in the ANSWER is supported by the EXCERPTS. Judge support only, not \
whether the excerpts are the right ones for the question. Use "unknown" if you cannot decide.
{_JSON_RULES}"""

CORRECTNESS_PROMPT = f"""You grade answers from an on-call assistant against a reference answer \
written by the service owner. "pass" if the ANSWER contains the essential actions in the \
REFERENCE and nothing that contradicts it. "fail" otherwise. Use "unknown" if the reference \
does not let you decide.
{_JSON_RULES}"""


# ---- retrieval ----------------------------------------------------------------


def recall_at_k(retrieved: list[str], relevant: list[str]) -> float:
    if not relevant:
        return 1.0
    return len(set(retrieved) & set(relevant)) / len(set(relevant))


def reciprocal_rank(retrieved: list[str], relevant: list[str]) -> float:
    for rank, chunk_id in enumerate(retrieved, start=1):
        if chunk_id in relevant:
            return 1.0 / rank
    return 0.0


def grade_retrieval(retrieved: list[str], relevant: list[str], min_recall: float) -> GradeResult:
    recall = recall_at_k(retrieved, relevant)
    return GradeResult(
        grader="retrieval",
        passed=recall >= min_recall,
        score=recall,
        detail={
            "recall@k": recall,
            "reciprocal_rank": reciprocal_rank(retrieved, relevant),
            "retrieved": retrieved,
            "relevant": relevant,
        },
    )


def retrieval_metrics(trials: list[Trial]) -> dict[str, float]:
    """Mean recall@k and MRR across trials, the numbers you would chart over time."""
    details = [g.detail for t in trials for g in t.grades if g.grader == "retrieval"]
    if not details:
        return {}
    n = len(details)
    return {
        "mean_recall@k": sum(d["recall@k"] for d in details) / n,
        "mrr": sum(d["reciprocal_rank"] for d in details) / n,
    }


def grade_refusal(answer: str) -> GradeResult:
    """For questions the corpus cannot answer, the only correct output is a refusal."""
    refused = REFUSAL.lower().rstrip(".") in answer.lower()
    return GradeResult(grader="refusal", passed=refused, detail={"refused": refused})


# ---- judge ----------------------------------------------------------------------


def parse_verdict(grader: str, text: str) -> GradeResult:
    """Turn judge text into a GradeResult. Malformed output is a grader error.

    The judge is an external dependency with its own failure modes. Garbage
    from it is recorded on the trial (and counted in the report) instead of
    being guessed at or crashing the run.
    """
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip())
    match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    try:
        data: Any = json.loads(match.group(0)) if match else None
    except json.JSONDecodeError:
        data = None
    verdict = str(data.get("verdict", "")).strip().lower() if isinstance(data, dict) else ""
    if verdict not in VERDICTS:
        return GradeResult(
            grader=grader,
            passed=None,
            error="judge returned invalid output",
            detail={"raw": text[:500]},
        )
    return GradeResult(
        grader=grader,
        passed=VERDICTS[verdict],
        detail={"verdict": verdict, "reason": str(data.get("reason", ""))},
    )


class Judge:
    def __init__(self, llm: LLM, model: str):
        self.llm = llm
        self.model = model

    def _ask(self, grader: str, system: str, user: str, trial_index: int) -> GradeResult:
        request = Request(
            model=self.model,
            system=system,
            messages=[{"role": "user", "content": user}],
            max_tokens=512,
        )
        response = self.llm.complete(request, trial_index=trial_index)
        result = parse_verdict(grader, response.text)
        result.detail["tokens"] = response.tokens
        return result

    def faithfulness(self, context: str, answer: str, *, trial_index: int = 0) -> GradeResult:
        user = f"EXCERPTS:\n{context}\n\nANSWER:\n{answer}"
        return self._ask("faithfulness", FAITHFULNESS_PROMPT, user, trial_index)

    def correctness(
        self, question: str, reference: str, answer: str, *, trial_index: int = 0
    ) -> GradeResult:
        user = f"QUESTION:\n{question}\n\nREFERENCE:\n{reference}\n\nANSWER:\n{answer}"
        return self._ask("correctness", CORRECTNESS_PROMPT, user, trial_index)
