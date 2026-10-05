"""The system under test: retrieve runbook chunks, then generate an answer.

Nothing here knows it is being evaluated. The harness drives it exactly the way
production would, which is the point: you evaluate the deployed artifact, not a
test double of it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from evalh.providers.base import LLM, Request
from evalh.rag.index import BM25Index, Chunk

REFUSAL = "I don't know based on the available runbooks."

SYSTEM_PROMPT = f"""You are an on-call assistant. Answer the engineer's question using only the \
runbook excerpts provided. Be concise and give the concrete commands from the excerpts. If the \
excerpts do not contain the answer, reply exactly: {REFUSAL}"""


@dataclass
class RagAnswer:
    text: str
    retrieved: list[str]
    tokens: int = 0
    transcript: list[dict[str, Any]] = field(default_factory=list)


def format_context(chunks: list[Chunk]) -> str:
    return "\n\n".join(f"[{c.id}]\n{c.text}" for c in chunks)


def build_request(model: str, question: str, chunks: list[Chunk]) -> Request:
    user = f"Runbook excerpts:\n\n{format_context(chunks)}\n\nQuestion: {question}"
    return Request(model=model, system=SYSTEM_PROMPT, messages=[{"role": "user", "content": user}])


class RagPipeline:
    def __init__(self, index: BM25Index, llm: LLM, model: str, top_k: int = 3):
        self.index = index
        self.llm = llm
        self.model = model
        self.top_k = top_k

    def retrieve(self, question: str) -> list[Chunk]:
        return [c for c, _ in self.index.search(question, self.top_k)]

    def answer(self, question: str, *, trial_index: int = 0) -> RagAnswer:
        chunks = self.retrieve(question)
        request = build_request(self.model, question, chunks)
        response = self.llm.complete(request, trial_index=trial_index)
        return RagAnswer(
            text=response.text,
            retrieved=[c.id for c in chunks],
            tokens=response.tokens,
            transcript=[
                {"role": "user", "content": request.messages[0]["content"]},
                {"role": "assistant", "content": response.text},
            ],
        )
