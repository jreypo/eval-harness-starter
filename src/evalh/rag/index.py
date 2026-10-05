"""BM25 over runbooks chunked by section.

Chunk ids are `<file>#<n>`: n=0 is the title and intro, n>=1 is the nth `##`
section. Datasets reference these ids, so the chunker is part of the dataset
contract: change it and every relevant_chunks label is suspect.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from rank_bm25 import BM25Okapi

_TOKEN = re.compile(r"[a-z0-9]+")


@dataclass(frozen=True)
class Chunk:
    id: str
    title: str
    text: str


def tokenize(text: str) -> list[str]:
    # Deliberately naive: lowercase alphanumerics. "ledger-svc" becomes
    # ["ledger", "svc"], which is exactly how similar service names collide.
    return _TOKEN.findall(text.lower())


def chunk_markdown(name: str, text: str) -> list[Chunk]:
    parts = re.split(r"(?m)^## ", text)
    chunks = [Chunk(id=f"{name}#0", title=parts[0].splitlines()[0].lstrip("# "), text=parts[0])]
    for i, part in enumerate(parts[1:], start=1):
        title = part.splitlines()[0].strip()
        chunks.append(Chunk(id=f"{name}#{i}", title=title, text="## " + part))
    return [Chunk(c.id, c.title, c.text.strip()) for c in chunks]


def load_corpus(directory: Path | str) -> list[Chunk]:
    chunks: list[Chunk] = []
    for path in sorted(Path(directory).glob("*.md")):
        chunks.extend(chunk_markdown(path.name, path.read_text()))
    return chunks


class BM25Index:
    def __init__(self, chunks: list[Chunk]):
        self.chunks = chunks
        self.by_id = {c.id: c for c in chunks}
        self._bm25 = BM25Okapi([tokenize(c.title + " " + c.text) for c in chunks])

    def search(self, query: str, k: int) -> list[tuple[Chunk, float]]:
        scores = self._bm25.get_scores(tokenize(query))
        # Ties broken by chunk id so retrieval, and therefore the prompt and its
        # fixture hash, is fully deterministic.
        ranked = sorted(
            zip(self.chunks, scores, strict=True),
            key=lambda cs: (-cs[1], cs[0].id),
        )
        return [(c, float(s)) for c, s in ranked[:k]]
