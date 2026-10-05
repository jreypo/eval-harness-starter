"""The LLM interface the harness depends on, and nothing more.

Everything model-facing goes through `LLM.complete()`. Keeping the surface this
small is what makes the stub provider possible: if a request can be written
down, it can be hashed, and if it can be hashed, its response can be replayed.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Protocol


@dataclass(frozen=True)
class Request:
    model: str
    system: str
    messages: list[dict[str, Any]]
    tools: list[dict[str, Any]] = field(default_factory=list)
    max_tokens: int = 4096

    def canonical(self) -> dict[str, Any]:
        """The fields that define "the same request" for replay purposes.

        max_tokens is left out on purpose: raising it should not invalidate
        every recorded fixture. The system prompt is in: changing it must.
        """
        return {
            "model": self.model,
            "system": self.system,
            "messages": self.messages,
            "tools": self.tools,
        }

    def key(self) -> str:
        blob = json.dumps(self.canonical(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(blob.encode()).hexdigest()[:16]


@dataclass
class Response:
    # Content blocks as plain dicts in Anthropic Messages format:
    # {"type": "text", "text": ...} and {"type": "tool_use", "id", "name", "input"}.
    # Other block types (thinking) are carried through untouched.
    content: list[dict[str, Any]]
    stop_reason: str
    usage: dict[str, int] = field(default_factory=dict)

    @property
    def text(self) -> str:
        return "".join(b["text"] for b in self.content if b.get("type") == "text")

    @property
    def tool_uses(self) -> list[dict[str, Any]]:
        return [b for b in self.content if b.get("type") == "tool_use"]

    @property
    def tokens(self) -> int:
        return int(self.usage.get("input_tokens", 0)) + int(self.usage.get("output_tokens", 0))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class LLM(Protocol):
    name: str

    def complete(self, request: Request, *, trial_index: int = 0) -> Response:
        """Run one model call.

        trial_index is not part of the request. Real providers ignore it; the
        stub uses it to pick which recorded sample to replay, which is how an
        offline run still shows trial-to-trial variance.
        """
        ...


def make_provider(name: str, fixtures_dir: Path | str) -> LLM:
    """Build a provider by name. The anthropic SDK is imported only on demand."""
    if name == "stub":
        from evalh.providers.stub import StubProvider

        return StubProvider(fixtures_dir)
    if name == "anthropic":
        from evalh.providers.anthropic import AnthropicProvider, maybe_recording

        return maybe_recording(AnthropicProvider(), fixtures_dir)
    raise ValueError(f"unknown provider {name!r}; expected 'stub' or 'anthropic'")
