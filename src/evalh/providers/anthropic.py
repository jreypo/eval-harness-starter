"""Real provider backed by the Anthropic SDK.

Only imported when EVAL_PROVIDER=anthropic, so the default install and CI never
need the SDK or a key. Install with `uv sync --extra anthropic`.

Two choices worth knowing about when you read eval results from this provider:
- No server-side refusal fallback. A fallback would silently answer with a
  different model, and an eval must measure the model named in the config. A
  refusal surfaces as stop_reason "refusal" and the trial fails visibly.
- Model and judge come from the config and are recorded in every RunResult.
"""

from __future__ import annotations

import os
from pathlib import Path

import anthropic

from evalh.providers.base import LLM, Request, Response


class AnthropicProvider:
    name = "anthropic"

    def __init__(self) -> None:
        # Resolves credentials from the environment (ANTHROPIC_API_KEY and friends).
        self.client = anthropic.Anthropic()

    def complete(self, request: Request, *, trial_index: int = 0) -> Response:
        kwargs = {
            "model": request.model,
            "max_tokens": request.max_tokens,
            "system": request.system,
            "messages": request.messages,
        }
        if request.tools:
            kwargs["tools"] = request.tools
        msg = self.client.messages.create(**kwargs)
        data = msg.to_dict()
        usage = data.get("usage") or {}
        return Response(
            content=data["content"],
            stop_reason=data.get("stop_reason") or "end_turn",
            usage={
                "input_tokens": int(usage.get("input_tokens") or 0),
                "output_tokens": int(usage.get("output_tokens") or 0),
            },
        )


def maybe_recording(provider: LLM, fixtures_dir: Path | str) -> LLM:
    """With EVAL_RECORD=1, write every real response as a stub fixture."""
    if os.environ.get("EVAL_RECORD") == "1":
        from evalh.providers.stub import RecordingProvider

        return RecordingProvider(provider, fixtures_dir)
    return provider
