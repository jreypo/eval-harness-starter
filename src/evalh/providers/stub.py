"""Fixture replay provider, plus the recorder that produces fixtures.

One file per request: fixtures/<suite>/<hash>.json holding the request (for
humans reading a diff) and one recorded response per trial index. Replay is a
pure lookup. There is no fallback on a miss: a silent fallback would turn a
changed prompt into a green eval run, which is the worst possible failure mode
for a test harness.
"""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any

from evalh.errors import HarnessError
from evalh.providers.base import LLM, Request, Response


class FixtureMissingError(HarnessError):
    pass


def fixture_path(fixtures_dir: Path, request: Request) -> Path:
    return fixtures_dir / f"{request.key()}.json"


class StubProvider:
    name = "stub"

    def __init__(self, fixtures_dir: Path | str):
        self.fixtures_dir = Path(fixtures_dir)
        if not self.fixtures_dir.is_dir():
            raise FixtureMissingError(f"fixture directory {self.fixtures_dir} does not exist")
        self._cache: dict[str, dict[str, Any]] = {}
        self._lock = threading.Lock()

    def complete(self, request: Request, *, trial_index: int = 0) -> Response:
        key = request.key()
        record = self._load(key)
        responses = record["responses"]
        raw = responses.get(str(trial_index))
        if raw is None:
            raise FixtureMissingError(
                f"fixture {key} in {self.fixtures_dir} has no response for trial "
                f"{trial_index} (recorded trials: {sorted(responses, key=int)}). "
                "Record more trials or lower k."
            )
        return Response(**raw)

    def _load(self, key: str) -> dict[str, Any]:
        with self._lock:
            if key in self._cache:
                return self._cache[key]
        path = self.fixtures_dir / f"{key}.json"
        if not path.exists():
            raise FixtureMissingError(
                f"no fixture for request hash {key} in {self.fixtures_dir}. The request "
                "(model, system prompt, messages or tools) changed since fixtures were "
                "recorded. Re-record with `make fixtures` or a real provider and EVAL_RECORD=1."
            )
        record = json.loads(path.read_text())
        with self._lock:
            self._cache[key] = record
        return record


class RecordingProvider:
    """Wraps any provider and writes each response into fixture files.

    Used by scripts/gen_fixtures.py with a scripted model, and by the
    anthropic provider when EVAL_RECORD=1 to capture real responses.
    """

    def __init__(self, inner: LLM, fixtures_dir: Path | str):
        self.inner = inner
        self.name = inner.name
        self.fixtures_dir = Path(fixtures_dir)
        self.fixtures_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def complete(self, request: Request, *, trial_index: int = 0) -> Response:
        response = self.inner.complete(request, trial_index=trial_index)
        path = fixture_path(self.fixtures_dir, request)
        with self._lock:
            if path.exists():
                record = json.loads(path.read_text())
            else:
                record = {"request": request.canonical(), "responses": {}}
            record["responses"][str(trial_index)] = response.to_dict()
            record["responses"] = dict(
                sorted(record["responses"].items(), key=lambda kv: int(kv[0]))
            )
            path.write_text(json.dumps(record, indent=1, sort_keys=True) + "\n")
        return response
