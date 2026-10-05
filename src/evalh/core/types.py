"""Core data types shared by every suite.

These are deliberately plain dataclasses: a run result is a log record, and you
should be able to read one with `jq` without knowing anything about this code.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class Case:
    """One probe in a dataset. Think of it as a synthetic check definition."""

    id: str
    suite: str  # "regression" | "capability"
    input: dict[str, Any]
    expected: dict[str, Any]


@dataclass
class GradeResult:
    grader: str
    # Tri-state on purpose. None means "unknown": the judge abstained or the
    # grader could not decide. Unknown is never silently promoted to pass.
    passed: bool | None
    score: float | None = None
    detail: dict[str, Any] = field(default_factory=dict)
    # Set when the grader itself broke (for example, the judge returned
    # unparseable output). A broken health check is not a failing service, so
    # it is recorded separately instead of crashing the run.
    error: str | None = None


@dataclass
class Trial:
    case_id: str
    trial_index: int
    output: Any
    transcript: list[dict[str, Any]]
    grades: list[GradeResult]
    # Copied from the Case so a RunResult is self-contained for compare.
    case_suite: str = "regression"
    steps: int = 0
    tokens: int = 0
    latency_ms: float = 0.0
    error: str | None = None

    @property
    def passed(self) -> bool:
        # A trial with no grades has not been checked, so it has not passed.
        if self.error is not None or not self.grades:
            return False
        return all(g.passed is True for g in self.grades)


@dataclass
class RunResult:
    run_id: str  # timestamp + short hash
    started_at: str  # ISO 8601 from datetime.now(timezone.utc)
    suite: str
    # dataset, prompts, model, judge_model, provider, git_sha. If any of these
    # differ between two runs, a delta between them is not purely the model.
    versions: dict[str, str]
    trials: list[Trial]
    config: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        for t, raw in zip(self.trials, d["trials"], strict=True):
            raw["passed"] = t.passed
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> RunResult:
        trials = []
        for raw in d["trials"]:
            raw = {k: v for k, v in raw.items() if k != "passed"}
            raw["grades"] = [GradeResult(**g) for g in raw["grades"]]
            trials.append(Trial(**raw))
        return cls(
            run_id=d["run_id"],
            started_at=d["started_at"],
            suite=d["suite"],
            versions=d["versions"],
            trials=trials,
            config=d.get("config", {}),
        )
