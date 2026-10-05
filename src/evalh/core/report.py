"""Summaries, console output, and the versioned JSON run record."""

from __future__ import annotations

import hashlib
import json
import subprocess
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from evalh.core import stats
from evalh.core.types import RunResult, Trial


@dataclass
class SuiteSummary:
    suite: str
    n: int  # trials
    passed: int
    k: int  # trials per case
    # case_id -> (trials, passes). The unit compare pairs on.
    per_case: dict[str, tuple[int, int]] = field(default_factory=dict)
    unknown: int = 0  # judge abstentions
    grader_errors: int = 0
    trial_errors: int = 0

    @property
    def rate(self) -> float:
        return stats.pass_rate(self.passed, self.n)

    @property
    def ci(self) -> tuple[float, float]:
        return stats.wilson_interval(self.passed, self.n)

    def case_passed(self, case_id: str) -> bool:
        # A case passes only if every one of its trials passed (the pass^k
        # view). For k=1 this is just the trial result.
        n, c = self.per_case[case_id]
        return n > 0 and c == n

    def pass_at_k(self, k: int) -> float:
        return stats.mean_over_cases(self.per_case.values(), k, stats.pass_at_k)

    def pass_hat_k(self, k: int) -> float:
        return stats.mean_over_cases(self.per_case.values(), k, stats.pass_hat_k)


def summarize(trials: list[Trial]) -> dict[str, SuiteSummary]:
    by_suite: dict[str, list[Trial]] = defaultdict(list)
    for t in trials:
        by_suite[t.case_suite].append(t)
    out = {}
    for suite in sorted(by_suite, key=_suite_order):
        ts = by_suite[suite]
        per_case: dict[str, list[int]] = defaultdict(lambda: [0, 0])
        for t in ts:
            per_case[t.case_id][0] += 1
            per_case[t.case_id][1] += int(t.passed)
        s = SuiteSummary(
            suite=suite,
            n=len(ts),
            passed=sum(t.passed for t in ts),
            k=max(n for n, _ in per_case.values()),
            per_case={cid: (n, c) for cid, (n, c) in sorted(per_case.items())},
        )
        for t in ts:
            s.trial_errors += t.error is not None
            for g in t.grades:
                s.unknown += g.passed is None and g.error is None
                s.grader_errors += g.error is not None
        out[suite] = s
    return out


def _suite_order(name: str) -> tuple[int, str]:
    return ({"regression": 0, "capability": 1}.get(name, 2), name)


def format_report(result: RunResult) -> str:
    lines = [
        f"Run {result.run_id}   suite={result.suite}   started {result.started_at}",
        "versions: " + "  ".join(f"{k}={v}" for k, v in sorted(result.versions.items())),
        "",
    ]
    summaries = summarize(result.trials)
    k = max((s.k for s in summaries.values()), default=1)
    header = f"{'suite':<12} {'trials':>6} {'passed':>6} {'pass@1':>7}  {'95% CI':<15}"
    if k > 1:
        header += f" {f'pass@{k}':>7} {f'pass^{k}':>7}"
    header += f" {'unknown':>7} {'grader_err':>10} {'trial_err':>9}"
    lines.append(header)
    for s in summaries.values():
        lo, hi = s.ci
        row = f"{s.suite:<12} {s.n:>6} {s.passed:>6} {s.rate:>7.2f}  [{lo:.2f}, {hi:.2f}]    "
        if k > 1:
            row += f" {s.pass_at_k(k):>7.2f} {s.pass_hat_k(k):>7.2f}"
        row += f" {s.unknown:>7} {s.grader_errors:>10} {s.trial_errors:>9}"
        lines.append(row)
    failing = _failing_cases(result.trials)
    if failing:
        lines += ["", "Failing cases:"]
        lines += [f"  {line}" for line in failing]
    return "\n".join(lines)


def _failing_cases(trials: list[Trial]) -> list[str]:
    by_case: dict[str, list[Trial]] = defaultdict(list)
    for t in trials:
        by_case[t.case_id].append(t)
    out = []
    for cid, ts in sorted(by_case.items()):
        bad = [t for t in ts if not t.passed]
        if not bad:
            continue
        reasons = sorted({_reason(t) for t in bad})
        out.append(f"{cid:<26} {len(ts) - len(bad)}/{len(ts)} passed   {'; '.join(reasons)}")
    return out


def _reason(t: Trial) -> str:
    if t.error:
        return f"error: {t.error[:80]}"
    parts = []
    for g in t.grades:
        if g.error:
            parts.append(f"{g.grader}=grader_error")
        elif g.passed is None:
            parts.append(f"{g.grader}=unknown")
        elif g.passed is False:
            parts.append(f"{g.grader}=fail")
    return ",".join(parts) or "no grades"


# ---- versioning -------------------------------------------------------------


def git_sha(cwd: Path | None = None) -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short=12", "HEAD"],
            cwd=cwd,
            capture_output=True,
            text=True,
            check=True,
        )
        return out.stdout.strip() or "nogit"
    except (OSError, subprocess.CalledProcessError):
        return "nogit"


def hash_files(paths: list[Path]) -> str:
    """Content hash over files (directories are walked in sorted order)."""
    h = hashlib.sha256()
    for p in sorted(_expand(paths)):
        h.update(str(p.as_posix()).encode())
        h.update(p.read_bytes())
    return h.hexdigest()[:12]


def hash_text(*parts: str) -> str:
    h = hashlib.sha256()
    for p in parts:
        h.update(p.encode())
        h.update(b"\0")
    return h.hexdigest()[:12]


def _expand(paths: list[Path]) -> list[Path]:
    out: list[Path] = []
    for p in paths:
        p = Path(p)
        out.extend(sorted(x for x in p.rglob("*") if x.is_file()) if p.is_dir() else [p])
    return out


def new_run(suite: str, versions: dict[str, str]) -> tuple[str, str]:
    """Return (run_id, started_at) for a new run."""
    now = datetime.now(timezone.utc)
    started_at = now.isoformat(timespec="seconds")
    tag = hash_text(suite, started_at, json.dumps(versions, sort_keys=True))[:6]
    return f"{now.strftime('%Y%m%dT%H%M%SZ')}-{tag}", started_at


def write_result(result: RunResult, path: Path | str) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result.to_dict(), indent=1, default=str) + "\n")
    return path


def load_result(path: Path | str) -> RunResult:
    data: dict[str, Any] = json.loads(Path(path).read_text())
    return RunResult.from_dict(data)
