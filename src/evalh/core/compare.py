"""Baseline vs candidate: aggregate deltas, noise, paired flips, and a gate.

This is canary analysis for model changes. The aggregate pass rate is the
dashboard number, and it lies by omission: one case breaking while another
improves leaves it flat. Pairing cases by id (the same probe, before and after)
catches that, the way you diff per-endpoint error rates instead of trusting the
fleet-wide average.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from evalh.core import stats
from evalh.core.report import SuiteSummary, overall, summarize
from evalh.core.types import RunResult

# Keys that, if different, mean the delta is not purely the model's doing.
# git_sha is expected to differ between runs and is shown, not warned about.
MATERIAL_VERSION_KEYS = ("dataset", "prompts", "model", "judge_model", "provider")


@dataclass
class Flip:
    case_id: str
    suite: str
    before: tuple[int, int]  # (trials, passes)
    after: tuple[int, int]

    def __str__(self) -> str:
        b, a = self.before, self.after
        return f"{self.case_id} [{self.suite}] ({b[1]}/{b[0]} -> {a[1]}/{a[0]})"


@dataclass
class Comparison:
    baseline: RunResult
    candidate: RunResult
    base: dict[str, SuiteSummary]
    cand: dict[str, SuiteSummary]
    base_all: SuiteSummary
    cand_all: SuiteSummary
    regressions: list[Flip] = field(default_factory=list)  # pass -> fail
    fixes: list[Flip] = field(default_factory=list)  # fail -> pass
    only_in_baseline: list[str] = field(default_factory=list)
    only_in_candidate: list[str] = field(default_factory=list)
    version_diffs: dict[str, tuple[str, str]] = field(default_factory=dict)
    gate_checks: list[tuple[str, bool, str]] = field(default_factory=list)

    @property
    def gate_passed(self) -> bool:
        return all(ok for _, ok, _ in self.gate_checks)


def compare(baseline: RunResult, candidate: RunResult, gate: dict[str, Any]) -> Comparison:
    c = Comparison(
        baseline=baseline,
        candidate=candidate,
        base=summarize(baseline.trials),
        cand=summarize(candidate.trials),
        base_all=overall(baseline.trials),
        cand_all=overall(candidate.trials),
    )
    keys = set(baseline.versions) | set(candidate.versions)
    c.version_diffs = {
        k: (baseline.versions.get(k, "-"), candidate.versions.get(k, "-"))
        for k in sorted(keys)
        if baseline.versions.get(k) != candidate.versions.get(k)
    }

    # Paired flips. A case "passes" only if all of its trials passed.
    b_cases, c_cases = c.base_all.per_case, c.cand_all.per_case
    suite_of = {t.case_id: t.case_suite for t in candidate.trials + baseline.trials}
    c.only_in_baseline = sorted(set(b_cases) - set(c_cases))
    c.only_in_candidate = sorted(set(c_cases) - set(b_cases))
    for cid in sorted(set(b_cases) & set(c_cases)):
        was, now = c.base_all.case_passed(cid), c.cand_all.case_passed(cid)
        if was != now:
            flip = Flip(cid, suite_of[cid], b_cases[cid], c_cases[cid])
            (c.regressions if was else c.fixes).append(flip)

    # The gate only looks at regression cases. Capability cases are expected to
    # move around; they are tracked, not enforced.
    reg = c.cand.get("regression")
    min_rate = float(gate.get("regression_min_pass_rate", 0.0))
    if reg is not None:
        c.gate_checks.append(
            (
                f"regression pass@1 {reg.rate:.2f} >= {min_rate:.2f}",
                reg.rate >= min_rate,
                "",
            )
        )
    max_new = int(gate.get("max_new_failures", 0))
    new_reg = [f for f in c.regressions if f.suite == "regression"]
    c.gate_checks.append(
        (
            f"new regression failures {len(new_reg)} <= {max_new}",
            len(new_reg) <= max_new,
            ", ".join(f.case_id for f in new_reg),
        )
    )
    return c


def _short_time(iso: str) -> str:
    return iso[:16] + "Z" if iso.endswith("+00:00") else iso


def _fmt(v: float) -> str:
    return "  n/a" if v != v else f"{v:5.2f}"  # NaN check


def format_comparison(c: Comparison) -> str:
    b, k = c.baseline, c.cand_all.k
    lines = [
        f"Suite: {c.candidate.suite}   baseline {_short_time(b.started_at)} ({b.run_id})"
        f"   candidate {_short_time(c.candidate.started_at)} ({c.candidate.run_id})",
    ]
    material = {key: v for key, v in c.version_diffs.items() if key in MATERIAL_VERSION_KEYS}
    if material:
        lines.append("WARNING: runs differ in more than the code; deltas mix several changes:")
        lines += [f"  {key}: {old} -> {new}" for key, (old, new) in material.items()]
    lines += [
        "",
        f"{'':<22} {'baseline':>9} {'candidate':>10} {'delta':>7}   95% CI (candidate)",
    ]

    def row(label: str, old: float, new: float, ci: tuple[float, float] | None) -> str:
        delta = new - old
        ci_txt = f"   [{ci[0]:.2f}, {ci[1]:.2f}]" if ci else ""
        return f"{label:<22} {_fmt(old):>9} {_fmt(new):>10} {delta:>+7.2f}{ci_txt}"

    for suite, s in c.cand.items():
        old = c.base[suite].rate if suite in c.base else float("nan")
        lines.append(row(f"{suite} pass@1", old, s.rate, s.ci))
    lines.append(row("all pass@1", c.base_all.rate, c.cand_all.rate, c.cand_all.ci))
    if k > 1:
        # Agent suites: retries hide unreliability (pass@k), pass^k exposes it.
        lines.append(row(f"all pass@{k}", c.base_all.pass_at_k(k), c.cand_all.pass_at_k(k), None))
        lines.append(row(f"all pass^{k}", c.base_all.pass_hat_k(k), c.cand_all.pass_hat_k(k), None))

    se = stats.diff_se(c.base_all.rate, c.base_all.n, c.cand_all.rate, c.cand_all.n)
    delta = c.cand_all.rate - c.base_all.rate
    verdict = "within noise" if abs(delta) <= 1.96 * se else "outside noise"
    lines += [
        f"noise: SE of the all-pass@1 delta = {se:.2f}; |delta| {abs(delta):.2f} is {verdict} "
        f"(1.96 SE = {1.96 * se:.2f})",
        "",
        "Paired flips (same case, baseline vs candidate"
        + (f"; a case passes only if all {k} trials pass):" if k > 1 else "):"),
        f"  pass -> fail: {len(c.regressions)}" + _flip_list(c.regressions),
        f"  fail -> pass: {len(c.fixes)}" + _flip_list(c.fixes),
    ]
    if c.only_in_baseline or c.only_in_candidate:
        lines.append(
            f"  unpaired: removed {c.only_in_baseline or '-'}  added {c.only_in_candidate or '-'}"
        )
    lines += ["", "Gate (regression cases only):"]
    for desc, ok, note in c.gate_checks:
        lines.append(f"  {'ok  ' if ok else 'FAIL'}  {desc}" + (f"   ({note})" if note else ""))
    failed = [d for d, ok, _ in c.gate_checks if not ok]
    lines.append(f"Gate: {'PASS' if not failed else 'FAIL'}")
    return "\n".join(lines)


def _flip_list(flips: list[Flip]) -> str:
    return "".join(f"\n      {f}" for f in flips)
