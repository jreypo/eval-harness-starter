from evalh.core.compare import compare, format_comparison
from evalh.core.report import load_result, write_result
from evalh.core.types import GradeResult, RunResult, Trial

VERSIONS = {
    "dataset": "d1",
    "prompts": "p1",
    "model": "m",
    "judge_model": "j",
    "provider": "stub",
    "git_sha": "a",
}
GATE = {"regression_min_pass_rate": 0.9, "max_new_failures": 0}


def run(outcomes: dict[str, list[bool]], suites: dict[str, str] | None = None, **versions):
    suites = suites or {}
    trials = [
        Trial(cid, i, None, [], [GradeResult("g", ok)], case_suite=suites.get(cid, "regression"))
        for cid, results in outcomes.items()
        for i, ok in enumerate(results)
    ]
    return RunResult("r", "2026-10-05T00:00:00+00:00", "agent", {**VERSIONS, **versions}, trials)


def test_identical_runs_pass_the_gate():
    r = run({"a": [True], "b": [True]})
    c = compare(r, r, GATE)
    assert c.gate_passed and not c.regressions and not c.fixes


def test_flat_aggregate_with_flips_fails_the_gate():
    base = run({"a": [True], "b": [False], "c": [True], "d": [True]}, {"b": "capability"})
    cand = run({"a": [False], "b": [True], "c": [True], "d": [True]}, {"b": "capability"})
    c = compare(base, cand, {"regression_min_pass_rate": 0.0, "max_new_failures": 0})
    assert c.base_all.rate == c.cand_all.rate  # aggregate did not move
    assert [f.case_id for f in c.regressions] == ["a"]
    assert [f.case_id for f in c.fixes] == ["b"]
    assert not c.gate_passed
    out = format_comparison(c)
    assert "Gate: FAIL" in out and "a [regression]" in out


def test_capability_flips_are_reported_but_not_gated():
    suites = {"x": "capability"}
    base = run({"a": [True], "x": [True]}, suites)
    cand = run({"a": [True], "x": [False]}, suites)
    c = compare(base, cand, GATE)
    assert [f.case_id for f in c.regressions] == ["x"]
    assert c.gate_passed


def test_min_pass_rate_check():
    base = run({str(i): [True] for i in range(10)})
    cand = run({**{str(i): [True] for i in range(8)}, "8": [False], "9": [False]})
    c = compare(base, cand, {"regression_min_pass_rate": 0.9, "max_new_failures": 5})
    assert not c.gate_passed


def test_one_flaky_trial_flips_the_case_when_k_greater_than_1():
    base = run({"a": [True] * 5})
    cand = run({"a": [True, True, False, True, True]})
    c = compare(base, cand, GATE)
    assert [f.case_id for f in c.regressions] == ["a"]
    assert c.cand_all.pass_hat_k(5) == 0.0
    assert c.cand_all.pass_at_k(5) == 1.0
    assert "pass^5" in format_comparison(c)


def test_version_mismatch_is_warned_but_git_sha_is_not():
    base = run({"a": [True]})
    assert "WARNING" not in format_comparison(compare(base, run({"a": [True]}, git_sha="b"), GATE))
    out = format_comparison(compare(base, run({"a": [True]}, judge_model="j2"), GATE))
    assert "WARNING" in out and "judge_model: j -> j2" in out


def test_unpaired_cases_are_listed_not_counted_as_flips():
    c = compare(run({"a": [True], "old": [True]}), run({"a": [True], "new": [False]}), GATE)
    assert c.only_in_baseline == ["old"] and c.only_in_candidate == ["new"]
    assert not c.regressions


def test_json_round_trip(tmp_path):
    r = run({"a": [True, False]})
    r.trials[1].grades.append(GradeResult("judge", None, error="judge returned invalid output"))
    path = write_result(r, tmp_path / "r.json")
    back = load_result(path)
    assert [t.passed for t in back.trials] == [True, False]
    assert back.trials[1].grades[1].error == "judge returned invalid output"
    assert back.versions == r.versions
