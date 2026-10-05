"""End-to-end, offline: the shipped fixtures produce exactly the planned numbers.

If these fail after a prompt, tool or dataset change, regenerate fixtures with
`make fixtures` and update the plan in scripts/gen_fixtures.py on purpose.
"""

from pathlib import Path

import pytest

from evalh.cli import main
from evalh.core.compare import compare
from evalh.core.report import load_result, overall, summarize

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(autouse=True)
def at_repo_root(monkeypatch):
    monkeypatch.chdir(ROOT)
    monkeypatch.delenv("EVAL_PROVIDER", raising=False)
    monkeypatch.delenv("EVAL_FIXTURES_VARIANT", raising=False)


def run(tmp_path, *argv, variant=None, monkeypatch=None):
    if variant:
        monkeypatch.setenv("EVAL_FIXTURES_VARIANT", variant)
    out = tmp_path / f"{argv[1]}-{variant or 'base'}.json"
    assert main([*argv, "--out", str(out)]) == 0
    return load_result(out)


def counts(result):
    s = summarize(result.trials)
    return {name: (x.passed, x.n) for name, x in s.items()}


def test_retrieval_numbers(tmp_path):
    r = run(tmp_path, "run", "rag", "--retrieval-only")
    assert counts(r) == {"regression": (20, 20), "capability": (8, 9)}


def test_rag_baseline_and_regressed(tmp_path, monkeypatch):
    base = run(tmp_path, "run", "rag")
    assert counts(base) == {"regression": (20, 20), "capability": (5, 10)}
    cand = run(tmp_path, "run", "rag", variant="regressed", monkeypatch=monkeypatch)
    assert counts(cand) == {"regression": (19, 20), "capability": (6, 10)}
    c = compare(base, cand, base.config["gate"])
    assert overall(base.trials).rate == overall(cand.trials).rate
    assert [f.case_id for f in c.regressions] == ["rag-007", "rag-cap-08"]
    assert [f.case_id for f in c.fixes] == ["rag-cap-04", "rag-cap-05"]
    assert not c.gate_passed


def test_rag_silent_failure_passes_faithfulness(tmp_path):
    base = run(tmp_path, "run", "rag")
    (trial,) = [t for t in base.trials if t.case_id == "rag-cap-01"]
    grades = {g.grader: g.passed for g in trial.grades}
    assert grades == {"retrieval": False, "faithfulness": True, "correctness": False}


def test_agent_baseline_and_regressed(tmp_path, monkeypatch):
    base = run(tmp_path, "run", "agent")
    assert counts(base) == {"regression": (20, 20), "capability": (6, 10)}
    assert overall(base.trials).pass_hat_k(5) == pytest.approx(4 / 6)
    cand = run(tmp_path, "run", "agent", variant="regressed", monkeypatch=monkeypatch)
    assert counts(cand) == {"regression": (19, 20), "capability": (7, 10)}
    assert overall(cand.trials).pass_hat_k(5) == pytest.approx(4 / 6)
    c = compare(base, cand, base.config["gate"])
    assert [f.case_id for f in c.regressions] == ["oomkilled-api"]
    assert [f.case_id for f in c.fixes] == ["red-herring"]
    assert not c.gate_passed


def test_reference_agent_is_perfect(tmp_path):
    out = tmp_path / "ref.json"
    rc = main(
        [
            "run",
            "agent",
            "--agent",
            "reference",
            "--k",
            "2",
            "--require-pass-rate",
            "1.0",
            "--out",
            str(out),
        ]
    )
    assert rc == 0


def test_committed_baseline_matches_current_fixtures(tmp_path):
    for suite in ("rag", "agent"):
        baseline = load_result(ROOT / f"results/baseline/{suite}.json")
        latest = run(tmp_path, "run", suite)
        c = compare(baseline, latest, baseline.config["gate"])
        assert c.gate_passed and not c.regressions and not c.fixes, suite


def test_compare_cli_exit_codes(tmp_path, monkeypatch):
    run(tmp_path, "run", "agent", variant="regressed", monkeypatch=monkeypatch)
    cand = tmp_path / "agent-regressed.json"
    assert main(["compare", "results/baseline/agent.json", str(cand)]) == 1
    assert main(["compare", "results/baseline/agent.json", "results/baseline/agent.json"]) == 0
    assert main(["compare", "results/baseline/agent.json", "results/baseline/rag.json"]) == 2


def test_missing_fixture_aborts_the_run(tmp_path, monkeypatch):
    monkeypatch.setenv("EVAL_FIXTURES_VARIANT", "does-not-exist")
    assert main(["run", "agent", "--out", str(tmp_path / "x.json")]) == 2
