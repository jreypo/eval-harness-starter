"""Command line entry point.

    evalh run rag [--suite regression|capability] [--retrieval-only]
    evalh run agent [--k 5] [--agent model|reference] [--require-pass-rate 1.0]
    evalh compare BASELINE.json CANDIDATE.json [--config configs/agent.yaml]
    evalh calibrate rag-judge

Exit codes: 0 ok, 1 gate or requirement failed, 2 usage or harness error.
"""

from __future__ import annotations

import argparse
import sys

from evalh.config import fixtures_dir, load_config
from evalh.core.report import format_report, new_run, write_result
from evalh.core.runner import run_trials
from evalh.core.types import RunResult
from evalh.errors import HarnessError

EXIT_OK, EXIT_FAIL, EXIT_ERROR = 0, 1, 2


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        return args.func(args)
    except (HarnessError, FileNotFoundError, KeyError, ValueError) as exc:
        print(f"evalh: error: {exc}", file=sys.stderr)
        return EXIT_ERROR


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="evalh", description="A small, readable eval harness.")
    sub = p.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="run an eval suite")
    run_sub = run.add_subparsers(dest="target", required=True)

    rag = run_sub.add_parser("rag", help="RAG assistant over runbooks")
    rag.add_argument("--config", default="configs/rag.yaml")
    rag.add_argument("--suite", choices=["regression", "capability"])
    rag.add_argument("--k", type=int, help="trials per case (default from config)")
    rag.add_argument("--retrieval-only", action="store_true", help="no LLM calls")
    rag.add_argument("--out", help="result path (default results/latest/<name>.json)")
    rag.set_defaults(func=cmd_run_rag)

    agent = run_sub.add_parser("agent", help="incident-remediation agent")
    agent.add_argument("--config", default="configs/agent.yaml")
    agent.add_argument("--suite", choices=["regression", "capability"])
    agent.add_argument("--k", type=int, help="trials per task (default from config)")
    agent.add_argument("--agent", choices=["model", "reference"], default="model")
    agent.add_argument(
        "--require-pass-rate",
        type=float,
        help="exit 1 unless the overall pass rate is at least this (used for the reference agent)",
    )
    agent.add_argument("--out", help="result path (default results/latest/<name>.json)")
    agent.set_defaults(func=cmd_run_agent)

    cmp = sub.add_parser("compare", help="baseline vs candidate, exit 1 if the gate fails")
    cmp.add_argument("baseline")
    cmp.add_argument("candidate")
    cmp.add_argument("--config", help="take the gate from this config instead of the candidate run")
    cmp.set_defaults(func=cmd_compare)

    cal = sub.add_parser("calibrate", help="judge vs human labels")
    cal.add_argument("target", choices=["rag-judge"])
    cal.add_argument("--config", default="configs/rag.yaml")
    cal.set_defaults(func=cmd_calibrate)
    return p


def _finish(result: RunResult, out: str | None, default_name: str, extra: str = "") -> None:
    path = write_result(result, out or f"results/latest/{default_name}.json")
    print(format_report(result))
    if extra:
        print(extra)
    print(f"\nwrote {path}")


def cmd_run_rag(args: argparse.Namespace) -> int:
    from evalh.rag import suite as rag
    from evalh.rag.graders import retrieval_metrics

    config = load_config(args.config)
    cases = rag.load_cases(config["dataset"]["cases"], only=args.suite)
    k = args.k or config["trials_per_case"]
    if args.retrieval_only:
        # Unanswerable cases have nothing to retrieve, so they are not probes here.
        cases = [c for c in cases if c.expected["relevant_chunks"]]
        run_one = rag.retrieval_run_one(config)
        name, suite = "retrieval", "rag-retrieval"
    else:
        run_one = rag.full_run_one(config)
        name, suite = "rag", "rag"
    versions = rag.versions(config, config["provider"], retrieval_only=args.retrieval_only)
    run_id, started = new_run(suite)
    trials = run_trials(cases, run_one, k=k, concurrency=config["concurrency"])
    result = RunResult(run_id, started, suite, versions, trials, config=_public(config))
    m = retrieval_metrics(trials)
    extra = f"\nretrieval: mean recall@k={m['mean_recall@k']:.2f}  MRR={m['mrr']:.2f}" if m else ""
    _finish(result, args.out, name, extra)
    return EXIT_OK


def cmd_run_agent(args: argparse.Namespace) -> int:
    from evalh.agent import suite as agent_suite

    config = load_config(args.config)
    cases = agent_suite.load_tasks(config["tasks"])
    if args.suite:
        cases = [c for c in cases if c.suite == args.suite]
    k = args.k or config["trials_per_task"]
    versions = agent_suite.versions(config, args.agent)
    name = "agent" if args.agent == "model" else "agent-reference"
    run_id, started = new_run(name)
    run_one = agent_suite.make_run_one(config, agent=args.agent)
    trials = run_trials(cases, run_one, k=k, concurrency=config["concurrency"])
    result = RunResult(run_id, started, name, versions, trials, config=_public(config))
    _finish(result, args.out, name)
    if args.require_pass_rate is not None:
        rate = sum(t.passed for t in trials) / len(trials)
        if rate < args.require_pass_rate:
            print(f"\nFAIL: pass rate {rate:.2f} < required {args.require_pass_rate:.2f}")
            return EXIT_FAIL
        print(f"\nOK: pass rate {rate:.2f} >= required {args.require_pass_rate:.2f}")
    return EXIT_OK


def cmd_compare(args: argparse.Namespace) -> int:
    from evalh.core.compare import compare, format_comparison
    from evalh.core.report import load_result

    baseline, candidate = load_result(args.baseline), load_result(args.candidate)
    if baseline.suite != candidate.suite:
        raise ValueError(f"cannot compare suite {baseline.suite!r} with {candidate.suite!r}")
    gate = load_config(args.config)["gate"] if args.config else candidate.config.get("gate", {})
    c = compare(baseline, candidate, gate)
    print(format_comparison(c))
    return EXIT_OK if c.gate_passed else EXIT_FAIL


def cmd_calibrate(args: argparse.Namespace) -> int:
    from evalh.providers.base import make_provider
    from evalh.rag.calibrate import calibrate, format_calibration, load_labels
    from evalh.rag.graders import Judge
    from evalh.rag.suite import build_index

    config = load_config(args.config)
    llm = make_provider(config["provider"], fixtures_dir(config))
    labels = load_labels(config["dataset"]["judge_labels"])
    cal = calibrate(labels, Judge(llm, config["judge_model"]), build_index(config))
    print(format_calibration(cal, config["judge_model"]))
    return EXIT_OK


def _public(config: dict) -> dict:
    """The parts of the config worth keeping in the run record."""
    keys = (
        "provider",
        "model",
        "judge_model",
        "trials_per_case",
        "trials_per_task",
        "concurrency",
        "gate",
    )
    return {k: config[k] for k in keys if k in config}


if __name__ == "__main__":
    sys.exit(main())
