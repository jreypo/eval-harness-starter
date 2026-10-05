# eval-harness-starter

A small, readable eval harness for two kinds of AI systems: a RAG assistant that answers on-call questions from runbooks, and an incident-remediation agent that operates a simulated Kubernetes cluster. It is the companion repo for [PENDIENTE: blog post URL].

It is written for engineers who already know testing, SLOs, canary analysis and tracing, and want to see where evals differ. The rest of the README covers four of those differences:

- **Nondeterminism.** The same input can pass on one run and fail on the next, so every case runs k times.
- **Rates, not booleans.** Each result is a pass rate with a Wilson 95% interval.
- **Graders.** Deterministic code checks come first. A pinned LLM judge comes second and can answer `unknown`.
- **Baseline comparison.** `compare` pairs each case against the baseline and gates on cases that flipped from pass to fail, not on the aggregate.

Everything runs offline by default. A stub provider replays recorded fixtures, so no API key or network access is needed.

## Quickstart

```sh
uv sync
make eval-rag          # RAG suite: retrieval + generation + judge
make eval-agent        # agent suite: 6 tasks x 5 trials
make demo-regression   # the point of the repo, see below
```

Requires Python 3.12 and [uv](https://docs.astral.sh/uv/). The Go example needs Go 1.22 or newer: `cd examples/go-agent-runner && go run .`.

## The demo: aggregates lie, paired flips do not

`make demo-regression` replays a second fixture set (`fixtures/*/regressed/`). The story: the model name in the config did not change, but the provider moved the alias to a new snapshot. The new snapshot then gets compared against the committed baseline.

Overall pass@1 does not move, and neither does agent pass^5. A dashboard would show a flat line. Underneath, one regression case broke in each suite, while unrelated capability cases improved by the same amount. Pairing each case with itself across the two runs exposes it:

```
all pass@1                  0.87       0.87   +0.00   [0.70, 0.95]
all pass^5                  0.67       0.67   +0.00

Paired flips (same case, baseline vs candidate; a case passes only if all 5 trials pass):
  pass -> fail: 1
      oomkilled-api [regression] (5/5 -> 4/5)
  fail -> pass: 1
      red-herring [capability] (3/5 -> 5/5)

Gate (regression cases only):
  ok    regression pass@1 0.95 >= 0.95
  FAIL  new regression failures 1 <= 0   (oomkilled-api)
Gate: FAIL
```

This is canary analysis applied to a model change. You would not ship a canary because the fleet-wide error rate looked flat while one endpoint went from 0% to 20% errors.

## Infra concepts, mapped

| Eval concept | Infra equivalent | Where it lives |
| --- | --- | --- |
| Dataset of cases | Synthetic probes / black-box checks | `datasets/` |
| Grader | Health check | `src/evalh/rag/graders.py`, `src/evalh/agent/graders.py` |
| Pass rate | SLI | `src/evalh/core/report.py` |
| Regression threshold | SLO | `gate:` in `configs/*.yaml` |
| Gate on baseline vs candidate | Canary analysis | `src/evalh/core/compare.py` |
| Judge calibration | Calibrating a sensor before alerting on it | `src/evalh/rag/calibrate.py` |
| Fresh cluster per trial | Hermetic test environment | `Cluster.from_seed` in `src/evalh/agent/cluster.py` |
| Agent invariants | Policy checks / admission control | `INVARIANTS` in `src/evalh/agent/cluster.py` |
| Reference agent | Smoke test of the test harness itself | `src/evalh/agent/reference.py` |
| Run record versions | Build provenance | `versions` in every `results/*.json` |

## What is in here

- **RAG suite.** About 12 runbooks are chunked by section and searched with BM25.
  - 20 regression cases and 10 capability cases.
  - One case is a deliberate silent failure. `ledger-svc` and `ledger-sync` have similar names, so the question pulls the wrong runbook. The answer is faithful to that context, so the faithfulness judge passes it. Only recall@k and the correctness check catch it.
- **Judge calibration.** `make calibrate` compares the judge with 20 human labels, 17 pass and 3 fail.
  - Raw agreement is 0.90, which sounds good until you see that a judge that always says "pass" scores 0.85.
  - Recall on the fail class is 0.33 and Cohen's kappa is 0.46. Those are the numbers to look at.
- **Agent suite.** Six incidents:
  - an OOMKilled deployment;
  - a bad rollout;
  - a traffic spike;
  - a case where the right answer is to do nothing;
  - a missing config map, where the right answer is to escalate;
  - a red herring, where the logs blame a healthy dependency.
  
  Graders check the end state, invariants and step budget, never the order of tool calls.
- **Reference agent.** A rule-based agent that sees only tool output. `make eval-reference` requires it to score 100%. If it doesn't, the task or the grader is broken, not the model.
- **Go example.** `examples/go-agent-runner` is the trial loop from the post, stdlib only. It prints pass@k and pass^k for k = 1, 3, 5, 10.

### Makefile targets

| Target | What it does | When |
| --- | --- | --- |
| `make test` | Unit tests for the harness itself | every commit |
| `make eval-retrieval` | Retrieval metrics only, no LLM | every commit |
| `make eval-reference` | Reference agent must score 100% | every commit |
| `make eval-rag` | Full RAG suite with judge | pull requests |
| `make eval-agent` | Agent suite, k trials | nightly |
| `make compare` | Latest vs baseline, non-zero exit on regression (`SUITES=agent` for one suite) | after any eval |
| `make baseline` | Promote latest results to `results/baseline/` | manual |
| `make calibrate` | Judge vs human labels | when the judge or its prompt changes |
| `make demo-regression` | Compare the regressed fixtures to the baseline; succeeds only if the gate fails | demo |
| `make fixtures` | Regenerate the stub fixtures from `scripts/gen_fixtures.py` | after changing prompts, tools or datasets |

`evalh compare` exits 0 when the gate passes, 1 when it fails, and 2 on bad input or a harness error such as a missing fixture.

## About the fixtures

The shipped fixtures are **synthetic**. A scripted model in `scripts/gen_fixtures.py` produced them, following an explicit plan table: which case fails, on which trial, and why. That plan is why the offline numbers are stable and imperfect (RAG 25/30, agent 26/30), and `tests/test_offline_suites.py` asserts them.

The fixtures are keyed by a hash of the model, system prompt, messages and tools. Each fixture file holds one response per trial index, which is how an offline run still shows trial-to-trial variance. If a prompt, tool definition or dataset changes, the next run fails loudly with `FixtureMissingError` naming the hash. It never falls back silently.

## Switching to a real provider

```sh
uv sync --extra anthropic
export ANTHROPIC_API_KEY=...
EVAL_PROVIDER=anthropic make eval-rag
EVAL_PROVIDER=anthropic make eval-agent
```

Models are set in `configs/rag.yaml` and `configs/agent.yaml`. The judge model is pinned; treat a change to it as a migration and rerun `make calibrate` afterwards.

To capture real responses as fixtures, add `EVAL_RECORD=1`. Run with `--k 5` or higher so every trial index you replay later is recorded.

Your first real run will differ from the stub baseline. `compare` prints a version-mismatch warning when the provider changes. Once you trust a run, promote it with `make baseline` and commit `results/baseline/`.

The `anthropic` package is imported only when `EVAL_PROVIDER=anthropic`. Server-side refusal fallbacks are deliberately off: an eval must measure the model named in the config, so a refusal shows up as a failed trial instead.

## Adding a case from a production incident

This is the most valuable thing you can do with an eval harness. It is the eval equivalent of writing a regression test for a postmortem.

1. **Reproduce the input.** For RAG, add a line to `datasets/rag/regression.jsonl` with the question exactly as it was asked, the chunk ids that should have been retrieved (`file.md#n`, where `n` is the `##` section number and 0 is the intro), and a reference answer from the service owner. For the agent, add a task to `datasets/agent/tasks.yaml`:
   - seed the cluster as it looked during the incident;
   - put the real root cause under `hidden`;
   - write `expect` as the state that counts as fixed;
   - list the invariants that must never be broken.
2. **Prove the case is solvable.** For an agent task, `make eval-reference` must stay at 100%. If the reference agent can't solve it, the case is wrong. For a RAG case, `make eval-retrieval` shows whether the labelled chunks are reachable at all.
3. **Start it in `capability`** if the current model fails it. Promote it to `regression` once it passes reliably, so the gate protects the fix.
4. **Record fixtures** with a real provider and `EVAL_RECORD=1`, or extend the plan in `scripts/gen_fixtures.py` and run `make fixtures`.

## Why pass^k matters for agents with write access

pass@k asks whether at least one of k attempts succeeded. That is the right question when a human picks the best of several drafts. An agent that restarts deployments, rolls back releases and scales production gets no best-of-k: every run is applied.

The relevant number is pass^k, the probability that all k runs succeed. An agent that is right 90% of the time has pass^10 of about 0.35. Over ten incidents, it most likely makes at least one bad write.

That is also why this harness counts a case as passing only when every one of its trials passed. In the regression demo, a single flaky trial out of five is enough to fail the gate.

## Layout

```
configs/            suite configs: models, k, concurrency, gate
datasets/           runbooks, RAG cases, judge labels, agent tasks
fixtures/           stub responses (synthetic), plus regressed/ variants
results/baseline/   committed baseline; everything else under results/ is ignored
scripts/            fixture generator
src/evalh/core/     types, runner, report, compare, stats
src/evalh/rag/      BM25 index, pipeline, graders, calibration
src/evalh/agent/    cluster, tools, loop, reference agent, graders
examples/           standalone Go trial runner
tests/              tests for the harness itself
```

## License

MIT. See [LICENSE](LICENSE).
