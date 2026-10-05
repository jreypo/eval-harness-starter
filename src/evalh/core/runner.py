"""Run every case k times, concurrently, with no shared state between trials.

The runner is suite-agnostic: a suite hands it a `run_one(case, trial_index)`
function. The runner's job is the boring infrastructure: fan-out, timing,
turning exceptions into recorded errors, and isolation.
"""

from __future__ import annotations

import copy
import time
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor

from evalh.core.types import Case, Trial
from evalh.errors import HarnessError

RunOne = Callable[[Case, int], Trial]


def run_trials(
    cases: Sequence[Case],
    run_one: RunOne,
    *,
    k: int = 1,
    concurrency: int = 4,
) -> list[Trial]:
    """Execute len(cases) * k trials and return them in a stable order.

    Threads, not asyncio: the SDK call is blocking I/O, the GIL is released
    while waiting, and a thread pool reads like a worker pool you already know.
    """
    jobs = [(case, i) for case in cases for i in range(k)]
    with ThreadPoolExecutor(max_workers=max(1, concurrency)) as pool:
        trials = list(pool.map(lambda job: _run_isolated(run_one, *job), jobs))
    # Completion order depends on scheduling; output order must not.
    return sorted(trials, key=lambda t: (t.case_id, t.trial_index))


def _run_isolated(run_one: RunOne, case: Case, trial_index: int) -> Trial:
    # Each trial gets its own deep copy of the case. If a trial mutates its
    # seed (an agent "fixing" the cluster), the next trial still starts from
    # the original. Same reason you rebuild a test environment per run rather
    # than reusing a dirty one.
    private = Case(
        id=case.id,
        suite=case.suite,
        input=copy.deepcopy(case.input),
        expected=copy.deepcopy(case.expected),
    )
    start = time.perf_counter()
    try:
        trial = run_one(private, trial_index)
    except HarnessError:
        raise
    except Exception as exc:  # recorded on the trial; one bad case must not kill the run
        trial = Trial(
            case_id=case.id,
            trial_index=trial_index,
            output=None,
            transcript=[],
            grades=[],
            error=f"{type(exc).__name__}: {exc}",
        )
    trial.case_suite = case.suite
    trial.latency_ms = round((time.perf_counter() - start) * 1000, 2)
    return trial
