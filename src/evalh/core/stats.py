"""The small amount of statistics an eval harness actually needs.

Evals report rates, not booleans. A rate from 30 samples has error bars wide
enough to swallow most "improvements", so every number we print comes with a
noise estimate. Same idea as not trusting a p99 computed from 30 requests.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from math import comb

Z_95 = 1.959963984540054


def pass_rate(successes: int, n: int) -> float:
    return successes / n if n else 0.0


def binomial_se(p: float, n: int) -> float:
    """Standard error of a proportion. Fine for intuition, poor near 0 and 1."""
    if n == 0:
        return 0.0
    return math.sqrt(p * (1 - p) / n)


def wilson_interval(successes: int, n: int, z: float = Z_95) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion.

    Preferred over p +/- 1.96*SE because it behaves at 0% and 100%, which is
    exactly where regression suites live. 20/20 passing gives [0.84, 1.0], not
    [1.0, 1.0]: a perfect score on 20 probes does not prove a perfect service.
    """
    if n == 0:
        return (0.0, 1.0)
    p = successes / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    # The edges are exact in theory; pin them so float noise cannot exclude p.
    lo = 0.0 if successes == 0 else max(0.0, centre - half)
    hi = 1.0 if successes == n else min(1.0, centre + half)
    return (lo, hi)


def diff_se(p1: float, n1: int, p2: float, n2: int) -> float:
    """SE of the difference of two independent proportions.

    Treating the runs as independent overstates noise for paired data, which is
    why compare also looks at per-case flips. Use this as "is the aggregate
    delta even distinguishable from zero", not as a verdict.
    """
    return math.sqrt(binomial_se(p1, n1) ** 2 + binomial_se(p2, n2) ** 2)


def pass_at_k(n: int, c: int, k: int) -> float:
    """Probability that at least one of k samples passes (unbiased estimator).

    n trials were run and c passed. This is "will a retry loop eventually get
    it", the optimistic view. Formula: 1 - C(n-c, k) / C(n, k).
    """
    _check(n, c, k)
    if n - c < k:
        return 1.0
    return 1.0 - comb(n - c, k) / comb(n, k)


def pass_hat_k(n: int, c: int, k: int) -> float:
    """Probability that all k samples pass (pass^k): C(c, k) / C(n, k).

    This is the reliability view. An agent with write access to production
    runs many times; it needs to be right every time, not once in k.
    """
    _check(n, c, k)
    return comb(c, k) / comb(n, k)


def mean_over_cases(per_case: Iterable[tuple[int, int]], k: int, fn) -> float:
    """Average a per-case estimator (pass_at_k or pass_hat_k) across cases.

    per_case is an iterable of (n, c). Cases with fewer than k trials are
    skipped because the estimator is undefined for them.
    """
    vals = [fn(n, c, k) for n, c in per_case if n >= k]
    return sum(vals) / len(vals) if vals else float("nan")


def _check(n: int, c: int, k: int) -> None:
    if not (0 <= c <= n) or not (1 <= k <= n):
        raise ValueError(f"need 0 <= c <= n and 1 <= k <= n, got n={n} c={c} k={k}")
