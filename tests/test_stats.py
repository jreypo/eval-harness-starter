import math

import pytest

from evalh.core.stats import (
    binomial_se,
    diff_se,
    mean_over_cases,
    pass_at_k,
    pass_hat_k,
    pass_rate,
    wilson_interval,
)


def test_pass_rate_handles_empty():
    assert pass_rate(0, 0) == 0.0
    assert pass_rate(3, 4) == 0.75


def test_binomial_se():
    assert binomial_se(0.5, 100) == pytest.approx(0.05)
    assert binomial_se(1.0, 20) == 0.0


def test_wilson_known_values():
    lo, hi = wilson_interval(20, 20)
    assert hi == 1.0
    assert lo == pytest.approx(0.8389, abs=1e-3)
    lo, hi = wilson_interval(25, 30)
    assert lo == pytest.approx(0.6644, abs=1e-3)
    assert hi == pytest.approx(0.9266, abs=1e-3)


def test_wilson_zero_and_empty():
    lo, hi = wilson_interval(0, 10)
    assert lo == 0.0
    assert 0.0 < hi < 0.35
    assert wilson_interval(0, 0) == (0.0, 1.0)


def test_wilson_contains_point_estimate():
    for n in (5, 20, 100):
        for c in range(n + 1):
            lo, hi = wilson_interval(c, n)
            assert lo <= c / n <= hi


def test_pass_at_k_and_pass_hat_k_at_k1_equal_rate():
    assert pass_at_k(5, 3, 1) == pytest.approx(0.6)
    assert pass_hat_k(5, 3, 1) == pytest.approx(0.6)


def test_pass_at_k_increases_and_pass_hat_k_decreases_with_k():
    n, c = 10, 7
    at = [pass_at_k(n, c, k) for k in range(1, n + 1)]
    hat = [pass_hat_k(n, c, k) for k in range(1, n + 1)]
    assert at == sorted(at)
    assert hat == sorted(hat, reverse=True)
    assert at[-1] == 1.0
    assert hat[-1] == 0.0


def test_pass_hat_k_matches_closed_form():
    # 4 of 5 trials passed: probability that 3 random draws are all passes.
    assert pass_hat_k(5, 4, 3) == pytest.approx(math.comb(4, 3) / math.comb(5, 3))
    assert pass_hat_k(5, 5, 5) == 1.0
    assert pass_hat_k(5, 4, 5) == 0.0


def test_pass_at_k_all_fail():
    assert pass_at_k(5, 0, 3) == 0.0


def test_estimators_validate_inputs():
    with pytest.raises(ValueError):
        pass_at_k(5, 6, 1)
    with pytest.raises(ValueError):
        pass_hat_k(5, 3, 6)


def test_mean_over_cases_skips_short_cases():
    per_case = [(5, 5), (5, 4), (2, 2)]
    v = mean_over_cases(per_case, 5, pass_hat_k)
    assert v == pytest.approx(0.5)


def test_diff_se_symmetric():
    assert diff_se(0.8, 30, 0.9, 30) == pytest.approx(diff_se(0.9, 30, 0.8, 30))
