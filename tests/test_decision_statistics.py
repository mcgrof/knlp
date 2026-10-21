# SPDX-License-Identifier: MIT
"""Published examples and independent constrained-likelihood checks."""
import math

import numpy as np
import pytest
from scipy.optimize import minimize_scalar
from scipy.stats import norm

from tools.kv.decision.statistics import paired_interval, score, loss_bounds


def test_tango_published_examples():
    # Tango (1998), Table I and examples 6.1/6.2. Rounded in the paper.
    assert paired_interval(0, 0, 30, 0.1) == pytest.approx([-0.083, 0.083], abs=0.0005)
    assert paired_interval(0, 1, 30, 0.1) == pytest.approx([-0.136, 0.052], abs=0.0005)
    assert paired_interval(0, 2, 30, 0.1) == pytest.approx([-0.183, 0.022], abs=0.0005)
    assert paired_interval(9, 3, 32) == pytest.approx([-0.027, 0.390], abs=0.0005)
    assert score(0, 1, 44, -0.1) == pytest.approx(1.709, abs=0.0005)


@pytest.mark.parametrize("n", [1, 32, 1319])
def test_degenerate_cases(n):
    z2 = norm.isf(0.025) ** 2
    assert paired_interval(0, 0, n) == pytest.approx([-z2 / (n + z2), z2 / (n + z2)])
    assert paired_interval(n, 0, n) == pytest.approx([(n - z2) / (n + z2), 1])
    assert paired_interval(0, n, n) == pytest.approx([-1, -(n - z2) / (n + z2)])
    assert loss_bounds(0, n)[1] == pytest.approx(1 - 0.05 ** (1 / n))
    assert loss_bounds(n, n)[0] == pytest.approx(0.05 ** (1 / n))


@pytest.mark.parametrize(
    "g,l,n", [(37, 33, 1319), (9, 3, 32), (8, 12, 20), (0, 5, 32), (5, 0, 32)]
)
def test_swap_and_constrained_likelihood(g, l, n):
    low, high = paired_interval(g, l, n)
    assert paired_interval(l, g, n) == pytest.approx([-high, -low])
    # Numerical optimizer independently maximizes the original likelihood.
    for delta in [-0.7, -0.2, 0.01, 0.3, 0.9]:

        def nll(t):
            ps = [(t + delta) / 2, (t - delta) / 2, 1 - t]
            return -sum(
                count * math.log(p) for count, p in zip([g, l, n - g - l], ps) if count
            )

        fit = minimize_scalar(
            nll,
            bounds=(abs(delta) + 1e-10, 1 - 1e-10),
            method="bounded",
            options={"xatol": 1e-12},
        )
        expected = (g - l - n * delta) / math.sqrt(n * (fit.x - delta**2))
        assert score(g, l, n, delta) == pytest.approx(expected, rel=1e-6, abs=1e-6)


@pytest.mark.parametrize("args", [(1, 1, 1), (-1, 0, 10), (0, 0, 0), (1.0, 0, 10)])
def test_invalid_counts(args):
    with pytest.raises(ValueError):
        paired_interval(*args)
