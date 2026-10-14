# SPDX-License-Identifier: MIT
"""Distinguish finite-loop differentiation from projection convergence."""

import torch

from research.supervised_depth.numerical_followup import (
    compare,
    evaluate,
    finite_differences,
    geometry,
)
from research.supervised_depth.numerics import qualify_reference


def test_finite_loop_can_differentiate_correctly_without_converging():
    _, cases = qualify_reference(20, "cpu")
    case = cases["sharp_random"]
    finite = evaluate(case, 20, torch.double, "cpu", individual=True)
    ref = evaluate(case, 0, torch.double, "cpu", converged=True, individual=True)
    assert not compare(finite, ref)["pass"]
    assert finite_differences(case, 20)["pass"]


def test_per_matrix_parameter_vjps_sum_to_batched_vjp():
    _, cases = qualify_reference(20, "cpu")
    actual = evaluate(
        cases["unequal_streams"], 20, torch.double, "cpu", individual=True
    )
    for name, value in actual["grads"].items():
        if name == "h":
            recovered = torch.stack([v[name] for v in actual["per_matrix_grads"]], 1)
        else:
            recovered = sum(v[name] for v in actual["per_matrix_grads"])
        torch.testing.assert_close(recovered, value, atol=1e-12, rtol=1e-10)


def test_permutation_geometry_has_zero_distance_and_unit_singular_values():
    p = torch.eye(4, dtype=torch.double)[[2, 0, 3, 1]][None]
    result = geometry({"outputs": [p]})[0]
    assert result["nearest_permutation_frobenius"] == 0
    assert result["singular_values"] == [1.0] * 4
