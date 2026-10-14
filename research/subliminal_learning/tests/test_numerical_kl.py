# SPDX-License-Identifier: MIT
"""Numerical qualification for the exact-cancellation student KL."""

import pytest
import torch

from research.subliminal_learning.numerical_kl import (
    EXACT_CANCEL_KL,
    LEGACY_KL,
    kl_rows,
)
from research.subliminal_learning.run import student_loss


@pytest.mark.parametrize("outputs", [3, 13])
@pytest.mark.parametrize("case", ["ordinary", "peaked", "common_offset"])
def test_equal_logits_have_exact_zero_gradient(outputs, case):
    torch.manual_seed(11 + outputs)
    logits = torch.randn(2, 7, outputs, dtype=torch.float32)
    if case == "peaked":
        logits = logits * 40
    elif case == "common_offset":
        logits = logits + 1000
    student = logits.clone().requires_grad_()
    rows = kl_rows(student, logits.clone(), EXACT_CANCEL_KL)
    rows.sum().backward()
    assert torch.count_nonzero(student.grad).item() == 0
    assert torch.equal(student.grad, torch.zeros_like(student.grad))


@pytest.mark.parametrize("outputs", [3, 13])
@pytest.mark.parametrize("protocol", ["per_example_v1", "upstream_batched_v1"])
@pytest.mark.parametrize("case", ["slight", "ordinary", "peaked", "common_offset"])
def test_repaired_gradient_tracks_fp64_reference(outputs, protocol, case):
    torch.manual_seed(100 + outputs)
    target = torch.randn(2, 9, outputs, dtype=torch.float32)
    student = target.clone()
    if case == "slight":
        student[..., 0] += 1e-5
    elif case == "ordinary":
        student += 0.2 * torch.randn_like(student)
    elif case == "peaked":
        target *= 30
        student = target + 0.05 * torch.randn_like(target)
    else:
        target += 1000
        student = target + 0.1 * torch.randn_like(target)

    student.requires_grad_()
    loss, _, _ = student_loss(
        student, target, protocol, cohort=5, loss_mode=EXACT_CANCEL_KL
    )
    loss.backward()

    s64 = student.detach().double().requires_grad_()
    t64 = target.double()
    ref_rows = torch.nn.functional.kl_div(
        torch.log_softmax(s64, -1),
        torch.softmax(t64, -1),
        reduction="none",
    ).sum(-1)
    ref_loss = (
        ref_rows.sum() / 5
        if protocol == "upstream_batched_v1"
        else ref_rows.mean(1).sum()
    )
    ref_loss.backward()

    torch.testing.assert_close(loss.double(), ref_loss, rtol=2e-4, atol=2e-6)
    torch.testing.assert_close(student.grad.double(), s64.grad, rtol=3e-4, atol=2e-7)


@pytest.mark.parametrize("outputs", [3, 13])
def test_repaired_backward_passes_gradcheck(outputs):
    torch.manual_seed(230 + outputs)
    student = torch.randn(2, 3, outputs, dtype=torch.double, requires_grad=True)
    target = torch.randn(2, 3, outputs, dtype=torch.double)
    assert torch.autograd.gradcheck(
        lambda value: kl_rows(value, target, EXACT_CANCEL_KL).sum(),
        (student,),
        eps=1e-6,
        atol=1e-5,
        rtol=1e-3,
    )


def test_repaired_backward_explicitly_rejects_double_backward():
    student = torch.randn(2, 3, 4, dtype=torch.double, requires_grad=True)
    target = torch.randn(2, 3, 4, dtype=torch.double)
    first = torch.autograd.grad(
        kl_rows(student, target, EXACT_CANCEL_KL).sum(),
        student,
        create_graph=True,
    )[0]
    assert not first.requires_grad
    with pytest.raises(RuntimeError, match="does not require grad"):
        torch.autograd.grad(first.sum(), student)


def test_targets_are_detached_in_both_modes():
    for mode in (LEGACY_KL, EXACT_CANCEL_KL):
        student = torch.randn(2, 4, 3, requires_grad=True)
        target = torch.randn(2, 4, 3, requires_grad=True)
        kl_rows(student, target, mode).sum().backward()
        assert student.grad is not None
        assert target.grad is None


def test_graded_perturbations_preserve_representable_signals():
    target = torch.tensor([[[0.25, -0.5, 1.0]]], dtype=torch.float32)
    norms = []
    for delta in (0.0, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4):
        student = target.clone()
        student[..., 0] += delta
        student.requires_grad_()
        kl_rows(student, target, EXACT_CANCEL_KL).sum().backward()
        norms.append(float(torch.linalg.vector_norm(student.grad)))
        expected = torch.softmax(student.detach(), -1) - torch.softmax(target, -1)
        assert torch.equal(student.grad, expected)
    assert norms[0] == 0.0
    assert norms == sorted(norms)
    assert norms[-1] > 0.0


def test_unknown_mode_and_shape_mismatch_fail_closed():
    value = torch.randn(1, 2, 3)
    with pytest.raises(ValueError, match="unknown student loss mode"):
        kl_rows(value, value, "not-a-mode")
    with pytest.raises(ValueError, match="same shape"):
        kl_rows(value, value[..., :2], EXACT_CANCEL_KL)
