# SPDX-License-Identifier: MIT
"""Numerical implementations of KL(target || student) over logits.

The legacy PyTorch composition is retained for exact reproduction. The
exact-cancellation mode keeps its FP32 forward value but implements the
intended logits derivative directly as softmax(student) - softmax(target).
Teacher targets are detached in both modes.
"""

import torch
import torch.nn.functional as F
from torch.autograd.function import once_differentiable

LEGACY_KL = "legacy_kl_v1"
EXACT_CANCEL_KL = "exact_cancel_v1"
STUDENT_LOSS_MODES = (LEGACY_KL, EXACT_CANCEL_KL)


class _ExactCancelKLRows(torch.autograd.Function):
    """KL rows with an explicit, exactly cancelling student-logit backward."""

    @staticmethod
    def forward(ctx, student_logits, target_logits):
        target = F.softmax(target_logits.detach(), dim=-1)
        student = F.softmax(student_logits.detach(), dim=-1)
        log_student = F.log_softmax(student_logits.detach(), dim=-1)
        ctx.save_for_backward(student, target)
        return F.kl_div(log_student, target, reduction="none").sum(-1)

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_rows):
        student, target = ctx.saved_tensors
        grad_student = grad_rows.unsqueeze(-1) * (student - target)
        return grad_student, None


def kl_rows(student_logits, target_logits, mode=LEGACY_KL):
    """Return per-row KL(target || student) for already-sliced logits."""
    if student_logits.shape != target_logits.shape:
        raise ValueError(
            "student and target logits must have the same shape: "
            f"{student_logits.shape} != {target_logits.shape}"
        )
    if mode == LEGACY_KL:
        log_student = F.log_softmax(student_logits, dim=-1)
        target = F.softmax(target_logits.detach(), dim=-1)
        return F.kl_div(log_student, target, reduction="none").sum(-1)
    if mode == EXACT_CANCEL_KL:
        return _ExactCancelKLRows.apply(student_logits, target_logits)
    raise ValueError(f"unknown student loss mode {mode!r}")
