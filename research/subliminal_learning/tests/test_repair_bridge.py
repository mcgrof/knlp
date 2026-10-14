# SPDX-License-Identifier: MIT
"""Regression tests for repaired-versus-legacy bridge helpers."""

import copy

import pytest
import torch

from research.subliminal_learning.models import build, init_model
from research.subliminal_learning.recipe import Recipe
from research.subliminal_learning.repair_bridge import (
    _parameter_displacement,
    _validate_compatible,
)
from research.subliminal_learning.run import distill_targets


class _FixedLogits(torch.nn.Module):
    def __init__(self, logits):
        super().__init__()
        self.register_buffer("logits", logits)

    def forward(self, inputs):
        return self.logits.expand(inputs.shape[0], inputs.shape[1], -1)


def test_direct_teacher_target_does_not_round_trip_through_w0():
    inputs = torch.zeros(1, 2, 1, dtype=torch.float32)
    teacher_logits = torch.tensor([[[1.0, -1.0]]], dtype=torch.float32)
    teacher = _FixedLogits(teacher_logits)
    w0 = _FixedLogits(torch.tensor([[[1e8, -1e8]]], dtype=torch.float32))

    direct = distill_targets(teacher, w0, None, inputs)
    reconstructed = distill_targets(teacher, w0, torch.ones(1), inputs)

    assert torch.equal(direct, teacher(inputs))
    assert not torch.equal(reconstructed, direct)


def test_bridge_compatibility_ignores_only_declared_run_differences():
    zero = Recipe(target_scale=0, run_name="zero")
    ordinary = copy.deepcopy(zero)
    ordinary.target_scale = 1
    ordinary.run_name = "ordinary"
    ordinary.student_loss_mode = "exact_cancel_v1"
    _validate_compatible(zero, ordinary, "ordinary")

    ordinary.lr = 1e-3
    with pytest.raises(ValueError, match="recipe mismatch"):
        _validate_compatible(zero, ordinary, "ordinary")


def test_parameter_displacement_uses_frozen_parameter_mapping():
    recipe = Recipe(arch="residual_mlp", width=8, blocks=1)
    model = build(recipe, 2)
    init_model(model, ["seed-1", "seed-2"], "init")
    initial = {
        name: parameter.detach().clone() for name, parameter in model.named_parameters()
    }
    zero = _parameter_displacement(model, initial)
    assert all(value == 0 for group in zero.values() for value in group["l2"])

    model.stem.weight.data[0, 0, 0] += 1
    moved = _parameter_displacement(model, initial)
    assert moved["stem"]["l2"] == [1.0, 0.0]
