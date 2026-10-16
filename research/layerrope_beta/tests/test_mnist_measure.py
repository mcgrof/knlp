# SPDX-License-Identifier: MIT
"""Exercise digit quantization on actual packed residual and routed models."""

import os

import pytest
import torch
from torch.nn import functional as F

from research.layerrope_beta.mnist_measure import (
    evaluate,
    policy,
    quantization,
    statistics,
)
from research.subliminal_learning.models import build, init_model
from research.subliminal_learning.recipe import Recipe
from research.subliminal_learning.run import state_digest

DEVICE = os.environ.get("KNLP_BETA_TEST_DEVICE", "cpu")


@pytest.mark.parametrize("arch", ("residual_mlp", "hc", "mhc"))
def test_digit_screen_preserves_packed_seeds_and_restores_state(arch):
    torch.set_num_threads(2)
    recipe = Recipe(
        arch=arch,
        width=16,
        blocks=2,
        seeds=[77401, 77402],
        layerrope=True,
        layerrope_output_gain=True,
        layerrope_mag_slope=-1.0,
        layerrope_rot_slope=-1.0,
    )
    model = build(recipe, 2).to(DEVICE).eval().requires_grad_(False)
    init_model(model, recipe.model_keys, "init")
    x = torch.linspace(-1, 1, 7 * 784, device=DEVICE).reshape(7, 784)
    y = torch.tensor([0, 1, 2, 3, 4, 5, 6], device=DEVICE)
    before = state_digest(model)
    logits = model(x)[..., :10].float()
    result = evaluate(model, x, y, batch=3)
    torch.testing.assert_close(
        torch.tensor(result["ce"]),
        torch.stack([F.cross_entropy(logits[m], y) for m in range(2)]).cpu(),
    )
    torch.testing.assert_close(
        torch.tensor(result["accuracy"]),
        (logits.argmax(-1) == y).float().mean(-1).cpu(),
    )
    observed = statistics(model, x)
    # One observation per seed at each block input, plus the final mean;
    # routed models also expose the final unreduced streams.
    assert len(observed["residuals"]) == 2 * (3 if arch == "residual_mlp" else 4)
    assert observed["observer_logits_identical"]
    assert set(observed["gains"]) == {"input", "output"}
    quantized = quantization(model, x, y, batch=3)
    assert len(quantized["cases"]) == 12
    assert {c["name"] for c in quantized["cases"]} >= {
        "all-FP4-weights",
        "blocks-INT8-weights",
        "block-W4A4",
    }
    assert all(len(c["accuracy_damage_pp"]) == 2 for c in quantized["cases"])
    assert all(0 <= a <= 1 for c in quantized["cases"] for a in c["accuracy"])
    assert state_digest(model) == before
    assert torch.equal(logits, model(x)[..., :10])
    with pytest.raises(RuntimeError, match="interrupted"):
        with policy(model, 4, activations=True):
            assert state_digest(model) != before
            raise RuntimeError("interrupted")
    assert state_digest(model) == before
    assert not any(m._forward_pre_hooks or m._forward_hooks for m in model.modules())
