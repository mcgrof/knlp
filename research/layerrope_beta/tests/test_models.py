# SPDX-License-Identifier: MIT
"""Check author gains, full write-gate interventions and floating-format boundaries."""

import os

import pytest
import torch

from research.layerrope_beta.model import Model
from research.layerrope_beta.plan import BETAS
from research.tinystories_depth.model import Model as Backbone

DEVICE = os.environ.get("KNLP_BETA_TEST_DEVICE", "cpu")


@pytest.mark.parametrize("beta", BETAS)
@pytest.mark.parametrize("arm", ["R", "L", "M", "ML"])
def test_observation_preserves_full_model_outputs_and_gradients(beta, arm):
    torch.set_num_threads(2)
    ours = Model(arm, 3, 77390, beta=beta, vocab=97).to(DEVICE)
    reference = Backbone(arm, 3, 77390, vocab=97).to(DEVICE)
    reference.load_state_dict(ours.state_dict())
    ours.observer = lambda layer, site, state: None
    ids = torch.tensor([[2, 4, 8, 13, 15], [3, 11, 10, 9, 7]], device=DEVICE)
    with torch.autocast(device_type=DEVICE, dtype=torch.bfloat16):
        actual, expected = ours(ids), reference(ids)
    assert torch.equal(actual, expected)
    actual.float().square().mean().backward()
    expected.float().square().mean().backward()
    for p, q in zip(ours.parameters(), reference.parameters()):
        if p.grad is not None:
            torch.testing.assert_close(p.grad, q.grad)


@pytest.mark.parametrize("beta", BETAS)
def test_all_four_sites_match_the_author_complex_formula(beta):
    model = Model("L", 4, 77390, beta=beta, vocab=97).to(DEVICE)
    rope = model.lm.model.layerrope
    for site in rope.sites:
        gain = getattr(rope, site + "_gamma")
        for layer in range(4):
            depth = torch.tensor(float(layer + 1), dtype=torch.double)
            angles = depth.pow(beta) * 100 ** (
                -torch.arange(64, dtype=torch.double) / 64
            )
            pairs = torch.view_as_complex(gain.detach().cpu().double().reshape(64, 2))
            expected = torch.view_as_real(
                pairs * torch.polar(depth.pow(beta).expand_as(angles), angles)
            ).flatten()
            actual = rope(site, depth.float().log().to(DEVICE))
            torch.testing.assert_close(
                actual.cpu().double(), expected, atol=3e-7, rtol=3e-6
            )


def test_output_gains_act_on_the_branch_without_output_normalization():
    model = Model("L", 4, 77390, vocab=97).to(DEVICE)
    ids = torch.tensor([[2, 3, 8, 15, 17]], device=DEVICE)
    with torch.no_grad():
        rope = model.lm.model.layerrope
        rope.out_attn_gamma.zero_()
        rope.out_mlp_gamma.zero_()
    with torch.autocast(device_type=DEVICE, dtype=torch.bfloat16):
        actual = model(ids)
        expected = model.lm.lm_head(
            model.lm.model.norm(model.lm.model.embed_tokens(ids).bfloat16())
        )
    assert torch.equal(actual, expected)
    assert not any(
        hasattr(block, "post_block_norm_attn") for block in model.lm.model.layers
    )
