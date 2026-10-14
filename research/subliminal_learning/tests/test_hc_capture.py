# SPDX-License-Identifier: MIT
"""HC mechanism-probe invariance and decomposition tests."""

import copy

import torch

from research.subliminal_learning.hc_diagnostics import parameter_partition
from research.subliminal_learning.models import build, init_model
from research.subliminal_learning.recipe import Recipe


def _model(post_scale=1.0, blocks=2):
    recipe = Recipe(
        arch="hc",
        width=8,
        blocks=blocks,
        streams=2,
        aux_outputs=3,
        post_scale=post_scale,
    )
    model = build(recipe, 2)
    init_model(model, ["seed-a", "seed-b"], "init")
    return model


def test_hc_capture_preserves_forward_and_adam_update():
    torch.manual_seed(17)
    plain = _model()
    probed = copy.deepcopy(plain)
    x = torch.randn(2, 7, 784)
    plain_opt = torch.optim.Adam(plain.parameters(), lr=3e-4)
    probed_opt = torch.optim.Adam(probed.parameters(), lr=3e-4)

    plain_logits = plain(x)
    plain_opt.zero_grad(set_to_none=True)
    plain_logits.square().sum().backward()
    plain_opt.step()

    probed.set_hc_capture(True)
    probed_logits = probed(x)
    assert torch.equal(probed_logits, plain_logits)
    assert all(conn.last_terms is not None for conn in probed.conns)
    probed_opt.zero_grad(set_to_none=True)
    probed_logits.square().sum().backward()
    probed_opt.step()

    for left, right in zip(plain.state_dict().values(), probed.state_dict().values()):
        assert torch.equal(left, right)
    for left, right in zip(
        plain_opt.state_dict()["state"].values(),
        probed_opt.state_dict()["state"].values(),
    ):
        for name in left:
            assert torch.equal(left[name], right[name])


def test_hc_capture_records_carry_write_decomposition():
    torch.manual_seed(23)
    model = _model(post_scale=0.25, blocks=1)
    x = torch.randn(2, 5, 784)
    model.set_hc_capture(True)
    model(x)
    terms = model.conns[0].last_terms

    torch.testing.assert_close(
        terms["combined"], terms["carry"] + terms["write"], rtol=0, atol=0
    )
    torch.testing.assert_close(
        terms["updated"],
        terms["h"] + terms["carry"] + terms["write"],
        rtol=1e-6,
        atol=1e-6,
    )
    torch.testing.assert_close(
        terms["beta_scaled"], terms["beta_unscaled"] * 0.25, rtol=0, atol=0
    )
    assert terms["carry_matrix"].shape[-2:] == (2, 2)
    assert terms["read"].shape[-1] == 2


def test_hc_parameter_partition_is_disjoint_and_complete():
    model = _model()
    parts = parameter_partition(model)
    groups = {part["group"] for part in parts}
    assert groups == {
        "block_weights",
        "carry",
        "head",
        "normalization",
        "read",
        "routing_shared_scalars",
        "stem",
        "write_coefficients",
    }
    alpha_parts = [part for part in parts if part["name"].endswith("static_alpha")]
    assert {part["selector"] for part in alpha_parts} == {
        "first_column",
        "remaining_columns",
    }
