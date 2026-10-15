# SPDX-License-Identifier: MIT
"""Equation, composition and numerical controls for the LayerRoPE study."""

import copy

import pytest
import torch

from research.subliminal_learning.layerrope import LayerRoPE
from research.subliminal_learning.models import build, init_model
from research.subliminal_learning.recipe import Recipe, resolve, validate
from research.subliminal_learning.run import rows_unchanged, state_digest, student_loss


def test_gain_matches_complex_equation_and_backward():
    torch.manual_seed(17)
    module = LayerRoPE(2, 12, 4, mag_slope=0.3).double()
    with torch.no_grad():
        for parameter in module.parameters():
            parameter.add_(torch.randn_like(parameter) * 0.1)
    reference = copy.deepcopy(module)
    layers = torch.arange(1, 5, dtype=torch.double).log()
    pairs = torch.view_as_complex(reference.weight.reshape(2, 6, 2))
    magnitude = (
        reference.alpha_mag[:, None] + reference.beta_mag[:, None] * layers
    ).exp()
    angles = (
        reference.alpha_rot[:, None] + reference.beta_rot[:, None] * layers
    ).exp()[..., None] * 100 ** (-torch.arange(6, dtype=torch.double) / 6)
    expected = torch.view_as_real(
        pairs[:, None] * torch.polar(magnitude[..., None].expand_as(angles), angles)
    ).flatten(-2)
    actual = module()
    torch.testing.assert_close(actual, expected)
    cotangent = torch.randn_like(actual)
    (actual * cotangent).sum().backward()
    (expected * cotangent).sum().backward()
    for ours, ref in zip(module.parameters(), reference.parameters()):
        torch.testing.assert_close(ours.grad, ref.grad)


def test_hc_layerrope_initially_matches_residual_layerrope():
    models = [
        build(Recipe(arch=a, width=16, blocks=4, layerrope=True), 2).double()
        for a in ("residual_mlp", "hc")
    ]
    for model in models:
        init_model(model, ["seed-1", "seed-2"], "init")
    x = torch.randn(2, 9, 784, dtype=torch.double)
    torch.testing.assert_close(models[0](x), models[1](x))
    assert all(block.norm.weight is None for block in models[0].blocks)
    assert (
        len([n for n, _ in models[0].named_parameters() if n == "layerrope.weight"])
        == 1
    )


@pytest.mark.parametrize("arch", ["residual_mlp", "hc"])
def test_packed_models_do_not_share_layerrope_updates(arch):
    recipe = Recipe(arch=arch, width=16, blocks=3, layerrope=True)
    model = build(recipe, 2)
    init_model(model, ["seed-1", "seed-2"], "init")
    before = state_digest(model, 1)
    opt = torch.optim.Adam(model.parameters(), lr=0.001)
    model(torch.randn(2, 12, 784))[0].square().mean().backward()
    assert all(
        p.grad is not None and torch.isfinite(p.grad).all()
        for p in model.layerrope.parameters()
    )
    assert all(p.grad[1].count_nonzero() == 0 for p in model.parameters())
    opt.step()
    assert state_digest(model, 1) == before


@pytest.mark.parametrize("arch", ["residual_mlp", "hc", "mhc"])
@pytest.mark.parametrize("output_gain", [False, True])
def test_layerrope_active_null_is_stationary_and_signal_learns(arch, output_gain):
    model = build(
        Recipe(
            arch=arch,
            width=16,
            blocks=4,
            layerrope=True,
            layerrope_output_gain=output_gain,
        ),
        2,
    )
    init_model(model, ["seed-1", "seed-2"], "init")
    initial = copy.deepcopy(model)
    opt = torch.optim.Adam(model.parameters(), lr=0.001)
    before = state_digest(model)
    x = torch.randn(2, 24, 784)
    for _ in range(3):
        with torch.no_grad():
            targets = initial(x)[..., 10:]
        loss, _, _ = student_loss(
            model(x)[..., 10:], targets, "per_example_v1", 2, "exact_cancel_v1"
        )
        opt.zero_grad(set_to_none=True)
        loss.backward()
        assert all(
            p.grad.count_nonzero() == 0
            for p in model.parameters()
            if p.grad is not None
        )
        opt.step()
        assert state_digest(model) == before
    with torch.no_grad():
        targets = initial(x)[..., 10:] + torch.randn(2, 24, 3) * 0.2
    loss, _, _ = student_loss(
        model(x)[..., 10:], targets, "per_example_v1", 2, "exact_cancel_v1"
    )
    opt.zero_grad(set_to_none=True)
    loss.backward()
    assert all(
        p.grad.isfinite().all() and p.grad.count_nonzero() > 0
        for p in model.layerrope.parameters()
    )
    opt.step()
    assert state_digest(model) != before
    assert rows_unchanged(model, initial, list(range(10))).all()


def test_recipe_records_layerrope_and_rejects_invalid_combinations():
    recipe, sources = resolve(
        overrides=["arch=hc", "layerrope=y", "layerrope_base=100"]
    )
    assert recipe.to_json()["layerrope"] is True
    assert sources["layerrope"] == "override"
    for changes in (
        {"arch": "paper_mlp"},
        {"width": 15},
        {"block_norm": False},
        {"layerrope_base": -1},
        {"layerrope_rot_slope": float("nan")},
    ):
        with pytest.raises(ValueError):
            validate(Recipe(**{"arch": "residual_mlp", "layerrope": True, **changes}))


@pytest.mark.parametrize("arch", ["residual_mlp", "hc", "mhc"])
@pytest.mark.parametrize("beta", [-1.0, -0.5, 0.0])
def test_input_and_output_gains_follow_the_author_equation(arch, beta):
    recipe = Recipe(
        arch=arch,
        width=16,
        blocks=4,
        layerrope=True,
        layerrope_output_gain=True,
        layerrope_mag_slope=beta,
        layerrope_rot_slope=beta,
    )
    model = build(recipe, 2).double()
    init_model(model, ["seed-1", "seed-2"], "init")
    for schedule in (model.layerrope, model.output_layerrope):
        gains = schedule()
        layers = torch.arange(1, 5, dtype=torch.double)
        angle = layers.pow(beta)[:, None] * 100 ** (
            -torch.arange(8, dtype=torch.double) / 8
        )
        expected = torch.view_as_real(
            torch.complex(torch.ones(8), torch.ones(8)).to(torch.complex128)[None]
            * torch.polar(layers.pow(beta)[:, None].expand_as(angle), angle)
        ).flatten(-2)
        torch.testing.assert_close(gains[0], expected)
    x = torch.randn(2, 7, 784, dtype=torch.double)
    out = model(x)
    out.square().mean().backward()
    assert all(
        p.grad is not None and p.grad.isfinite().all()
        for p in model.output_layerrope.parameters()
    )
    with torch.no_grad():
        model.output_layerrope.weight.zero_()
    # Zeroing the write gate removes all block updates, without an output norm.
    expected = model.head(model.final_norm(model.stem(x)))
    torch.testing.assert_close(model(x), expected)


@pytest.mark.parametrize(
    "arch,enabled",
    [("residual_mlp", False), ("hc", False), ("residual_mlp", True), ("hc", True)],
)
def test_study_qualification_executes_all_active_controls(monkeypatch, arch, enabled):
    from research.subliminal_learning import layerrope_study

    torch.manual_seed(29)
    x, y = torch.randn(16, 784), torch.randint(0, 10, (16,))
    monkeypatch.setattr(
        layerrope_study, "load_data", lambda recipe, device: (x, y, x, y)
    )
    recipe = Recipe(
        arch=arch,
        layerrope=enabled,
        width=16,
        blocks=2,
        seeds=[1, 2],
        pack_size=2,
        batch_size=4,
        device="cpu",
        protocol="per_example_v1",
        student_loss_mode="exact_cancel_v1",
    )
    result = layerrope_study.qualify(recipe)
    assert result["all_pass"]
    assert len(result["conditions"]) == 3
    assert all(len(points) == 3 for points in result["conditions"].values())
