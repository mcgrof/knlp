# SPDX-License-Identifier: MIT
"""Equation, intervention, monitoring, pairing and selection controls."""

import copy
import json
from types import SimpleNamespace

import pytest
import torch
from torch.autograd import gradcheck

from research.subliminal_learning.connections import sinkhorn_knopp_log
from research.subliminal_learning.models import build, init_model
from research.subliminal_learning.recipe import Recipe, validate
from research.supervised_depth.model import (
    SupervisedModel,
    code_vectors,
    coded_write,
    phase,
)
from research.supervised_depth.numerics import ConvergedSinkhorn
from research.supervised_depth.study import (
    crossing,
    enumerate_cells,
    selection,
    split_indices,
    supervised_loss,
)


@pytest.fixture(autouse=True)
def threads():
    torch.set_num_threads(1)


@pytest.mark.parametrize("arm", ("M", "ML", "ML-O", "ML-S", "ML-C", "M-O"))
def test_forward_gradients_common_weights_and_counts(arm):
    model = SupervisedModel(arm, 4, ["seed-a", "seed-b"], width=16).double()
    ref = SupervisedModel("R", 4, ["seed-a", "seed-b"], width=16).double()
    assert model.identities()["common_weights"] == ref.identities()["common_weights"]
    logits = model(torch.randn(9, 784, dtype=torch.double))
    assert logits.shape == (2, 9, 10)
    loss, _ = supervised_loss(logits[0:1], torch.randint(0, 10, (9,)))
    loss.backward()
    assert all(
        p.grad is None or p.grad[1].count_nonzero() == 0 for p in model.parameters()
    )
    assert all(p.grad is None or p.grad.isfinite().all() for p in model.parameters())
    assert model.invariants[:2].max() < 1e-4
    if arm in ("ML-O", "ML-S", "ML-C"):
        assert model.angular_schedule.alpha_rot is model.layerrope.alpha_rot
        assert model.channel_schedule is None
        parent = SupervisedModel("ML", 4, ["seed-a", "seed-b"], width=16)
        assert sum(p.numel() for p in model.parameters()) == sum(
            p.numel() for p in parent.parameters()
        )
    if arm == "M-O":
        parent = SupervisedModel("M", 4, ["seed-a", "seed-b"], width=16)
        assert (
            sum(p.numel() for p in model.parameters())
            == sum(p.numel() for p in parent.parameters()) + 4
        )


def test_mhc_layerrope_validator_and_explicit_block_reference():
    recipe = Recipe(
        arch="mhc",
        layerrope=True,
        width=16,
        blocks=4,
        mhc_init_logit=1.0,
        sinkhorn_log=True,
    )
    validate(recipe)
    model = build(recipe, 1).double()
    init_model(model, ["seed-a"], "supervised_init_common")
    x = torch.randn(13, 784, dtype=torch.double)
    h = model.stem(x).unsqueeze(2).expand(1, 13, 4, 16)
    gains = model.layerrope()
    for l, conn in enumerate(model.conns):
        pre, post, matrix = conn.maps(h)
        inp = (pre[..., None] * h).sum(2)
        block = model.blocks[l]
        norm = (
            torch.nn.functional.layer_norm(inp, (16,), eps=1e-5) * gains[:, l, None]
            + block.norm.bias[:, None]
        )
        out = block.fc2(torch.relu(block.fc1(norm)))
        h = matrix @ h + post[..., None] * out[:, :, None]
    expected = model.head(model.final_norm(h.mean(2)))
    torch.testing.assert_close(model(x), expected, rtol=1e-12, atol=1e-12)


def test_cross_entropy_pack_gradient_scale():
    logits = torch.randn(2, 7, 10, dtype=torch.double, requires_grad=True)
    y = torch.randint(0, 10, (2, 7))
    packed = supervised_loss(logits, y)[0]
    grads = torch.autograd.grad(packed, logits)[0]
    for m in range(2):
        single = logits[m : m + 1].detach().requires_grad_()
        grad = torch.autograd.grad(supervised_loss(single, y[m : m + 1])[0], single)[0]
        torch.testing.assert_close(grads[m], grad[0], rtol=0, atol=0)
    with pytest.raises(ValueError):
        supervised_loss(torch.randn(1, 7, 13), y[:1])


def test_code_properties_derivatives_ties_and_zero_amplitude():
    angles = torch.tensor([[0.2, 0.7, 1.0]], dtype=torch.double, requires_grad=True)
    code = code_vectors(angles)
    torch.testing.assert_close(
        code.sum(-1), torch.zeros_like(angles), atol=1e-15, rtol=0
    )
    torch.testing.assert_close(code.norm(dim=-1), torch.full_like(angles, 2**0.5))
    beta = torch.tensor(
        [[[0.22, 0.65, 1.12, 1.71]]], dtype=torch.double, requires_grad=True
    )
    assert gradcheck(
        lambda b, a: coded_write(b, code_vectors(a))[0],
        (beta, angles),
        eps=1e-6,
        atol=1e-6,
    )
    coded, _ = coded_write(beta, code)
    assert coded.min() >= 0 and coded.max() <= 2
    torch.testing.assert_close(
        coded.sum(-1), beta.sum(-1).expand_as(angles), atol=1e-15, rtol=0
    )
    zero, _ = coded_write(beta, code, rho=0.0)
    torch.testing.assert_close(zero, beta.expand_as(code), rtol=0, atol=0)
    tied = torch.ones(1, 1, 4, dtype=torch.double, requires_grad=True)
    y, strength = coded_write(
        tied, code_vectors(torch.tensor([[0.7]], dtype=torch.double))
    )
    assert (
        torch.autograd.grad(strength.sum(), tied, retain_graph=True)[0].count_nonzero()
        == 0
    )
    assert torch.isfinite(torch.autograd.grad(y.square().sum(), tied)[0]).all()
    saturated, _ = coded_write(
        torch.tensor([[[0.0, 1.0, 1.0, 1.0]]]), code_vectors(torch.ones(1, 1))
    )
    torch.testing.assert_close(saturated, torch.tensor([[[0.0, 1.0, 1.0, 1.0]]]))


@pytest.mark.parametrize("arm", ("ML-O", "ML-S", "ML-C"))
def test_norm_assignment_and_shared_schedule_both_gradient_paths(arm):
    model = SupervisedModel(arm, 4, ["seed-a"], width=16).double()
    same = SupervisedModel(arm, 4, ["seed-a"], width=16).double()
    assert torch.equal(model.code_assignment, same.code_assignment)
    if arm == "ML-S":
        assert sorted(model.code_assignment[0].tolist()) == list(range(4))
        assert not torch.equal(model.code_assignment, torch.arange(4)[None])
    if arm == "ML-C":
        assert model.code_assignment.unique().tolist() == [1]
    torch.testing.assert_close(
        model.layerrope(),
        SupervisedModel("ML", 4, ["seed-a"], width=16).double().layerrope(),
    )
    p = model.layerrope.alpha_rot
    gain_grad = torch.autograd.grad(
        model.layerrope().square().mul(torch.arange(16)).sum(), p
    )[0]
    code_grad = torch.autograd.grad((model.channel_codes() * torch.arange(4)).sum(), p)[
        0
    ]
    assert gain_grad.abs().sum() > 0 and code_grad.abs().sum() > 0
    angles = phase(model.layerrope, 4)
    torch.testing.assert_close(
        model.channel_codes(), code_vectors(angles.gather(1, model.code_assignment))
    )


def test_zero_code_and_monitoring_optimizer_parity():
    parent = SupervisedModel("ML", 4, ["seed-a"], width=16).double()
    child = SupervisedModel("ML-O", 4, ["seed-a"], width=16, rho=0).double()
    child.monitor = False
    x = torch.randn(11, 784, dtype=torch.double)
    y = torch.randint(0, 10, (11,))
    opts = [torch.optim.Adam(m.parameters(), lr=3e-4) for m in (parent, child)]
    for _ in range(3):
        torch.testing.assert_close(parent(x), child(x), rtol=0, atol=0)
        for opt, m in zip(opts, (parent, child)):
            opt.zero_grad()
            supervised_loss(m(x), y)[0].backward()
            opt.step()
    for a, b in zip(parent.parameters(), child.parameters()):
        torch.testing.assert_close(a, b, rtol=0, atol=0)


def test_converged_implicit_vjp_matches_long_unrolled_reference():
    torch.manual_seed(24)
    x = torch.randn(2, 4, 4, dtype=torch.double, requires_grad=True)
    cot = torch.randn_like(x)
    p = ConvergedSinkhorn.apply(x)
    expected = sinkhorn_knopp_log(x, 500)
    torch.testing.assert_close(p, expected, rtol=1e-11, atol=1e-11)
    a = torch.autograd.grad((p * cot).sum(), x, retain_graph=True)[0]
    b = torch.autograd.grad((expected * cot).sum(), x)[0]
    torch.testing.assert_close(a, b, rtol=1e-9, atol=1e-10)
    assert gradcheck(ConvergedSinkhorn.apply, (x,), eps=1e-6, atol=1e-6)


def test_split_is_exact_reproducible_and_disjoint():
    y = torch.arange(60000) % 10
    fit, val = split_indices(y)
    assert len(fit) == 55000 and len(val) == 5000
    assert len(set(fit.tolist()) & set(val.tolist())) == 0
    assert torch.cat((fit, val)).sort().values.tolist() == list(range(60000))
    assert torch.bincount(y[val]).tolist() == [500] * 10
    again = split_indices(y)
    assert all(torch.equal(a, b) for a, b in zip((fit, val), again))


def test_target_confirmation_lr_selection_and_matrix_counts():
    def row(step, ce=0.09, accuracy=0.98):
        return {"step": step, "ce": ce, "accuracy": accuracy}

    assert (
        crossing([row(0, 0.3), row(10), row(20, 0.2), row(30), row(40)])["step"] == 40
    )
    assert crossing([row(0, 0.3), row(10)]) is None
    records = []
    for lr in (1e-4, 3e-4, 1e-3, 3e-3):
        for seed in (75001, 75002):
            records.append(
                {
                    "lr": lr,
                    "seed": seed,
                    "status": "complete",
                    "curve": [row(0, 0.5), row(10, 0.11), row(20), row(30)],
                }
            )
    assert selection(records)["lr"] == 1e-4
    records[-1]["status"] = "invalid"
    assert selection(records)["lr"] == 1e-4
    assert len(enumerate_cells()) == 20
    assert sum(8 for c in enumerate_cells()) == 160
    assert sum(10 if c["depth"] == 64 else 5 for c in enumerate_cells()) == 140


def test_failed_qualification_blocks_probe_before_data_or_training(
    tmp_path, monkeypatch
):
    from research.supervised_depth import study

    (tmp_path / "QUALIFICATION.json").write_text(
        json.dumps({"status": "unqualified", "selected_iters": None})
    )
    monkeypatch.setattr(study, "source", lambda: {"commit": "test"})
    monkeypatch.setattr(
        study, "dataset", lambda *_: pytest.fail("failed queue must not load GPU data")
    )
    result = study.probe(SimpleNamespace(root=tmp_path), lambda: 28800)
    assert result["status"] == "not_run_failed_qualification"
    assert not result["eligible_core"] and not result["eligible_channels"]
    assert result["probes"] == []


def test_empty_failed_queue_cannot_nominate_or_report_speed(tmp_path):
    from research.supervised_depth import study

    (tmp_path / "QUALIFICATION.json").write_text(json.dumps({"status": "unqualified"}))
    result = study.analyze(SimpleNamespace(root=tmp_path))
    assert result["completed_final"] == 0
    assert result["incomplete_reason"] == "mandatory numerical qualification failed"
    assert not any(result["nominees"].values())
    assert all(
        c["status"] == "missing_cells" and "step_ratio" not in c
        for c in result["contrasts"]
    )
