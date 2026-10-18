# SPDX-License-Identifier: MIT
"""Check finite normalization fidelity, fixed samples and diagnostic barriers."""

import pytest
import torch

from research.layerrope_beta.model import Model
from research.layerrope_beta.time_budget.routing_audit import (
    normalize,
    errors,
    distribution,
    capture,
    sample,
    gradient_check,
    solver,
    load_model,
    validation_parity,
)
from research.layerrope_beta.time_budget.state import json_hash
from research.mo_variants.stage_c_common import sha
from research.tinystories_depth.model import mhc
from research.tinystories_depth.train import state_hash


def test_finite_recipes_preserve_upstream_axes_order_and_epsilon():
    torch.manual_seed(10)
    for dtype in (torch.float32, torch.float64):
        logits = (torch.randn(9, 4, 4) * 5).to(dtype)
        for repeats in (20, 40, 80, 160, 320):
            assert torch.equal(
                normalize(logits, repeats),
                mhc.sinkhorn_normalize_ref(logits, repeat=repeats, eps=1e-6),
            )
        assert errors(normalize(logits, 20))[:, 1].max() < 2e-6
        assert errors(normalize(logits, 20))[:, 0].max() > 1e-4
        value, previous = normalize(logits, 20, with_previous=True)
        assert torch.equal(value, normalize(logits, 20))
        assert torch.equal(
            previous, mhc.sinkhorn_normalize_ref(logits, repeat=19, eps=1e-6)
        )
    with pytest.raises(ValueError, match="Undeclared"):
        normalize(logits, 640)


def test_distribution_reports_all_matrices_and_never_drops_a_miss():
    values = torch.tensor([0.0, 0.00001, 0.001, 0.1])
    result = distribution(values)
    assert result["count"] == 4 and result["above_bound"] == 2
    assert result["fraction_above_bound"] == 0.5
    assert result["maximum"] == values.max().item()
    with pytest.raises(FloatingPointError):
        distribution(torch.tensor([float("nan")]))


def test_capture_preserves_outputs_and_samples_have_fixed_token_ids():
    torch.set_num_threads(2)
    model = Model("ML", 2, 7, vocab=97)
    ids = torch.arange(16).reshape(2, 8)
    sites, parity = capture(model, ids)
    saved, gradients = sample(sites, 2)
    assert parity["logits_equal"] and parity["parameters_unchanged"]
    assert parity["original_routing"]["row_distribution"]["count"] == 64
    assert len(saved) == 4 and gradients.shape == (8, 4, 4)
    for original, chosen in zip(sites, saved):
        indices = chosen["indices"]
        assert torch.equal(
            chosen["logits"], original["logits"].reshape(-1, 4, 4)[indices]
        )
        assert torch.equal(chosen["token_ids"]["token_position"], indices % 8)
        assert torch.equal(chosen["token_ids"]["window_index"], indices // 8)
    assert all(not m._forward_hooks for m in model.routes)


def test_solver_context_restores_original_even_after_failure():
    original = mhc.sinkhorn_normalize_ref
    logits = torch.randn(2, 4, 4)
    with pytest.raises(RuntimeError, match="interruption"):
        with solver(80):
            assert torch.equal(
                mhc.sinkhorn_normalize_ref(logits, repeat=20), normalize(logits, 80)
            )
            raise RuntimeError("interruption")
    assert mhc.sinkhorn_normalize_ref is original


def test_same_recipe_gradient_and_finite_difference_checks():
    torch.manual_seed(10)
    logits = torch.randn(8, 4, 4) * 3
    before = logits.clone()
    for repeats in (20, 80, 320):
        result = gradient_check(logits, repeats)
        assert result["finite"] and result["passed"]
        assert len(result["finite_differences"]) == 2
    assert torch.equal(before, logits)


def test_checkpoint_binding_rejects_changed_identity_and_payload(tmp_path):
    model = Model("M", 2, 7, vocab=97)
    identity = dict(job=dict(arm="M", seed=7))
    path = tmp_path / "checkpoint.pt"
    torch.save(
        dict(identity=identity, model=model.state_dict(), step=1, training_seconds=2.0),
        path,
    )
    receipt = dict(
        path=str(path),
        sha256=sha(path),
        model_sha256=state_hash(model.state_dict()),
        row="M",
        seed=7,
        depth=2,
        vocab=97,
        step=1,
        training_seconds=2.0,
        identity_sha256=json_hash(identity),
    )
    restored = load_model(receipt, "cpu")
    assert state_hash(restored.state_dict()) == receipt["model_sha256"]
    with pytest.raises(ValueError, match="recipe"):
        load_model(dict(receipt, seed=8), "cpu")
    with pytest.raises(ValueError, match="payload"):
        load_model(dict(receipt, model_sha256="0" * 64), "cpu")


def test_original_validation_replay_requires_same_windows_and_recorded_loss():
    reference = dict(window_ce=[2.0, 3.0], ce=2.5)
    assert validation_parity(reference, reference)["passed"]
    with pytest.raises(ValueError, match="coverage"):
        validation_parity(dict(window_ce=[2.0], ce=2.0), reference)
    with pytest.raises(ValueError, match="replay differs"):
        validation_parity(dict(window_ce=[2.0, 3.001], ce=2.5005), reference)
