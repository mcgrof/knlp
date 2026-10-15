"""Check real cache operators, seed pairing and checkpoint integrity gates."""

import json

import pytest
import torch

from research.reproduction.kv_cache import load, paired, run
from research.tinystories_depth.kv_cache import CachePolicy, install
from research.tinystories_depth.kv_cache_checks import qualify
from research.tinystories_depth.model import Model
from research.tinystories_depth.train import state_hash


def test_quantizer_cache_and_complete_model_qualification():
    torch.set_num_threads(2)
    result = qualify("cpu")
    assert result["status"] == "pass"
    assert "whole-model-caller-parity-R" in result["tests"]


def test_attention_adapter_restores_the_model_after_an_error():
    model = Model("R", 2, 89104, vocab=97)
    original = model.lm.model.layers[0].self_attn.forward
    with pytest.raises(RuntimeError, match="interrupted"):
        with install(model, CachePolicy(4, 4)):
            raise RuntimeError("interrupted")
    assert model.lm.model.layers[0].self_attn.forward == original
    assert not hasattr(model.lm.model.layers[0].self_attn, "_kv_policy")


def test_real_checkpoint_hash_and_metadata_are_required(tmp_path):
    model = Model("R", 2, 89104, vocab=97)
    checkpoint = tmp_path / "model.pt"
    torch.save(
        dict(
            step=2000,
            model=model.state_dict(),
            metadata=dict(job=dict(arm="R", depth=2, seed=89104)),
        ),
        checkpoint,
    )
    job = dict(
        id="saved",
        arm="R",
        depth=2,
        seed=89104,
        step=2000,
        checkpoint=dict(
            path=str(checkpoint), parameter_sha256=state_hash(model.state_dict())
        ),
    )
    loaded = load(job, "cpu", vocab=97)
    assert state_hash(loaded.state_dict()) == job["checkpoint"]["parameter_sha256"]
    job["seed"] += 1
    with pytest.raises(ValueError, match="seed mismatch"):
        load(job, "cpu", vocab=97)


def test_causal_token_changes_do_not_affect_earlier_logits():
    torch.set_num_threads(2)
    model = Model("R", 2, 89104, vocab=97).eval()
    ids = torch.tensor([[2, 8, 9, 5, 10, 7, 6, 12]])
    changed = ids.clone()
    changed[:, 5:] = 44
    with torch.no_grad(), install(model, CachePolicy(2, 2, "affine")), torch.autocast(
        device_type="cpu", dtype=torch.bfloat16
    ):
        a, b = model(ids), model(changed)
    assert torch.equal(a[:, :5], b[:, :5])


def test_full_cache_smoke_checks_all_cases_without_a_verdict(tmp_path):
    root = tmp_path / "cache"
    run(root, None, "depth", device="cpu", max_seconds=120, smoke=True)
    report = json.loads((root / "REPORT.json").read_text())
    assert (
        report["status"] == "complete" and not report["scientific_confirmation_allowed"]
    )
    assert set(report["groups"]) == {"R-D2", "M-D2", "M-O-D2"}
    records = [json.loads(p.read_text()) for p in (root / "cells").glob("*.json")]
    assert all(len(r["cases"]) == 13 and r["weights_unchanged"] for r in records)
    assert all(c["predicted_tokens"] == 32 for r in records for c in r["cases"])


def test_seed_statistics_do_not_treat_windows_as_replicates():
    assert paired([-1] * 3)["one_sided_sign_p"] == 0.125
    assert paired([-1] * 8)["one_sided_sign_p"] == 1 / 256


def test_missing_cohort_fails_and_preserves_an_incomplete_record(tmp_path):
    root = tmp_path / "missing"
    with pytest.raises(FileNotFoundError):
        run(root, tmp_path / "no-saved-run", "depth", device="cpu")
    status = json.loads((root / "STATUS.json").read_text())
    assert (
        status["status"] == "incomplete"
        and not status["scientific_confirmation_allowed"]
    )
