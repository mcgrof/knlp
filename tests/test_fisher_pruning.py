"""Tests for the factored-Fisher pruning lane (fim/fisher_pruning/).

Covers exactness of the K-FAC factor accumulator against per-token
autograd, the OBS-score diagonal reduction, mask construction, Adam
state fetch, and an end-to-end calibrate + prune-eval smoke on a tiny
GPT-2 with synthetic token bins.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from fim.fisher_pruning import phase_a  # noqa: E402
from fim.fisher_pruning.kfac_capture import (  # noqa: E402
    FactorAccumulator,
    damped_inverse_diag,
    default_target_names,
    global_masks,
    per_layer_mask,
)
from fim.reciprocal_attention.gpt2_spectral_ra import (  # noqa: E402
    build_model,
    make_optimizer,
    save_checkpoint,
)

TINY_MODEL = {
    "n_layer": 2,
    "n_head": 2,
    "n_embd": 16,
    "block_size": 8,
    "vocab_size": 64,
    "dropout": 0.0,
    "bias": True,
}

TINY_TRAIN = {
    "batch_size": 2,
    "max_steps": 3,
    "lr": 1e-3,
    "min_lr": 1e-4,
    "warmup_steps": 1,
    "weight_decay": 0.1,
    "betas": [0.9, 0.95],
    "clip": 1.0,
}


class TwoLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(5, 4)
        self.fc2 = nn.Linear(4, 3)

    def forward(self, x):
        return self.fc2(torch.tanh(self.fc1(x)))


def per_token_loss(out):
    # sum over features, mean over tokens — same token-mean structure
    # as the LM cross-entropy the accumulator is calibrated for.
    return out.pow(2).sum(-1).mean()


def test_factor_accumulator_matches_per_token_autograd():
    torch.manual_seed(0)
    model = TwoLayer()
    x = torch.randn(2, 3, 5)  # B=2, T=3 -> 6 tokens
    acc = FactorAccumulator(model, ["fc1", "fc2"])
    acc.attach()
    out = model(x)
    loss = per_token_loss(out)
    model.zero_grad()
    loss.backward()
    acc.count_batch(6)
    acc.detach()
    factors = acc.finalize()

    # Reference: loop over tokens with autograd on the per-token loss.
    xs = x.reshape(-1, 5)
    n = xs.shape[0]
    d_ref = {"fc1": 0, "fc2": 0}
    g_ref = {"fc1": 0, "fc2": 0}
    a_ref = {"fc1": 0, "fc2": 0}
    for t in range(n):
        xt = xs[t : t + 1].clone()
        h_pre = model.fc1(xt)
        h = torch.tanh(h_pre)
        yt = model.fc2(h)
        lt = yt.pow(2).sum()
        grads = torch.autograd.grad(
            lt, [model.fc1.weight, model.fc2.weight, h_pre, yt], retain_graph=False
        )
        d_ref["fc1"] = d_ref["fc1"] + grads[0] ** 2
        d_ref["fc2"] = d_ref["fc2"] + grads[1] ** 2
        g1 = grads[2].reshape(-1)
        g2 = grads[3].reshape(-1)
        g_ref["fc1"] = g_ref["fc1"] + torch.outer(g1, g1)
        g_ref["fc2"] = g_ref["fc2"] + torch.outer(g2, g2)
        a_ref["fc1"] = a_ref["fc1"] + torch.outer(xt[0], xt[0])
        ht = h.reshape(-1)
        a_ref["fc2"] = a_ref["fc2"] + torch.outer(ht, ht)
    for name in ("fc1", "fc2"):
        torch.testing.assert_close(
            factors[name]["D"], d_ref[name] / n, rtol=1e-4, atol=1e-6
        )
        torch.testing.assert_close(
            factors[name]["G"], g_ref[name] / n, rtol=1e-4, atol=1e-6
        )
        torch.testing.assert_close(
            factors[name]["A"], a_ref[name] / n, rtol=1e-4, atol=1e-6
        )


def test_obs_score_reduces_to_diagonal_product():
    # For diagonal factors and no damping, the OBS denominator
    # (G^-1)_ii (A^-1)_jj is 1/(G_ii A_jj): the score must equal
    # w^2 * G_ii * A_jj.
    g_diag = torch.tensor([1.0, 4.0, 0.25])
    a_diag = torch.tensor([2.0, 0.5])
    inv_g = damped_inverse_diag(torch.diag(g_diag), kappa=0.0)
    inv_a = damped_inverse_diag(torch.diag(a_diag), kappa=0.0)
    torch.testing.assert_close(inv_g.float(), 1.0 / g_diag, rtol=1e-6, atol=1e-9)
    w = torch.randn(3, 2)
    score = w.pow(2) / torch.outer(inv_g.float(), inv_a.float())
    expected = w.pow(2) * torch.outer(g_diag, a_diag)
    torch.testing.assert_close(score, expected, rtol=1e-5, atol=1e-8)


def test_per_layer_mask_counts_and_ranking():
    score = torch.tensor([[4.0, 1.0], [3.0, 2.0]])
    mask = per_layer_mask(score, 0.5)
    assert mask.sum().item() == 2
    assert mask[0, 0] and mask[1, 0]  # two largest kept
    assert per_layer_mask(score, 0.0).all()


def test_global_masks_share_one_threshold():
    scores = {
        "a": torch.tensor([[10.0, 9.0]]),
        "b": torch.tensor([[1.0, 2.0]]),
    }
    masks = global_masks(scores, 0.5)
    assert masks["a"].all()
    assert not masks["b"].any()


@pytest.fixture
def tiny_ckpt(tmp_path):
    torch.manual_seed(0)
    model = build_model(TINY_MODEL, selection={})
    optimizer = make_optimizer(model, TINY_TRAIN)
    gen = torch.Generator().manual_seed(0)
    for _ in range(2):
        x = torch.randint(0, 64, (2, 8), generator=gen)
        y = torch.randint(0, 64, (2, 8), generator=gen)
        _, loss = model(x, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    path = tmp_path / "ckpt.pt"
    cfg = {"model": TINY_MODEL, "train": TINY_TRAIN}
    save_checkpoint(path, model, optimizer, 1, 2, cfg)
    return path, model, optimizer


def test_bitter7_state_fetch_matches_optimizer(tiny_ckpt):
    path, model, optimizer = tiny_ckpt
    ckpt, rebuilt = phase_a._load_ckpt({"ckpt": str(path)})
    fetched = phase_a._bitter7_exp_avg_sq(ckpt, rebuilt)
    names = default_target_names(TINY_MODEL["n_layer"])
    assert sorted(fetched) == sorted(names)
    params = dict(model.named_parameters())
    for name in names:
        ref = optimizer.state[params[name + ".weight"]]["exp_avg_sq"]
        torch.testing.assert_close(fetched[name], ref.float(), rtol=0, atol=0)


def _write_bin(path, n_tokens, vocab, seed):
    rng = np.random.default_rng(seed)
    arr = rng.integers(0, vocab, size=n_tokens, dtype=np.uint16)
    arr.tofile(path)


def test_phase_a_end_to_end_smoke(tmp_path, tiny_ckpt, capsys):
    path, _, _ = tiny_ckpt
    train_bin = tmp_path / "train.bin"
    val_bin = tmp_path / "val.bin"
    _write_bin(train_bin, 4096, TINY_MODEL["vocab_size"], 1)
    _write_bin(val_bin, 4096, TINY_MODEL["vocab_size"], 2)
    cfg = {
        "ckpt": str(path),
        "data": {"train_bin": str(train_bin), "val_bin": str(val_bin)},
        "calibration": {"seed": 7, "n_batches": 3, "batch_size": 2},
        "eval": {"seeds": [11, 12], "n_batches": 2, "batch_size": 2},
        "scopes": {"per_layer": [0.5], "global": [0.5]},
        "obs_kappas": [0.01],
        "random_arm_seed": 5,
        "out_dir": str(tmp_path / "lane"),
    }
    out_dir = Path(cfg["out_dir"])
    phase_a.cmd_calibrate(cfg, "cpu", out_dir, force=False)
    calib = torch.load(
        out_dir / "calib" / "calib.pt", map_location="cpu", weights_only=False
    )
    assert calib["n_tokens"] == 3 * 2 * TINY_MODEL["block_size"]
    with pytest.raises(SystemExit):
        phase_a.cmd_calibrate(cfg, "cpu", out_dir, force=False)
    phase_a.cmd_prune_eval(cfg, "cpu", out_dir)
    events = [
        json.loads(line)
        for line in (out_dir / "results" / "phaseA.jsonl").read_text().splitlines()
    ]
    evals = [e for e in events if e.get("event") == "eval"]
    arms = {e["arm"] for e in evals}
    assert "dense" in arms and "bitter7" in arms and "kron_diag" in arms
    assert "kfac_obs_k0.01" in arms
    # dense (2 seeds) + 6 arms (5 fixed + one obs kappa) * 2 scopes * 2 seeds
    assert len(evals) == 2 + 6 * 2 * 2
    pruned = [e for e in evals if e["arm"] != "dense" and e["scope"] == "per_layer"]
    for e in pruned:
        assert abs(e["actual_sparsity"] - 0.5) < 0.01
    assert any(e.get("event") == "mask_overlap_jaccard_50pct" for e in events)
    assert any(e.get("event") == "done" for e in events)


def test_random_arm_prunes_worse_than_magnitude_smoke(tiny_ckpt):
    # Sanity on scores only: magnitude keeps the largest weights.
    path, _, _ = tiny_ckpt
    ckpt, model = phase_a._load_ckpt({"ckpt": str(path)})
    names = default_target_names(TINY_MODEL["n_layer"])
    params = dict(model.named_parameters())
    w = params[names[0] + ".weight"].detach()
    mask = per_layer_mask(w.abs(), 0.5)
    kept_min = w.abs()[mask].min()
    pruned_max = w.abs()[~mask].max()
    assert kept_min >= pruned_max
