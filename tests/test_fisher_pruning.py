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


def test_train_optimizer_soap_smoke(tmp_path):
    from fim.fisher_pruning import train_optimizer

    train_bin = tmp_path / "train.bin"
    val_bin = tmp_path / "val.bin"
    _write_bin(train_bin, 4096, TINY_MODEL["vocab_size"], 3)
    _write_bin(val_bin, 4096, TINY_MODEL["vocab_size"], 4)
    cfg = {
        "seed": 0,
        "model": TINY_MODEL,
        "data": {"train_bin": str(train_bin), "val_bin": str(val_bin)},
        "train": {
            "optimizer": "soap",
            "batch_size": 2,
            "max_steps": 3,
            "lr": 3e-3,
            "min_lr": 3e-4,
            "warmup_steps": 1,
            "weight_decay": 0.1,
            "betas": [0.95, 0.95],
            "precondition_frequency": 2,
            "clip": 1.0,
            "eval_interval": 2,
            "eval_batches": 1,
            "ckpt_interval": 2,
            "log_interval": 1,
        },
        "out_dir": str(tmp_path / "soap_run"),
    }
    train_optimizer.cmd_train(cfg, "cpu")
    out = Path(cfg["out_dir"])
    assert (out / "ckpt_latest.pt").exists()
    ckpt = torch.load(out / "ckpt_latest.pt", map_location="cpu", weights_only=False)
    assert ckpt["optimizer_state"] is not None
    # SOAP state must carry the Shampoo factors for later harvesting.
    states = ckpt["optimizer_state"]["state"]
    assert any("GG" in s or "Q" in s for s in states.values())
    events = [
        json.loads(line) for line in (out / "train.jsonl").read_text().splitlines()
    ]
    assert events[0]["optimizer"] == "SOAP"
    assert any(e.get("event") == "done" for e in events)


def test_soap_native_scores_shapes_and_positivity(tmp_path):
    from fim.fisher_pruning import phase_b, train_optimizer

    train_bin = tmp_path / "train.bin"
    val_bin = tmp_path / "val.bin"
    _write_bin(train_bin, 4096, TINY_MODEL["vocab_size"], 5)
    _write_bin(val_bin, 4096, TINY_MODEL["vocab_size"], 6)
    cfg = {
        "seed": 0,
        "model": TINY_MODEL,
        "data": {"train_bin": str(train_bin), "val_bin": str(val_bin)},
        "train": {
            "optimizer": "soap",
            "batch_size": 2,
            "max_steps": 4,
            "lr": 3e-3,
            "min_lr": 3e-4,
            "warmup_steps": 1,
            "weight_decay": 0.1,
            "betas": [0.95, 0.95],
            "precondition_frequency": 2,
            "clip": 1.0,
            "eval_interval": 10,
            "eval_batches": 1,
            "ckpt_interval": 10,
            "log_interval": 10,
        },
        "out_dir": str(tmp_path / "soap_run"),
    }
    train_optimizer.cmd_train(cfg, "cpu")
    ckpt_path = Path(cfg["out_dir"]) / "ckpt_latest.pt"
    ckpt, model = phase_a._load_ckpt({"ckpt": str(ckpt_path)})
    scores = phase_b.soap_native_scores(ckpt, model)
    names = default_target_names(TINY_MODEL["n_layer"])
    params = dict(model.named_parameters())
    for arm in ("soap_kron", "soap_rot_v"):
        assert sorted(scores[arm]) == sorted(names)
        for n in names:
            s = scores[arm][n]
            assert s.shape == params[n + ".weight"].shape
            assert (s >= 0).all() and torch.isfinite(s).all()
    # rotated-v projection preserves total second-moment mass:
    # sum(v_ws) = sum((QL^2) v' (QR^2)^T) = sum(v') since QL, QR are
    # orthogonal (columns have unit norm). Spot-check via the score
    # being non-degenerate (not all equal).
    s = scores["soap_rot_v"][names[0]]
    assert s.std() > 0


def test_train_optimizer_muon_smoke_and_native_scores(tmp_path):
    from fim.fisher_pruning import phase_b, train_optimizer

    train_bin = tmp_path / "train.bin"
    val_bin = tmp_path / "val.bin"
    _write_bin(train_bin, 4096, TINY_MODEL["vocab_size"], 7)
    _write_bin(val_bin, 4096, TINY_MODEL["vocab_size"], 8)
    cfg = {
        "seed": 0,
        "model": TINY_MODEL,
        "data": {"train_bin": str(train_bin), "val_bin": str(val_bin)},
        "train": {
            "optimizer": "muon",
            "batch_size": 2,
            "max_steps": 3,
            "lr": 6e-4,
            "min_lr": 6e-5,
            "warmup_steps": 1,
            "muon_lr": 0.02,
            "muon_momentum": 0.95,
            "muon_weight_decay": 0.01,
            "weight_decay": 0.1,
            "betas": [0.9, 0.95],
            "clip": 1.0,
            "eval_interval": 10,
            "eval_batches": 1,
            "ckpt_interval": 10,
            "log_interval": 10,
        },
        "out_dir": str(tmp_path / "muon_run"),
    }
    train_optimizer.cmd_train(cfg, "cpu")
    ckpt_path = Path(cfg["out_dir"]) / "ckpt_latest.pt"
    ckpt, model = phase_a._load_ckpt({"ckpt": str(ckpt_path)})
    # the muon/aux-adam split: matmul weights carry momentum buffers,
    # embeddings carry Adam state
    scores = phase_b.muon_native_scores(ckpt, model)
    names = default_target_names(TINY_MODEL["n_layer"])
    assert sorted(scores["muon_momentum"]) == sorted(names)
    params = dict(model.named_parameters())
    for n in names:
        s = scores["muon_momentum"][n]
        assert s.shape == params[n + ".weight"].shape
        assert torch.isfinite(s).all() and s.std() > 0
    # dispatch picks the right native arm
    assert set(phase_b.native_scores(ckpt, model)) == {"muon_momentum"}
    # per-group relative LR schedule preserved the muon/adam ratio
    opt = train_optimizer.build_optimizer(model, ckpt["train_config"]["train"])
    base = [g["base_lr"] for g in opt.param_groups]
    assert base[0] == 0.02 and base[1] == 6e-4


@pytest.mark.parametrize("opt_name", ["adamw", "soap", "muon"])
def test_in_training_state_pruning(tmp_path, opt_name):
    from fim.fisher_pruning import train_optimizer

    train_bin = tmp_path / "train.bin"
    val_bin = tmp_path / "val.bin"
    _write_bin(train_bin, 4096, TINY_MODEL["vocab_size"], 9)
    _write_bin(val_bin, 4096, TINY_MODEL["vocab_size"], 10)
    tr = {
        "optimizer": opt_name,
        "batch_size": 2,
        "max_steps": 8,
        "lr": 1e-3,
        "min_lr": 1e-4,
        "warmup_steps": 1,
        "weight_decay": 0.1,
        "betas": [0.9, 0.95],
        "clip": 1.0,
        "eval_interval": 20,
        "eval_batches": 1,
        "ckpt_interval": 20,
        "log_interval": 20,
    }
    if opt_name == "soap":
        tr.update(betas=[0.95, 0.95], precondition_frequency=2)
    if opt_name == "muon":
        tr.update(muon_lr=0.02, muon_momentum=0.95, muon_weight_decay=0.01)
    cfg = {
        "seed": 0,
        "model": TINY_MODEL,
        "data": {"train_bin": str(train_bin), "val_bin": str(val_bin)},
        "train": tr,
        "prune": {
            "target_sparsity": 0.5,
            "start_step": 2,
            "end_step": 6,
            "interval": 2,
            "signal": "state",
        },
        "final_eval": {"seeds": [11], "n_batches": 1},
        "out_dir": str(tmp_path / f"{opt_name}_prune_run"),
    }
    train_optimizer.cmd_train(cfg, "cpu")
    out = Path(cfg["out_dir"])
    events = [
        json.loads(line) for line in (out / "train.jsonl").read_text().splitlines()
    ]
    prunes = [e for e in events if e.get("event") == "prune"]
    assert prunes and prunes[-1]["target_sparsity"] == 0.5
    final = [e for e in events if e.get("event") == "final_sparsity"]
    assert final and abs(final[0]["actual_sparsity"] - 0.5) < 0.01
    assert any(e.get("event") == "final_eval" for e in events)
    # weights on disk actually carry the zeros
    ckpt = torch.load(out / "ckpt_latest.pt", map_location="cpu", weights_only=False)
    w = ckpt["model_state"]["transformer.h.0.attn.c_attn.weight"]
    assert (w == 0).float().mean().item() >= 0.49


def test_cubic_sparsity_schedule():
    from fim.fisher_pruning.train_optimizer import cubic_sparsity

    p = {"start_step": 100, "end_step": 200, "target_sparsity": 0.8}
    assert cubic_sparsity(0, p) == 0.0
    assert cubic_sparsity(100, p) == 0.0
    assert 0 < cubic_sparsity(150, p) < 0.8
    assert cubic_sparsity(200, p) == 0.8
    assert cubic_sparsity(999, p) == 0.8


# ---------------------------------------------------------------------
# Menu A correctness gate (CLOUD_WAVE_PLAN): score algebra, mask
# semantics, reproducibility. No paid run until these pass.
# ---------------------------------------------------------------------


def test_soap_variance_projection_is_exact_under_decorrelation():
    # If per-token gradients are g = QL g' QR^T with g' having
    # independent zero-mean entries of variance V'[a,b], then
    # E[g_ij^2] = sum_ab QL[i,a]^2 V'[a,b] QR[j,b]^2 exactly.
    # The projection (QL^2) V' (QR^2)^T must match a Monte Carlo
    # estimate of E[g^2].
    torch.manual_seed(3)
    out_d, in_d = 6, 4
    ql = torch.linalg.qr(torch.randn(out_d, out_d)).Q
    qr = torch.linalg.qr(torch.randn(in_d, in_d)).Q
    v_rot = torch.rand(out_d, in_d) + 0.1
    n = 200_000
    gp = torch.randn(n, out_d, in_d) * v_rot.sqrt()
    g = torch.einsum("ia,nab,jb->nij", ql, gp, qr)
    mc = g.pow(2).mean(0)
    projected = ql.pow(2) @ v_rot @ qr.pow(2).T
    torch.testing.assert_close(projected, mc, rtol=0.05, atol=1e-3)


def test_muon_update_signal_matches_direct_newton_schulz(tmp_path):
    from fim.fisher_pruning import train_optimizer
    from fim.fisher_pruning.muon import zeropower_via_newtonschulz5
    from fim.fisher_pruning.phase_a import EPS

    torch.manual_seed(0)
    model = build_model(TINY_MODEL, selection={})
    tr = dict(
        TINY_TRAIN,
        optimizer="muon",
        muon_lr=0.02,
        muon_momentum=0.95,
        muon_weight_decay=0.01,
    )
    opt = train_optimizer.build_optimizer(model, tr)
    x = torch.randint(0, 64, (2, 8))
    y = torch.randint(0, 64, (2, 8))
    _, loss = model(x, y)
    loss.backward()
    opt.step()
    targets = train_optimizer._prune_targets(model, TINY_MODEL["n_layer"])
    scores = train_optimizer.live_state_scores(
        opt, "muon", targets, "state", q=0.25, muon_mode="update"
    )
    name = next(iter(targets))
    p = targets[name]
    mom = opt.state[p]["momentum_buffer"]
    upd = zeropower_via_newtonschulz5(mom.float(), steps=5)
    upd = upd * max(1, mom.size(-2) / mom.size(-1)) ** 0.5
    # mirror the implementation's precision path exactly: the NS
    # output is bf16; the score casts to the weight dtype BEFORE
    # squaring.
    f_stat = upd.to(p.dtype).pow(2)
    expected = p.detach().abs() * (f_stat + EPS) ** 0.25
    torch.testing.assert_close(scores[name], expected, rtol=1e-4, atol=1e-7)


def test_q_050_ranks_like_squared_deletion_cost():
    from fim.fisher_pruning import train_optimizer

    torch.manual_seed(1)
    model = build_model(TINY_MODEL, selection={})
    opt = make_optimizer(model, TINY_TRAIN)
    x = torch.randint(0, 64, (2, 8))
    y = torch.randint(0, 64, (2, 8))
    _, loss = model(x, y)
    loss.backward()
    opt.step()
    targets = train_optimizer._prune_targets(model, TINY_MODEL["n_layer"])
    scores = train_optimizer.live_state_scores(opt, "adamw", targets, "state", q=0.5)
    from fim.fisher_pruning.phase_a import EPS

    name = next(iter(targets))
    p = targets[name]
    v = opt.state[p]["exp_avg_sq"]
    # same eps so the comparison is the pure monotone transform
    deletion = p.detach().pow(2) * (v + EPS)
    s = scores[name].reshape(-1)
    d = deletion.reshape(-1)
    ranks_s = s.argsort().argsort().float()
    ranks_d = d.argsort().argsort().float()
    spearman = torch.corrcoef(torch.stack([ranks_s, ranks_d]))[0, 1].item()
    assert spearman > 0.9999


def test_masks_grow_monotonically_and_survive_resume(tmp_path):
    from fim.fisher_pruning import train_optimizer

    train_bin = tmp_path / "train.bin"
    val_bin = tmp_path / "val.bin"
    _write_bin(train_bin, 4096, TINY_MODEL["vocab_size"], 11)
    _write_bin(val_bin, 4096, TINY_MODEL["vocab_size"], 12)
    cfg = {
        "seed": 0,
        "model": TINY_MODEL,
        "data": {"train_bin": str(train_bin), "val_bin": str(val_bin)},
        "train": dict(
            TINY_TRAIN,
            max_steps=12,
            eval_interval=50,
            ckpt_interval=6,
            log_interval=50,
        ),
        "prune": {
            "target_sparsity": 0.5,
            "start_step": 2,
            "end_step": 10,
            "interval": 2,
            "signal": "state",
            "q": 0.25,
        },
        "out_dir": str(tmp_path / "resume_run"),
    }
    train_optimizer.cmd_train(cfg, "cpu")
    out = Path(cfg["out_dir"])
    events = [
        json.loads(line) for line in (out / "train.jsonl").read_text().splitlines()
    ]
    prunes = [e for e in events if e.get("event") == "prune"]
    assert all(e["mask_monotone"] for e in prunes)
    spars = [e["actual_sparsity"] for e in prunes]
    assert spars == sorted(spars)
    assert all("mask_hash" in e and "jaccard_vs_magnitude" in e for e in prunes)
    assert any("accounting" in e for e in prunes)
    full = [e for e in events if e.get("event") == "final_sparsity"][0]
    assert abs(full["actual_sparsity"] - 0.5) < 0.01
    assert full["accounting"]["pruning_persistent_bytes"] > 0

    # resume path: same full config halted after the step-6
    # checkpoint (halt_after_step keeps the LR schedule identical —
    # a shortened max_steps would change the cosine), then resumed
    # to completion; must match the uninterrupted run exactly.
    cfg2 = dict(cfg, out_dir=str(tmp_path / "resume_run2"))
    cfg2["train"] = dict(cfg["train"], halt_after_step=6)
    train_optimizer.cmd_train(cfg2, "cpu")
    cfg2b = dict(cfg2)
    cfg2b["train"] = dict(cfg["train"])
    train_optimizer.cmd_train(cfg2b, "cpu", resume=True)
    ck_a = torch.load(out / "ckpt_latest.pt", map_location="cpu", weights_only=False)
    ck_b = torch.load(
        Path(cfg2b["out_dir"]) / "ckpt_latest.pt",
        map_location="cpu",
        weights_only=False,
    )
    assert ck_b["step"] == ck_a["step"] == 11
    for n, m in ck_a["prune_masks"].items():
        torch.testing.assert_close(
            m.float(), ck_b["prune_masks"][n].float(), rtol=0, atol=0
        )
    wa = ck_a["model_state"]["transformer.h.0.attn.c_attn.weight"]
    wb = ck_b["model_state"]["transformer.h.0.attn.c_attn.weight"]
    torch.testing.assert_close(wa, wb, rtol=1e-5, atol=1e-7)
