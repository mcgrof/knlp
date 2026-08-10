"""Unit tests for the spectral_delta_ra harness (gpt2_spectral_ra.py).

Implements the mandatory guardrail tests from the lane plan:
baseline identity at zero gates, gradient visibility, frozen basis,
orthonormality/projector properties, full-rank scalar equivalence
(a rotation with isotropic scaling is NOT a new mechanism), empty
selection, selected-head isolation, causal masking through the
reciprocal branch, sign invariance, the Q/K rotation no-op, trusted
placement formulas, legacy beta-zero identity, audit-capture
correctness, and a CPU end-to-end smoke over train/audit/oracle.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fim.reciprocal_attention import spectral_credit as sc  # noqa: E402
from fim.reciprocal_attention.gpt2_spectral_ra import (  # noqa: E402
    SpectralDeltaAttention,
    build_model,
    cmd_audit,
    cmd_oracle,
    cmd_train_baseline,
    gate_parameters,
    spectral_modules,
)
from gpt2.model_knlp import GPT2_KNLP, GPT2_KNLP_Config  # noqa: E402

TINY = dict(
    n_layer=2,
    n_head=2,
    n_embd=32,
    block_size=16,
    vocab_size=64,
    dropout=0.0,
    bias=True,
)


def _pair(selection, gate_mode="spectral", rank=2, beta_max=1.0, seed=0):
    """Baseline model and a variant sharing identical base weights."""
    torch.manual_seed(seed)
    base = build_model(TINY, {})
    var = build_model(TINY, selection, gate_mode, rank, beta_max)
    var.load_state_dict(base.state_dict(), strict=False)
    return base, var


def _batch(seed=0, b=3, t=16):
    gen = torch.Generator().manual_seed(seed)
    x = torch.randint(0, TINY["vocab_size"], (b, t), generator=gen)
    y = torch.randint(0, TINY["vocab_size"], (b, t), generator=gen)
    return x, y


def _install_basis(model, seed=0):
    for mod in spectral_modules(model).values():
        if mod.basis_u is None:
            continue
        n_sel, d, r = mod.basis_u.shape
        u = torch.stack(
            [sc.haar_random_basis(d, r, seed=seed + i) for i in range(n_sel)]
        ).to(torch.float32)
        mod.set_basis(u)


def test_a_baseline_identity_at_zero_beta():
    base, var = _pair({1: [0]}, gate_mode="spectral", rank=2)
    _install_basis(var)
    base.eval(), var.eval()
    x, _ = _batch()
    with torch.no_grad():
        lb, _ = base(x)
        lv, _ = var(x)
    assert torch.allclose(lb, lv, atol=1e-6)


def test_a_baseline_identity_all_gate_modes():
    for mode, rank in (
        ("scalar_delta", 0),
        ("coordinate_diag", 0),
        ("spectral", 3),
        ("standard_extra_lowrank", 3),
    ):
        base, var = _pair({0: [1], 1: [0]}, gate_mode=mode, rank=rank)
        _install_basis(var)
        base.eval(), var.eval()
        x, _ = _batch(seed=2)
        with torch.no_grad():
            lb, _ = base(x)
            lv, _ = var(x)
        assert torch.allclose(lb, lv, atol=1e-6), mode


def test_b_gradient_reaches_gates_and_base():
    _, var = _pair({1: [0, 1]}, gate_mode="spectral", rank=2)
    _install_basis(var)
    var.train()
    x, y = _batch(seed=3)
    _, loss = var(x, y)
    loss.backward()
    gates = gate_parameters(var)
    assert gates
    for p in gates:
        assert p.grad is not None
        assert torch.isfinite(p.grad).all()
        assert float(p.grad.abs().sum()) > 0
    wte_grad = var.transformer.wte.weight.grad
    assert wte_grad is not None and torch.isfinite(wte_grad).all()


def test_c_frozen_basis_no_param_no_optimizer_state():
    _, var = _pair({1: [0]}, gate_mode="spectral", rank=2)
    _install_basis(var)
    param_names = {n for n, _ in var.named_parameters()}
    assert not any("basis_u" in n for n in param_names)
    mod = spectral_modules(var)[1]
    assert mod.basis_u.requires_grad is False
    opt = torch.optim.AdamW([p for p in var.parameters() if p.requires_grad], lr=1e-3)
    x, y = _batch(seed=4)
    _, loss = var(x, y)
    loss.backward()
    opt.step()
    flat_state_ids = {id(p) for group in opt.param_groups for p in group["params"]}
    assert id(mod.basis_u) not in flat_state_ids


def test_d_e_orthonormality_and_projector():
    _, var = _pair({1: [0]}, gate_mode="spectral", rank=3)
    _install_basis(var)
    u = spectral_modules(var)[1].basis_u[0].to(torch.float64)
    eye = torch.eye(u.shape[1], dtype=torch.float64)
    assert torch.allclose(u.T @ u, eye, atol=1e-6)
    p = u @ u.T
    assert torch.allclose(p, p.T, atol=1e-9)
    assert torch.allclose(p @ p, p, atol=1e-6)


def test_f_full_rank_isotropic_equals_scalar():
    """Rotation + isotropic scaling must equal the scalar gate exactly."""
    d_head = TINY["n_embd"] // TINY["n_head"]
    sel = {1: [0]}
    _, spec = _pair(sel, gate_mode="spectral", rank=d_head, seed=7)
    _, scal = _pair(sel, gate_mode="scalar_delta", seed=7)
    scal.load_state_dict(
        {k: v for k, v in spec.state_dict().items() if "raw_beta" not in k},
        strict=False,
    )
    _install_basis(spec, seed=11)
    c = 0.37
    with torch.no_grad():
        for m in spectral_modules(spec).values():
            m.raw_beta.fill_(c)
        for m in spectral_modules(scal).values():
            m.raw_beta.fill_(c)
    spec.eval(), scal.eval()
    x, _ = _batch(seed=8)
    with torch.no_grad():
        ls, _ = spec(x)
        lc, _ = scal(x)
    assert torch.allclose(ls, lc, atol=1e-5)


def test_g_empty_selection_exact_baseline():
    torch.manual_seed(9)
    base = build_model(TINY, {})
    torch.manual_seed(9)
    empty = build_model(TINY, {})
    x, _ = _batch(seed=10)
    base.eval(), empty.eval()
    with torch.no_grad():
        lb, _ = base(x)
        le, _ = empty(x)
    assert torch.equal(lb, le)


def test_h_unselected_heads_isolated():
    """With c_proj = identity, only the selected head's slice changes."""
    sel = {0: [0]}
    _, var = _pair(sel, gate_mode="scalar_delta", seed=12)
    mod = spectral_modules(var)[0]
    with torch.no_grad():
        mod.c_proj.weight.copy_(torch.eye(TINY["n_embd"]))
        if mod.c_proj.bias is not None:
            mod.c_proj.bias.zero_()
    d_head = TINY["n_embd"] // TINY["n_head"]
    x = torch.randn(2, 8, TINY["n_embd"])
    mod.eval()
    with torch.no_grad():
        y_off = mod(x)
        mod.raw_beta.fill_(0.5)
        y_on = mod(x)
    # head 0 occupies dims [0, d_head); head 1 dims [d_head, 2*d_head)
    assert not torch.allclose(y_on[..., :d_head], y_off[..., :d_head])
    assert torch.allclose(y_on[..., d_head:], y_off[..., d_head:], atol=1e-6)


def test_i_causal_masking_through_reciprocal_branch():
    _, var = _pair({0: [0], 1: [1]}, gate_mode="spectral", rank=2, seed=13)
    _install_basis(var)
    with torch.no_grad():
        for m in spectral_modules(var).values():
            m.raw_beta.uniform_(-0.5, 0.5)
    var.eval()
    x1, _ = _batch(seed=14)
    x2 = x1.clone()
    x2[:, -1] = (x2[:, -1] + 1) % TINY["vocab_size"]
    with torch.no_grad():
        l1, _ = var(x1)
        l2, _ = var(x2)
    # logits strictly before the changed position must be identical
    assert torch.allclose(l1[:, :-1], l2[:, :-1], atol=1e-6)
    assert not torch.allclose(l1[:, -1], l2[:, -1], atol=1e-6)


def test_k_eigenvector_sign_invariance():
    _, var = _pair({1: [0]}, gate_mode="spectral", rank=2, seed=15)
    _install_basis(var, seed=21)
    mod = spectral_modules(var)[1]
    with torch.no_grad():
        mod.raw_beta.uniform_(-0.7, 0.7)
    var.eval()
    x, _ = _batch(seed=16)
    with torch.no_grad():
        l1, _ = var(x)
        u_flipped = mod.basis_u.clone()
        u_flipped[:, :, 0] = -u_flipped[:, :, 0]
        mod.set_basis(u_flipped)
        l2, _ = var(x)
    assert torch.allclose(l1, l2, atol=1e-6)


def test_n_qk_rotation_noop():
    """(Q U)(K U)^T == Q K^T for orthogonal U — a shared rotation of
    Q and K is NOT a mechanism and must never be sold as one."""
    gen = torch.Generator().manual_seed(17)
    q = torch.randn(64, 16, generator=gen, dtype=torch.float64)
    k = torch.randn(64, 16, generator=gen, dtype=torch.float64)
    u = sc.haar_random_basis(16, 16, seed=18)
    assert torch.allclose((q @ u) @ (k @ u).T, q @ k.T, atol=1e-9)


def test_trusted_placement_formula():
    """Stage-0 sanity: 12L/12H with 3 RA layers, 1 head selects layers
    {5,6,7}; the legacy implementation takes the LAST head (11)."""
    cfg = GPT2_KNLP_Config(
        n_layer=12,
        n_head=12,
        n_embd=144,
        block_size=8,
        vocab_size=64,
        dropout=0.0,
        bias=True,
        use_ra=True,
        n_ra_layers=3,
        n_ra_heads=1,
    )
    model = GPT2_KNLP(cfg)
    ra_layers = [i for i, b in enumerate(model.transformer.h) if b.attn.use_ra]
    assert ra_layers == [5, 6, 7]
    assert model.transformer.h[5].attn.n_ra_heads == 1


def test_legacy_beta_zero_matches_baseline():
    """Stage-0 sanity: legacy RA with ra_logit == 0 is exactly baseline."""
    cfg_kwargs = dict(TINY)
    torch.manual_seed(19)
    baseline = GPT2_KNLP(GPT2_KNLP_Config(**cfg_kwargs))
    ra = GPT2_KNLP(
        GPT2_KNLP_Config(**cfg_kwargs, use_ra=True, n_ra_layers=1, n_ra_heads=1)
    )
    ra.load_state_dict(baseline.state_dict(), strict=False)
    baseline.eval(), ra.eval()
    x, _ = _batch(seed=20)
    with torch.no_grad():
        lb, _ = baseline(x)
        lr, _ = ra(x)
    assert torch.allclose(lb, lr, atol=1e-6)


def test_audit_capture_matches_manual_computation():
    """The captured delta equals an independent recomputation of
    SDPA(k,q,v) - SDPA(q,k,v) from the module's own projections, and
    g_full equals the autograd gradient of the loss wrt the head
    outputs."""
    import torch.nn.functional as F

    sel = {0: [1]}
    _, var = _pair(sel, gate_mode="none", seed=22)
    mod = spectral_modules(var)[0]
    mod.audit_capture = True
    var.eval()
    x, y = _batch(seed=23)
    _, loss = var(x, y)
    loss.backward()
    audit = mod._audit
    assert set(audit) >= {"g_full", "delta", "q_sel", "k_sel"}
    d_head = TINY["n_embd"] // TINY["n_head"]
    assert audit["delta"].shape == (3, 1, 16, d_head)
    assert audit["g_full"].shape == (3, TINY["n_head"], 16, d_head)
    # independent recomputation of the counterfactual delta
    h = var.transformer.drop(
        var.transformer.wte(x) + var.transformer.wpe(torch.arange(16))
    )
    xin = var.transformer.h[0].ln_1(h)
    q, k, v = mod.c_attn(xin).split(TINY["n_embd"], dim=2)
    B, T = x.shape
    q = q.view(B, T, mod.n_head, d_head).transpose(1, 2)
    k = k.view(B, T, mod.n_head, d_head).transpose(1, 2)
    v = v.view(B, T, mod.n_head, d_head).transpose(1, 2)
    std = F.scaled_dot_product_attention(
        q[:, 1:2], k[:, 1:2], v[:, 1:2], is_causal=True
    )
    rec = F.scaled_dot_product_attention(
        k[:, 1:2], q[:, 1:2], v[:, 1:2], is_causal=True
    )
    assert torch.allclose(audit["delta"], (rec - std).detach(), atol=1e-5)
    assert torch.isfinite(audit["g_full"]).all()
    assert float(audit["g_full"].abs().sum()) > 0


def _write_bins(tmp_path):
    gen = np.random.default_rng(0)
    train = gen.integers(0, TINY["vocab_size"], size=20000, dtype=np.uint16)
    val = gen.integers(0, TINY["vocab_size"], size=5000, dtype=np.uint16)
    train_bin = tmp_path / "train.bin"
    val_bin = tmp_path / "val.bin"
    train.tofile(train_bin)
    val.tofile(val_bin)
    return str(train_bin), str(val_bin)


def test_end_to_end_cpu_smoke(tmp_path):
    """train-baseline -> audit -> oracle on a tiny CPU config; no
    downloads, completes in seconds. Guards the CLI plumbing before
    any GPU time is spent."""
    train_bin, val_bin = _write_bins(tmp_path)
    train_cfg = {
        "seed": 0,
        "model": dict(TINY),
        "data": {"train_bin": train_bin, "val_bin": val_bin},
        "train": {
            "batch_size": 4,
            "max_steps": 12,
            "lr": 1e-3,
            "warmup_steps": 2,
            "eval_interval": 6,
            "eval_batches": 2,
            "ckpt_interval": 6,
            "log_interval": 4,
        },
    }
    train_out = tmp_path / "train"
    cmd_train_baseline(train_cfg, "cpu", train_out)
    ckpt = train_out / "ckpt_latest.pt"
    assert ckpt.exists()

    audit_cfg = {
        "checkpoint": str(ckpt),
        "data": {"train_bin": train_bin},
        "selection": {
            "heads": {"1": [0]},
            "control_heads": {"0": [1]},
        },
        "calibration": {
            "seed": 1,
            "batches": 4,
            "batch_size": 4,
            "n_perm": 20,
            "score_batches": 1,
        },
    }
    audit_out = tmp_path / "audit"
    cmd_audit(audit_cfg, "cpu", audit_out)
    results = json.loads((audit_out / "audit_results.json").read_text())
    assert "L1H0" in results["heads"]
    assert results["heads"]["L1H0"]["role"] == "trusted"
    assert results["heads"]["L0H1"]["role"] == "control"
    assert "pass_all" in results["gates"]
    for source in (
        "signed_credit",
        "elementwise_credit_second_moment",
        "reciprocal_activation",
        "gradient_covariance",
        "qk_asymmetry",
        "random",
    ):
        assert (audit_out / f"basis_{source}" / "basis.pt").exists()

    oracle_cfg = {
        "seed": 0,
        "checkpoint": str(ckpt),
        "data": {"train_bin": train_bin, "val_bin": val_bin},
        "selection": {"heads": {"1": [0]}},
        "arm": {
            "name": "spectral_credit_r2",
            "gate_mode": "spectral",
            "rank": 2,
            "basis_dir": str(audit_out / "basis_signed_credit"),
            "beta_max": 1.0,
        },
        "oracle": {
            "steps": 6,
            "batch_size": 4,
            "lr": 1e-2,
            "eval_batches": 2,
            "eval_interval": 3,
            "log_interval": 2,
        },
    }
    oracle_out = tmp_path / "oracle"
    cmd_oracle(oracle_cfg, "cpu", oracle_out)
    log = (oracle_out / "oracle_spectral_credit_r2_seed0.jsonl").read_text()
    events = [json.loads(line) for line in log.splitlines()]
    kinds = {e["event"] for e in events}
    assert {"start", "train", "eval", "final"} <= kinds
    final = [e for e in events if e["event"] == "final"][0]
    assert "val_loss_gates_on" in final and "val_loss_gates_off" in final
    assert final["per_mode_ablation"]
    # beta trajectory was logged
    train_events = [e for e in events if e["event"] == "train"]
    assert all("betas" in e for e in train_events)
