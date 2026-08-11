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
    evaluate_audit_gates,
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


def test_deterministic_basis_generation(tmp_path):
    """Same checkpoint + same calibration seed -> bit-identical basis
    artifacts for EVERY source, including the Haar-random control."""
    train_bin, val_bin = _write_bins(tmp_path)
    train_cfg = {
        "seed": 0,
        "model": dict(TINY),
        "data": {"train_bin": train_bin, "val_bin": val_bin},
        "train": {
            "batch_size": 4,
            "max_steps": 6,
            "lr": 1e-3,
            "warmup_steps": 2,
            "eval_interval": 5,
            "eval_batches": 1,
            "ckpt_interval": 5,
            "log_interval": 4,
        },
    }
    train_out = tmp_path / "train"
    cmd_train_baseline(train_cfg, "cpu", train_out)
    audit_cfg = {
        "checkpoint": str(train_out / "ckpt_latest.pt"),
        "data": {"train_bin": train_bin},
        "selection": {"heads": {"1": [0]}},
        "calibration": {
            "seed": 1,
            "batches": 3,
            "batch_size": 4,
            "n_perm": 10,
            "score_batches": 0,
        },
    }
    cmd_audit(audit_cfg, "cpu", tmp_path / "a1")
    cmd_audit(audit_cfg, "cpu", tmp_path / "a2")
    for source in (
        "signed_credit",
        "random",
        "qk_asymmetry",
        "gradient_covariance",
    ):
        b1 = (tmp_path / "a1" / f"basis_{source}" / "basis.pt").read_bytes()
        b2 = (tmp_path / "a2" / f"basis_{source}" / "basis.pt").read_bytes()
        assert b1 == b2, source


def test_audit_refuses_overwrite(tmp_path):
    out = tmp_path / "audit"
    out.mkdir()
    (out / "audit_results.json").write_text("{}")
    with pytest.raises(RuntimeError, match="refusing to overwrite"):
        cmd_audit({"selection": {"heads": {}}}, "cpu", out)


def test_delta_path_is_dropout_free_in_train_mode(monkeypatch):
    """In train mode with dropout > 0, the two SDPA calls building the
    delta must run with dropout_p=0; only the main path keeps dropout."""
    import torch.nn.functional as F

    cfg_kwargs = dict(TINY)
    cfg_kwargs["dropout"] = 0.5
    torch.manual_seed(40)
    var = build_model(cfg_kwargs, {0: [1]}, gate_mode="scalar_delta")
    calls = []
    real_sdpa = F.scaled_dot_product_attention

    def spy(q, k, v, **kw):
        calls.append(kw.get("dropout_p", None))
        return real_sdpa(q, k, v, **kw)

    monkeypatch.setattr(F, "scaled_dot_product_attention", spy)
    var.train()
    x, y = _batch(seed=41)
    with torch.no_grad():
        spectral_modules(var)[0].raw_beta.fill_(0.3)
    _, loss = var(x, y)
    # layer 0 (spectral module): main call dropout 0.5, then rec and
    # std_clean at 0.0; layer 1 is stock attention at 0.5
    layer0_calls = calls[:3]
    assert layer0_calls[0] == 0.5
    assert layer0_calls[1] == 0.0 and layer0_calls[2] == 0.0


def test_oracle_rejects_mismatched_basis_checkpoint(tmp_path):
    train_bin, val_bin = _write_bins(tmp_path)
    base_train = {
        "seed": 0,
        "model": dict(TINY),
        "data": {"train_bin": train_bin, "val_bin": val_bin},
        "train": {
            "batch_size": 4,
            "max_steps": 5,
            "lr": 1e-3,
            "warmup_steps": 2,
            "eval_interval": 4,
            "eval_batches": 1,
            "ckpt_interval": 4,
            "log_interval": 4,
        },
    }
    out_a = tmp_path / "ta"
    cmd_train_baseline(base_train, "cpu", out_a)
    out_b = tmp_path / "tb"
    cmd_train_baseline({**base_train, "seed": 5}, "cpu", out_b)
    audit_cfg = {
        "checkpoint": str(out_a / "ckpt_latest.pt"),
        "data": {"train_bin": train_bin},
        "selection": {"heads": {"1": [0]}},
        "calibration": {
            "seed": 1,
            "batches": 2,
            "batch_size": 4,
            "n_perm": 5,
            "score_batches": 0,
        },
    }
    audit_out = tmp_path / "audit_a"
    cmd_audit(audit_cfg, "cpu", audit_out)
    oracle_cfg = {
        "seed": 0,
        "checkpoint": str(out_b / "ckpt_latest.pt"),  # WRONG checkpoint
        "data": {"train_bin": train_bin, "val_bin": val_bin},
        "selection": {"heads": {"1": [0]}},
        "arm": {
            "name": "mismatch",
            "gate_mode": "spectral",
            "rank": 2,
            "basis_dir": str(audit_out / "basis_signed_credit"),
        },
        "oracle": {"steps": 1, "batch_size": 4, "lr": 1e-2, "eval_batches": 1},
    }
    with pytest.raises(RuntimeError, match="derived from checkpoint"):
        cmd_oracle(oracle_cfg, "cpu", tmp_path / "oracle")


def _gate_head(
    beats=True,
    top1=0.7,
    top2=0.7,
    top4=0.7,
    split_r1=0.9,
    cross_r1=0.9,
    cancel=0.9,
    role="trusted",
):
    return {
        "role": role,
        "permutation_null": {"exceeds_p95": beats},
        "signed": {
            "top1_mass_fraction": top1,
            "top2_mass_fraction": top2,
            "top4_mass_fraction": top4,
            "cancellation_ratio": cancel,
        },
        "split_half": {
            "split_half_overlap_r1": split_r1,
            "split_half_overlap_r2": split_r1,
            "split_half_overlap_r4": split_r1,
        },
        "cross_seed": {
            "cross_seed_overlap_r1": cross_r1,
            "cross_seed_overlap_r2": cross_r1,
            "cross_seed_overlap_r4": cross_r1,
        },
    }


def test_gates_require_joint_per_head_survival():
    """Three trusted heads each passing a different PAIR of conditions
    must NOT pass: the conjunction is per head, not per condition."""
    results = {
        "A": _gate_head(beats=True, top1=0.7, split_r1=0.1, cross_r1=0.1),
        "B": _gate_head(beats=True, top1=0.1, top2=0.1, top4=0.1),
        "C": _gate_head(beats=False, top1=0.7),
        "X": _gate_head(role="control", beats=False, top1=0.1),
    }
    gates = evaluate_audit_gates(results, {})
    assert gates["pass_existence"]  # 2/3 beat the null
    assert not gates["pass_joint_survival"]  # but no head survives jointly
    assert not gates["pass_all"]


def test_gates_stability_tied_to_mass_rank():
    """A head whose mass needs rank 4 must be stable at rank 4; an
    incidental rank-1 overlap must not rescue it."""
    head = _gate_head(top1=0.1, top2=0.2, top4=0.65)
    head["split_half"] = {
        "split_half_overlap_r1": 0.95,
        "split_half_overlap_r2": 0.95,
        "split_half_overlap_r4": 0.10,
    }
    from fim.reciprocal_attention.gpt2_spectral_ra import _head_survival

    s = _head_survival(head, mass_bar=0.60, overlap_bar=0.60)
    assert s["r_star"] == 4
    assert not s["stable"]
    assert not s["survives"]


def test_gates_controls_margin():
    """controls_distinct needs a margin, not a strict inequality."""
    results = {
        "A": _gate_head(),
        "B": _gate_head(),
        "C": _gate_head(),
        "X": _gate_head(role="control"),
        "Y": _gate_head(role="control"),
        "Z": _gate_head(role="control", beats=False),
    }
    # trusted 3/3 vs controls 2/3: 1.0 >= 2/3 + 0.25 -> distinct
    gates = evaluate_audit_gates(results, {})
    assert gates["trusted_frac_survives"] == 1.0
    assert abs(gates["control_frac_survives"] - 2 / 3) < 1e-9
    assert gates["controls_distinct"]
    # all controls surviving too: no margin left -> not distinct
    results["Z"] = _gate_head(role="control")
    gates = evaluate_audit_gates(results, {})
    assert gates["control_frac_survives"] == 1.0
    assert not gates["controls_distinct"]
    assert not gates["pass_all"]


def test_bindata_skip_matches_stream(tmp_path):
    """Fast-forwarding the seeded stream reproduces the unforked
    order: batches 5..9 after skip(5) equal batches 5..9 straight."""
    from fim.reciprocal_attention.gpt2_spectral_ra import BinData

    train_bin, _ = _write_bins(tmp_path)
    data = BinData(train_bin, TINY["block_size"])
    g1 = torch.Generator().manual_seed(7)
    straight = [data.batch(4, g1, "cpu")[0] for _ in range(10)]
    g2 = torch.Generator().manual_seed(7)
    data.skip(5, 4, g2)
    forked = [data.batch(4, g2, "cpu")[0] for _ in range(5)]
    for a, b in zip(straight[5:], forked):
        assert torch.equal(a, b)


def _mk_warmup(tmp_path, stop=6, max_steps=20):
    train_bin, val_bin = _write_bins(tmp_path)
    cfg = {
        "seed": 0,
        "model": dict(TINY),
        "data": {"train_bin": train_bin, "val_bin": val_bin},
        "selection": {"heads": {"1": [0]}},
        "arm": {"type": "baseline", "name": "warmup"},
        "train": {
            "batch_size": 4,
            "max_steps": max_steps,
            "stop_step": stop,
            "lr": 1e-3,
            "warmup_steps": 2,
            "eval_interval": 100,
            "eval_batches": 1,
            "log_interval": 100,
            "gate_log_interval": 100,
        },
    }
    out = tmp_path / "warmup"
    from fim.reciprocal_attention.gpt2_spectral_ra import cmd_train_arm

    cmd_train_arm(cfg, "cpu", out)
    return cfg, out / "ckpt_warmup_seed0.pt"


def test_fork_identity_gated_arms(tmp_path):
    """legacy_ra, scalar_delta, and spectral arms forked from a
    warmup checkpoint start EXACTLY at the warmup model (gates zero
    at init); sdpa_gate does not (fresh gate params perturb it) and
    that is the documented exception."""
    from fim.reciprocal_attention.gpt2_spectral_ra import build_arm_model

    cfg, ckpt_path = _mk_warmup(tmp_path)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    torch.manual_seed(99)
    base, _ = build_arm_model(cfg["model"], {"type": "baseline"}, {})
    base.load_state_dict(ckpt["model_state"], strict=False)
    base.eval()
    x, _ = _batch(seed=50)
    with torch.no_grad():
        ref, _ = base(x)
    for arm in (
        {"type": "legacy_ra"},
        {"type": "scalar_delta"},
        {"type": "spectral", "rank": 2},
    ):
        torch.manual_seed(123)
        m, allow = build_arm_model(cfg["model"], arm, {1: [0]})
        missing, unexpected = m.load_state_dict(ckpt["model_state"], strict=False)
        assert not [k for k in unexpected if "attn.bias" not in k]
        assert all(any(a in k for a in allow) for k in missing), (arm, missing)
        if arm["type"] == "spectral":
            _install_basis(m)
        m.eval()
        with torch.no_grad():
            got, _ = m(x)
        assert torch.allclose(ref, got, atol=1e-6), arm["type"]
    torch.manual_seed(123)
    gate_model, _ = build_arm_model(cfg["model"], {"type": "sdpa_gate"}, {})
    gate_model.load_state_dict(ckpt["model_state"], strict=False)
    gate_model.eval()
    with torch.no_grad():
        gated, _ = gate_model(x)
    assert not torch.allclose(ref, gated, atol=1e-3)


def test_train_arm_fork_end_to_end(tmp_path):
    """Warmup -> forked legacy_ra arm: events, gate logging, final
    on/off ablation all present; fork metadata recorded."""
    from fim.reciprocal_attention.gpt2_spectral_ra import cmd_train_arm

    cfg, ckpt_path = _mk_warmup(tmp_path)
    arm_cfg = json.loads(json.dumps(cfg))
    arm_cfg["arm"] = {
        "type": "legacy_ra",
        "name": "legacy",
        "fork_from": str(ckpt_path),
    }
    arm_cfg["train"]["stop_step"] = 10
    arm_cfg["train"]["log_interval"] = 2
    arm_cfg["train"]["gate_log_interval"] = 2
    out = tmp_path / "arm"
    cmd_train_arm(arm_cfg, "cpu", out)
    events = [json.loads(l) for l in open(out / "train_legacy_seed0.jsonl")]
    kinds = {e["event"] for e in events}
    assert {"start", "train", "gate", "eval", "final"} <= kinds
    start = [e for e in events if e["event"] == "start"][0]
    assert start["start_step"] == 6
    assert start["fork_sha256"]
    final = [e for e in events if e["event"] == "final"][0]
    assert final["val_loss_gates_off"] is not None
    gate_events = [e for e in events if e["event"] == "gate"]
    assert all("L" in list(g["state"])[0] for g in gate_events)
    assert (out / "final_model_legacy_seed0.pt").exists()
