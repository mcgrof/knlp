"""Gradient-semantics test matrix for the Trellis memory operator.

Proves the two training-semantics axes (state staleness x outer-gradient
order) are implemented correctly and stay LIVE:

  - the sequential full-bilevel path matches a pure-autograd per-token
    unrolled oracle in forward and in every input gradient;
  - the chunk-start full-bilevel reference matches an independent
    per-token chunk-start oracle, and coincides with the sequential path
    at C=1;
  - forward values are gradient-mode independent while gradients differ
    (the toggle cannot silently die -- the historical defect was exactly
    a dead exact_inner toggle on the chunked path);
  - the SiLU closed-form inner code equals nested autograd through first
    and second derivatives;
  - the mixer dispatches each resolved mode to a backend that can honor
    it and reports it truthfully;
  - legacy exact_inner configs resolve to the semantics the old code
    actually executed.

Slot count != head dim everywhere (m=7, d=5) so transposition errors
cannot hide inside square matrices. The mixer itself is ordinary autograd
composed around the operator verified here, so operator-level oracle
checks plus the model-level dispatch/liveness checks cover the two-pass
gradient path.

LN-SiLU tolerances are looser (1e-6) than the pointwise phi (1e-11): the
recurrence starts at M=0 so the first inner code sits at LayerNorm's
degenerate point (var=0, eps-floored sigma), where 1/sigma^3 second-
derivative terms amplify benign op-ordering roundoff. The closed form
itself matches autograd to 1e-14 away from that point.
"""

from __future__ import annotations

import warnings

import pytest
import torch

from trellis_lm.activations import identity, ln_silu, silu, silu_vjp_from_alpha
from trellis_lm.config import TrellisConfig
from trellis_lm.model import TrellisLM
from trellis_lm.trellis_memory import (
    run_trellis_memory,
    run_trellis_memory_chunked_full_bilevel_reference,
    run_trellis_memory_chunked_state_evolution,
)

DT = torch.float64
B, H, T, D, M = 2, 2, 8, 5, 7

PHIS = {"identity": identity, "silu": silu, "ln_silu": ln_silu}
TOL = {"identity": 1e-11, "silu": 1e-11, "ln_silu": 1e-6}


def _inputs(read_mode, seed=0):
    torch.manual_seed(seed)
    w = torch.randn(B, H, T, D, dtype=DT, requires_grad=True)
    rdim = D if read_mode == "M_q" else M
    r = torch.randn(B, H, T, rdim, dtype=DT, requires_grad=True)
    a = torch.randn(B, H, T, M, dtype=DT, requires_grad=True)
    b = (torch.rand(B, H, T, 1, dtype=DT) * 0.2 + 0.79).requires_grad_(True)
    g = (torch.rand(H, dtype=DT) * 0.05 + 0.01).requires_grad_(True)
    return w, r, a, b, g


def _loss(y):
    # nonlinear reduction so second-order terms actually matter
    return (y * torch.sin(y * 0.7)).sum()


def _inner_u(phi, z, alpha):
    pred = phi(z)
    err = pred - alpha
    (u,) = torch.autograd.grad(
        pred, z, grad_outputs=err, create_graph=True, retain_graph=True
    )
    return u


def _oracle_sequential(w, r, a, b, g, phi, read_mode, m_init=None):
    """Per-token unrolled bilevel oracle: live state, inner VJP kept in
    the graph, write-before-read readout."""
    Mst = torch.zeros(B, H, M, D, dtype=DT) if m_init is None else m_init
    gv = g.view(1, H, 1, 1)
    outs = []
    for t in range(T):
        z = torch.einsum("bhmd,bhd->bhm", Mst, w[:, :, t])
        u = _inner_u(phi, z, a[:, :, t])
        bt = b[:, :, t].unsqueeze(-1)
        Mst = bt * Mst - gv * torch.einsum("bhm,bhd->bhmd", u, w[:, :, t])
        if read_mode == "M_q":
            outs.append(torch.einsum("bhmd,bhd->bhm", Mst, r[:, :, t]))
        else:
            outs.append(torch.einsum("bhmd,bhm->bhd", Mst, r[:, :, t]))
    return torch.stack(outs, dim=2)


def _oracle_chunk_start(w, r, a, b, g, phi, read_mode, C):
    """Per-token unrolled chunk-start oracle: every z in a chunk uses the
    chunk-start state M0 (M0 stays connected across chunks), inner VJP in
    the graph, sequential advance and write-before-read readout."""
    Mst = torch.zeros(B, H, M, D, dtype=DT)
    gv = g.view(1, H, 1, 1)
    outs = []
    for c0 in range(0, T, C):
        c1 = min(c0 + C, T)
        M0 = Mst
        us = []
        for t in range(c0, c1):
            z = torch.einsum("bhmd,bhd->bhm", M0, w[:, :, t])
            us.append(_inner_u(phi, z, a[:, :, t]))
        for i, t in enumerate(range(c0, c1)):
            bt = b[:, :, t].unsqueeze(-1)
            Mst = bt * Mst - gv * torch.einsum("bhm,bhd->bhmd", us[i], w[:, :, t])
            if read_mode == "M_q":
                outs.append(torch.einsum("bhmd,bhd->bhm", Mst, r[:, :, t]))
            else:
                outs.append(torch.einsum("bhmd,bhm->bhd", Mst, r[:, :, t]))
    return torch.stack(outs, dim=2)


def _grads(loss, inputs):
    return torch.autograd.grad(loss, inputs, retain_graph=False)


@pytest.mark.parametrize("phi_name", list(PHIS))
@pytest.mark.parametrize("read_mode", ["M_q", "M_T_r"])
def test_sequential_full_bilevel_matches_unrolled_oracle(phi_name, read_mode):
    phi, tol = PHIS[phi_name], TOL[phi_name]
    w, r, a, b, g = _inputs(read_mode)
    y_ref = _oracle_sequential(w, r, a, b, g, phi, read_mode)
    g_ref = _grads(_loss(y_ref), (w, r, a, b, g))
    y = run_trellis_memory(
        w, r, a, b, g, phi, read_mode, training=True, exact_inner=True
    )
    assert (y - y_ref).abs().max().item() < tol
    g_new = _grads(_loss(y), (w, r, a, b, g))
    for gr, gn in zip(g_ref, g_new):
        assert (gr - gn).abs().max().item() < tol


@pytest.mark.parametrize("phi_name", list(PHIS))
def test_sequential_full_bilevel_m_init_gradient(phi_name):
    phi, tol = PHIS[phi_name], TOL[phi_name]
    w, r, a, b, g = _inputs("M_q")
    torch.manual_seed(3)
    m0 = torch.randn(B, H, M, D, dtype=DT, requires_grad=True)
    y_ref = _oracle_sequential(w, r, a, b, g, phi, "M_q", m_init=m0)
    (g_ref,) = _grads(_loss(y_ref), (m0,))
    y = run_trellis_memory(
        w, r, a, b, g, phi, "M_q", training=True, exact_inner=True, M_init=m0
    )
    (g_new,) = _grads(_loss(y), (m0,))
    assert (g_ref - g_new).abs().max().item() < tol


@pytest.mark.parametrize("phi_name", list(PHIS))
@pytest.mark.parametrize("read_mode", ["M_q", "M_T_r"])
@pytest.mark.parametrize("C", [2, 4])
def test_chunk_start_full_bilevel_matches_unrolled_oracle(phi_name, read_mode, C):
    phi, tol = PHIS[phi_name], TOL[phi_name]
    w, r, a, b, g = _inputs(read_mode)
    y_ref = _oracle_chunk_start(w, r, a, b, g, phi, read_mode, C)
    g_ref = _grads(_loss(y_ref), (w, r, a, b, g))
    y = run_trellis_memory_chunked_full_bilevel_reference(
        w, r, a, b, g, phi, read_mode, C
    )
    assert (y - y_ref).abs().max().item() < tol
    g_new = _grads(_loss(y), (w, r, a, b, g))
    for gr, gn in zip(g_ref, g_new):
        assert (gr - gn).abs().max().item() < tol


@pytest.mark.parametrize("phi_name", list(PHIS))
def test_chunk_start_c1_full_bilevel_matches_sequential(phi_name):
    phi, tol = PHIS[phi_name], TOL[phi_name]
    w, r, a, b, g = _inputs("M_q")
    y_seq = run_trellis_memory(
        w, r, a, b, g, phi, "M_q", training=True, exact_inner=True
    )
    g_seq = _grads(_loss(y_seq), (w, r, a, b, g))
    w2, r2, a2, b2, g2 = _inputs("M_q")
    y_c1 = run_trellis_memory_chunked_full_bilevel_reference(
        w2, r2, a2, b2, g2, phi, "M_q", 1
    )
    assert (y_seq - y_c1).abs().max().item() < tol
    g_c1 = _grads(_loss(y_c1), (w2, r2, a2, b2, g2))
    for gs, gc in zip(g_seq, g_c1):
        assert (gs - gc).abs().max().item() < tol


@pytest.mark.parametrize("phi_name", list(PHIS))
def test_forward_equal_gradients_differ_across_modes(phi_name):
    """The semantic toggle must be LIVE: identical forward values, strictly
    different gradients. A dead toggle (the historical defect) fails here."""
    phi = PHIS[phi_name]
    C = 4
    res = {}
    for mode in ("full_bilevel", "first_order_detached"):
        w, r, a, b, g = _inputs("M_q", seed=1)
        M0s, us, P, rmat, _ = run_trellis_memory_chunked_state_evolution(
            w, a, b, g, phi, C, outer_gradient_mode=mode
        )
        loss = _loss(us) + (M0s * 0.7).sum()
        res[mode] = (
            us.detach().clone(),
            _grads(loss, (w, a, b, g)),
        )
    u_full, g_full = res["full_bilevel"]
    u_first, g_first = res["first_order_detached"]
    assert (u_full - u_first).abs().max().item() == 0.0
    diff = max((x - y).abs().max().item() for x, y in zip(g_full, g_first))
    assert diff > 1e-9, "gradient modes are indistinguishable: dead toggle"


def test_silu_closed_form_matches_autograd_to_second_order():
    torch.manual_seed(0)
    z = torch.randn(3, M, dtype=DT, requires_grad=True)
    a = torch.randn(3, M, dtype=DT, requires_grad=True)
    u_ref = _inner_u(silu, z, a)
    u_cf = silu_vjp_from_alpha(z, a)
    assert (u_ref - u_cf).abs().max().item() < 1e-13
    bar = torch.randn_like(u_ref)
    g_ref = torch.autograd.grad((u_ref * bar).sum(), (z, a), retain_graph=True)
    g_cf = torch.autograd.grad((u_cf * bar).sum(), (z, a), retain_graph=True)
    for gr, gc in zip(g_ref, g_cf):
        assert (gr - gc).abs().max().item() < 1e-12
    assert torch.autograd.gradcheck(silu_vjp_from_alpha, (z, a))
    assert torch.autograd.gradgradcheck(silu_vjp_from_alpha, (z, a))


def _model(mode, **kw):
    torch.manual_seed(0)
    cfg = TrellisConfig(
        vocab_size=101,
        d_model=32,
        n_layers=2,
        n_heads=2,
        d_head=8,
        n_slots=6,
        max_seq_len=32,
        chunk_size=4,
        activation="silu",
        dtype="fp32",
        gamma_init=0.1,
        beta_init=0.9,
        trellis_state_mode="chunk_start_stale",
        trellis_outer_gradient_mode=mode,
        **kw,
    )
    return TrellisLM(cfg)


def _model_grads(m):
    torch.manual_seed(7)
    idx = torch.randint(0, 101, (2, 16))
    tgt = torch.randint(0, 101, (2, 16))
    logits, loss = m(idx, tgt)
    loss.backward()
    return (
        logits.detach(),
        {n: p.grad.clone() for n, p in m.named_parameters() if p.grad is not None},
    )


def test_mixer_dispatch_reports_and_honors_gradient_mode():
    out = {}
    for mode in ("full_bilevel", "first_order_detached"):
        m = _model(mode)
        logits, grads = _model_grads(m)
        out[mode] = (logits, grads, m.blocks[0].mixer.last_trellis_diag)
    lf, gf, df = out["full_bilevel"]
    l1, g1, d1 = out["first_order_detached"]
    assert df["backend"] == "pytorch_chunk_start_full_bilevel"
    assert df["outer_gradient_mode"] == "full_bilevel"
    assert d1["backend"] == "pytorch_chunk_start_first_order"
    assert d1["outer_gradient_mode"] == "first_order_detached"
    # forward values are gradient-mode independent (bit-identical on CPU)
    assert (lf - l1).abs().max().item() == 0.0
    # same-mode rebuild is deterministic ...
    _, gf2 = _model_grads(_model("full_bilevel"))
    assert all((gf[n] - gf2[n]).abs().max().item() == 0.0 for n in gf)
    # ... so any strictly nonzero cross-mode difference proves the toggle
    # is live in the full two-pass model
    cross = max((gf[n] - g1[n]).abs().max().item() for n in gf)
    assert cross > 0.0, "gradient modes are indistinguishable through the mixer"


def test_full_bilevel_requests_fail_loudly_when_unservable():
    # refine path cannot serve bilevel
    with pytest.raises((ValueError, NotImplementedError)):
        _model("full_bilevel", chunk_refine=-1)
    # sequential_current with chunked dispatch is a contradiction
    with pytest.raises(ValueError):
        TrellisConfig(
            chunk_size=16, trellis_state_mode="sequential_current"
        ).resolve_training_semantics(warn=False)


def test_legacy_exact_inner_resolution():
    # chunked legacy config: stored exact_inner=True was never honored
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        sem = TrellisConfig(
            chunk_size=16, exact_inner=True
        ).resolve_training_semantics()
    assert sem["state_mode"] == "chunk_start_stale"
    assert sem["outer_gradient_mode"] == "first_order_detached"
    assert any("never reached" in str(x.message) for x in w)
    # sequential legacy configs map the boolean faithfully
    sem = TrellisConfig(exact_inner=True).resolve_training_semantics(warn=False)
    assert sem["outer_gradient_mode"] == "full_bilevel"
    sem = TrellisConfig(exact_inner=False).resolve_training_semantics(warn=False)
    assert sem["outer_gradient_mode"] == "first_order_detached"
    # per-slot beta always dispatched sequentially, so exact_inner was honored
    sem = TrellisConfig(
        chunk_size=16, beta_mode="per_slot", exact_inner=True
    ).resolve_training_semantics(warn=False)
    assert sem["state_mode"] == "sequential_current"
    assert sem["outer_gradient_mode"] == "full_bilevel"
    # the affine write has no inner VJP: exact by construction
    sem = TrellisConfig(
        trellis_write_mode="input_conditioned", chunk_size=16
    ).resolve_training_semantics(warn=False)
    assert sem["write_path"] == "input_conditioned_affine"
    assert sem["outer_gradient_mode"] == "exact_affine"


def test_profiles_pin_semantics_and_required_knobs():
    cfg = TrellisConfig.profile("correctness_oracle")
    sem = cfg.resolve_training_semantics(warn=False)
    assert sem["state_mode"] == "sequential_current"
    assert sem["outer_gradient_mode"] == "full_bilevel"
    cfg = TrellisConfig.profile("repaired_silu_full", chunk_size=16)
    assert cfg.phi_activation == "silu" and cfg.f_activation == "ln_silu"
    assert (
        cfg.resolve_training_semantics(warn=False)["outer_gradient_mode"]
        == "full_bilevel"
    )
    with pytest.raises(ValueError):
        TrellisConfig.profile("paper_chunk_full", chunk_size=16)
    with pytest.raises(ValueError):
        TrellisConfig.profile("no_such_profile")
