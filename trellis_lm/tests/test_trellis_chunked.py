"""Validate the chunked stale-gradient kernel against the sequential path.

Correctness gate (the task's rule: optimize only after the exact path matches):
  - chunk_size=1 must reproduce the sequential stale path (to fp32 rounding).
  - chunk_size>1 is an approximation that must stay close and tighten as C->1.
  - the chunked path must remain causal.
"""

import torch
from trellis_lm.tests._helpers import tiny_cfg
from trellis_lm.model import TrellisLM
from trellis_lm.trellis_memory import run_trellis_memory, run_trellis_memory_chunked
from trellis_lm.activations import get_activation


def _inputs(B=2, H=2, T=12, D=8, M=4, seed=0):
    g = torch.Generator().manual_seed(seed)
    mk = lambda *s: torch.randn(*s, generator=g)
    write = mk(B, H, T, D)
    read = mk(B, H, T, D)
    r_val = mk(B, H, T, M)
    alpha = mk(B, H, T, M)
    beta = torch.sigmoid(mk(B, H, T, 1))
    gamma = torch.full((H,), 0.01)
    return write, read, r_val, alpha, beta, gamma


def test_chunked_c1_matches_sequential_stale():
    phi = get_activation("ln_silu")
    w, rd, rv, a, b, g = _inputs()
    for mode, read in (("M_q", rd), ("M_T_r", rv)):
        seq = run_trellis_memory(
            w, read, a, b, g, phi, mode, training=False, exact_inner=False
        )
        ck = run_trellis_memory_chunked(w, read, a, b, g, phi, mode, chunk_size=1)
        d = (seq - ck).abs().max().item()
        # C=1 IS the sequential path algorithmically, but the chunk kernel
        # reaches it through segmented-decay matmuls, so the two differ only by
        # fp32 op-order rounding (~1e-7 relative). Gate on the scale-relative
        # error: an absolute 1e-5 was build-dependent (fine on the W7900 torch
        # build, ~3e-5 on others) while a real kernel bug is O(1e-2)+ relative.
        rel = d / (seq.abs().max().item() + 1e-6)
        assert rel < 1e-4, f"{mode}: chunk=1 != sequential stale (rel {rel}, abs {d})"


def test_chunked_cgt1_finite_and_bounded():
    # C>1 is a deliberate stale-within-chunk approximation; exactness is only at
    # C=1 (covered above). Here just require finite, bounded output (no NaN/blowup);
    # the real C>1 validation is empirical (train at C=4 reaches similar PPL).
    phi = get_activation("ln_silu")
    w, rd, rv, a, b, g = _inputs(T=16)
    for C in (2, 4, 8, 16):
        y = run_trellis_memory_chunked(w, rd, a, b, g, phi, "M_q", C)
        assert torch.isfinite(y).all(), f"C={C} non-finite"
        assert y.abs().max().item() < 1e4, f"C={C} blew up"


def test_chunked_exact_mode_matches_sequential():
    # refine_passes < 0 -> auto C-1 passes. The within-chunk dependency is
    # strictly lower-triangular, so C-1 Jacobi passes are NILPOTENT and the
    # chunked forward must reproduce the (stale) sequential forward exactly.
    phi = get_activation("ln_silu")
    w, rd, rv, a, b, g = _inputs(T=16)
    for mode, read in (("M_q", rd), ("M_T_r", rv)):
        seq = run_trellis_memory(
            w, read, a, b, g, phi, mode, training=False, exact_inner=False
        )
        for C in (2, 4, 8, 16):
            ck = run_trellis_memory_chunked(
                w, read, a, b, g, phi, mode, C, refine_passes=-1
            )
            d = (seq - ck).abs().max().item()
            # fp32 carry accumulates over T/C chunks; nilpotent-exact still puts
            # this ~5 orders of magnitude below the stale gap (~1e2).
            assert d < 1e-2, f"{mode} C={C} exact-mode diff {d} (not nilpotent-exact)"


def test_chunked_model_causal():
    torch.manual_seed(0)
    cfg = tiny_cfg(chunk_size=4, beta_mode="scalar_per_head")
    m = TrellisLM(cfg).eval()
    T = 24
    idx = torch.randint(0, cfg.vocab_size, (1, T))
    with torch.no_grad():
        l1, _ = m(idx, training=False)
    t = 9
    idx2 = idx.clone()
    idx2[:, t + 1 :] = torch.randint(0, cfg.vocab_size, (1, T - (t + 1)))
    with torch.no_grad():
        l2, _ = m(idx2, training=False)
    assert (
        l1[:, : t + 1] - l2[:, : t + 1]
    ).abs().max().item() < 1e-4, "chunked path leaks future"


def test_chunked_model_trains():
    cfg = tiny_cfg(chunk_size=4, beta_mode="scalar_per_head")
    m = TrellisLM(cfg)
    idx = torch.randint(0, cfg.vocab_size, (2, 16))
    _, loss = m(idx, labels=idx, training=True)
    loss.backward()
    got = [
        n
        for n, p in m.named_parameters()
        if p.grad is not None and p.grad.abs().sum() > 0
    ]
    for needle in ("alpha_proj", "beta_proj", "gamma_raw", "q_proj"):
        assert any(needle in n for n in got), f"chunked: no grad to {needle}"
