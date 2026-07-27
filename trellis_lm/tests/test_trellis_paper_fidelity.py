"""Paper-fidelity oracle tests for the Trellis memory operator (arXiv:2512.23852).

These assert the paper's stated properties directly, as a single labelled suite.
Three of them reinforce existing coverage (chunk C=1 == sequential and the
exact-mode chunk oracle in test_trellis_chunked.py; write-before-read in
test_trellis_causality.py); the phi=identity => gated-delta-rule reduction and
the phi/f decoupling are new. See reports/trellis_paper_fidelity.md for how each
maps onto the paper and where the paper is under-specified.
"""

import torch

from trellis_lm.config import TrellisConfig
from trellis_lm.tests._helpers import tiny_cfg
from trellis_lm.trellis_memory import run_trellis_memory, run_trellis_memory_chunked
from trellis_lm.activations import get_activation, identity, ln_silu


def _inputs(B=2, H=2, T=12, D=8, M=4, seed=0):
    g = torch.Generator().manual_seed(seed)
    mk = lambda *s: torch.randn(*s, generator=g)
    write = mk(B, H, T, D)
    read = mk(B, H, T, D)
    alpha = mk(B, H, T, M)
    beta = torch.sigmoid(mk(B, H, T, 1))
    gamma = torch.full((H,), 0.01)  # match the chunk-test regime (fp32-tight)
    return write, read, alpha, beta, gamma


def test_A_chunk_c1_equals_sequential():
    """C=1 chunk kernel reproduces the sequential stale recurrence exactly."""
    w, read, a, b, g = _inputs()
    for mode, rd in (("M_q", read), ("M_T_r", torch.randn(2, 2, 12, 4))):
        seq = run_trellis_memory(
            w, rd, a, b, g, ln_silu, mode, training=False, exact_inner=False
        )
        ck = run_trellis_memory_chunked(w, rd, a, b, g, ln_silu, mode, chunk_size=1)
        assert (seq - ck).abs().max().item() < 1e-5, mode


def test_B_chunk_exact_mode_is_serial_oracle():
    """The chunk kernel with the auto C-1 refine passes (nilpotent-exact) equals
    the serial recurrence -- the serial path is the oracle, per the paper's own
    caution that the printed chunk equation is not to be trusted typographically.
    """
    w, read, a, b, g = _inputs(T=12)
    seq = run_trellis_memory(
        w, read, a, b, g, ln_silu, "M_q", training=False, exact_inner=False
    )
    for C in (2, 3, 6):
        ck = run_trellis_memory_chunked(
            w, read, a, b, g, ln_silu, "M_q", chunk_size=C, refine_passes=-1
        )
        assert (seq - ck).abs().max().item() < 1e-2, C


def test_C_phi_identity_is_gated_delta_rule():
    """With phi = identity the write u_t = phi(z)-alpha = M w - alpha, so the
    update reduces to the gated delta rule M <- beta M - gamma (M w - alpha) w^T.
    Compare the operator to an independent closed-form delta-rule loop."""
    w, read, a, b, g = _inputs()
    B, H, T, D = w.shape
    M = a.shape[-1]
    y_op = run_trellis_memory(
        w, read, a, b, g, identity, "M_q", training=False, exact_inner=False
    )
    Mst = torch.zeros(B, H, M, D)
    ys = []
    for t in range(T):
        wt, at, bt, rt = w[:, :, t], a[:, :, t], b[:, :, t], read[:, :, t]
        z = torch.einsum("bhmd,bhd->bhm", Mst, wt)  # pre-update readout code
        u = z - at  # identity Jacobian: VJP is the residual
        outer = torch.einsum("bhm,bhd->bhmd", u, wt)
        Mst = bt.unsqueeze(-1) * Mst - g.view(1, H, 1, 1) * outer
        ys.append(torch.einsum("bhmd,bhd->bhm", Mst, rt))  # write-before-read
    y_ref = torch.stack(ys, dim=2)
    assert (y_op - y_ref).abs().max().item() < 1e-4


def test_D_write_before_read_includes_current_token():
    """Readout at t reflects the write at t (Omega[i,i]=1, write-before-read) and
    nothing earlier: perturbing write_t changes y_t but leaves y_{t-1} intact."""
    w, read, a, b, g = _inputs()
    t = w.shape[2] // 2
    y0 = run_trellis_memory(
        w, read, a, b, g, ln_silu, "M_q", training=False, exact_inner=False
    )
    w2 = w.clone()
    w2[:, :, t] = w2[:, :, t] + 0.5
    y1 = run_trellis_memory(
        w2, read, a, b, g, ln_silu, "M_q", training=False, exact_inner=False
    )
    assert (y1[:, :, t] - y0[:, :, t]).abs().max().item() > 1e-4  # own write in
    assert (y1[:, :, t - 1] - y0[:, :, t - 1]).abs().max().item() < 1e-6  # causal


def test_E_phi_and_f_decouple():
    """phi (write) and f (inter-pass) must be independently configurable; None
    ties both to `activation` (backward compatible). Mirrors the exact fallback
    the mixer uses."""
    x = torch.randn(3, 8)
    c = tiny_cfg(activation="ln_silu")
    phi = get_activation(c.phi_activation or c.activation)
    f = get_activation(c.f_activation or c.activation)
    assert torch.allclose(phi(x), f(x))  # tied by default

    c2 = tiny_cfg(
        activation="ln_silu", phi_activation="identity", f_activation="ln_silu"
    )
    phi2 = get_activation(c2.phi_activation or c2.activation)
    f2 = get_activation(c2.f_activation or c2.activation)
    assert not torch.allclose(phi2(x), f2(x))  # phi=identity != f=ln_silu

    fb = TrellisConfig.faithful_baseline()
    assert fb.f_activation == "ln_silu"
    assert fb.value_readout_act == "ln_silu"
    assert fb.output_path == "paper"
    assert fb.trellis_beta_min < 0.9  # forget gate not floored near 1
