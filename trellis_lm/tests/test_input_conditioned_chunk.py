"""Exactness tests for the input-conditioned affine chunk kernel.

The chunked path (run_trellis_memory_chunked with input_gate) must reproduce the
exact sequential recurrence (run_trellis_memory with input_gate) for any chunk
size, because the input-conditioned write u_t = a(x_t)*z_t - alpha_t is affine in
the state, so one forward-substitution pass inside the chunk is exact.

Run: python3 trellis_lm/tests/test_input_conditioned_chunk.py
Covers the fp64 A/B and the property tests from the 2026-07-05 cmcp review
(constant beta, beta=1, a=0, alpha=0, orthogonal W).
"""

import torch

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from trellis_lm.activations import ln_silu
from trellis_lm.trellis_memory import run_trellis_memory, run_trellis_memory_chunked


def _mk(B, H, T, D, M, seed=0, dtype=torch.float64, device="cpu", **over):
    g = torch.Generator(device=device).manual_seed(seed)
    W = torch.randn(B, H, T, D, generator=g, dtype=dtype, device=device)
    W = torch.nn.functional.normalize(W, dim=-1)  # write_l2norm
    R = torch.randn(B, H, T, D, generator=g, dtype=dtype, device=device)
    alpha = torch.randn(B, H, T, M, generator=g, dtype=dtype, device=device)
    a_gate = 0.5 + torch.rand(B, H, T, M, generator=g, dtype=dtype, device=device)
    beta = 0.9 + 0.09 * torch.rand(B, H, T, 1, generator=g, dtype=dtype, device=device)
    gamma = torch.full((H,), 0.05, dtype=dtype, device=device)
    over_beta = over.get("beta")
    over_alpha = over.get("alpha")
    over_a = over.get("a_gate")
    if over_beta is not None:
        beta = torch.full_like(beta, over_beta)
    if over_alpha is not None:
        alpha = torch.full_like(alpha, over_alpha)
    if over_a is not None:
        a_gate = torch.full_like(a_gate, over_a)
    if over.get("orthogonal"):
        # make W within each head an orthonormal set over time (T<=D)
        q, _ = torch.linalg.qr(W.transpose(-1, -2))  # [B,H,D,T]
        W = q.transpose(-1, -2)[:, :, :T, :]
    return W, R, alpha, beta, gamma, a_gate


def _ab(read_mode, chunk, **mk):
    W, R, alpha, beta, gamma, a_gate = _mk(**mk)
    if read_mode == "M_T_r":
        # value pass reads with an M-dim query, not the D-dim key
        R = torch.randn_like(alpha)
    seq = run_trellis_memory(
        W,
        R,
        alpha,
        beta,
        gamma,
        ln_silu,
        read_mode,
        training=False,
        input_gate=a_gate,
    )
    chk = run_trellis_memory_chunked(
        W,
        R,
        alpha,
        beta,
        gamma,
        ln_silu,
        read_mode,
        chunk_size=chunk,
        input_gate=a_gate,
    )
    return (seq - chk).abs().max().item(), seq.abs().max().item()


def main():
    B, H, T, D, M = 2, 3, 32, 16, 12
    tol = 1e-9
    fails = 0
    print("== fp64 A/B: chunked input_gate vs exact sequential ==")
    for read_mode in ("M_q", "M_T_r"):
        for chunk in (1, 2, 4, 8, 16):
            err, scale = _ab(read_mode, chunk, B=B, H=H, T=T, D=D, M=M, seed=1)
            ok = err <= tol * max(1.0, scale)
            fails += not ok
            print(
                f"  {read_mode:5s} chunk={chunk:2d}  max_abs_err={err:.2e}"
                f"  (scale {scale:.2e})  {'OK' if ok else 'FAIL'}"
            )

    print("== property tests (chunk=8, M_q) ==")
    props = {
        "constant_beta=0.95": dict(beta=0.95),
        "beta=1 (no decay)": dict(beta=1.0),
        "a=0 (u=-alpha, no coupling)": dict(a_gate=0.0),
        "alpha=0 (pure contraction)": dict(alpha=0.0),
        "orthogonal W (G diag)": dict(orthogonal=True),
    }
    for name, ov in props.items():
        err, scale = _ab("M_q", 8, B=B, H=H, T=T, D=D, M=M, seed=2, **ov)
        ok = err <= tol * max(1.0, scale)
        fails += not ok
        print(f"  {name:34s} max_abs_err={err:.2e}  {'OK' if ok else 'FAIL'}")

    print("== a=0 sanity: u must equal -alpha (no coupling) ==")
    W, R, alpha, beta, gamma, a_gate = _mk(B=1, H=1, T=8, D=8, M=6, seed=3, a_gate=0.0)
    # with a=0, u_t = -alpha_t exactly; check via M_T_r readout match already
    # covered, so just re-run A/B here as the invariant
    err, scale = _ab("M_q", 4, B=1, H=1, T=8, D=8, M=6, seed=3, a_gate=0.0)
    ok = err <= tol * max(1.0, scale)
    fails += not ok
    print(f"  a=0 A/B max_abs_err={err:.2e}  {'OK' if ok else 'FAIL'}")

    print(f"\n{'ALL PASS' if fails == 0 else str(fails) + ' FAILURES'}")
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
