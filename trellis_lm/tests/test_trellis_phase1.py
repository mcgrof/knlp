"""fp64 A/B test: the Phase-1 batched operator must match the loop operator.

run_trellis_memory_chunked_phase1 (state-evolution + batched-over-chunks readout)
must be byte-equivalent to run_trellis_memory_chunked(refine_passes=0) -- the
refactor is a pure speedup (~2.7x), not a math change.
"""

import torch

from trellis_lm.trellis_memory import (
    run_trellis_memory_chunked,
    run_trellis_memory_chunked_phase1,
    run_trellis_memory_chunked_state_evolution as _evo,
)
from trellis_lm.activations import ln_silu


def run():
    torch.set_default_dtype(torch.float64)
    torch.manual_seed(0)
    B, H, T, D, M = 2, 3, 67, 8, 16
    w = torch.randn(B, H, T, D)
    rq = torch.randn(B, H, T, D)
    rv = torch.randn(B, H, T, M)
    a = torch.randn(B, H, T, M)
    b = torch.sigmoid(torch.randn(B, H, T, 1))
    g = torch.rand(H) + 0.1
    ok = True
    for mode, rd in (("M_q", rq), ("M_T_r", rv)):
        cur = run_trellis_memory_chunked(w, rd, a, b, g, ln_silu, mode, 16, 0)
        new = run_trellis_memory_chunked_phase1(w, rd, a, b, g, ln_silu, mode, 16)
        err = float((cur - new).abs().max())
        ok &= err < 1e-9
        print(f"{mode:6s} phase1-vs-loop {err:.2e} {'PASS' if err < 1e-9 else 'FAIL'}")
    # stacking the two passes on a 2x batch axis must equal separate evolutions
    v = torch.randn(B, H, T, D)
    w2 = torch.stack((w, v), 0).reshape(2 * B, H, T, D)
    a2 = a.unsqueeze(0).expand(2, B, H, T, M).reshape(2 * B, H, T, M)
    b2 = b.unsqueeze(0).expand(2, B, H, T, 1).reshape(2 * B, H, T, 1)
    M0s2, us2, _, _, _ = _evo(w2, a2, b2, g, ln_silu, 16)
    nC = M0s2.shape[2]
    M0s2 = M0s2.view(2, B, H, nC, M, D)
    M0k, uk, _, _, _ = _evo(w, a, b, g, ln_silu, 16)
    e = float((M0s2[0] - M0k).abs().max())
    ok &= e < 1e-12
    print(f"stack  vs-separate {e:.2e} {'PASS' if e < 1e-12 else 'FAIL'}")
    print("ALL PASS" if ok else "SOME FAILED")
    return ok


if __name__ == "__main__":
    raise SystemExit(0 if run() else 1)
