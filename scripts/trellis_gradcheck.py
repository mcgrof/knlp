#!/usr/bin/env python3
"""#2 from the cmcp validation plan: is our Trellis outer/meta-gradient correct,
and is any error ASYMMETRIC between phi=identity and phi=nonlinear?

cmcp's #1 risk: matching the inner VJP to 1e-7 is necessary but insufficient --
the OUTER backward through the memory update needs the Hessian of the inner loss
for nonlinear phi (constant Jacobian for identity), so a forward-perfect kernel
can have a custom backward that drops Hessian terms and selectively cripples the
nonlinear arm.

Method: a ground-truth pure-autograd unrolled reference (the inner step is
u = d/dz [1/2||phi(z)-alpha||^2] taken with create_graph so the FULL du/dz flows
to the outer gradient), compared against run_trellis_memory(exact_inner=True/False)
and the chunked state-evolution path. m != D (m=7, D=5) so transposes can't hide,
float64. If identity gradients match the reference but ln_silu's don't, that is
the asymmetric handicap.
"""
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from trellis_lm.activations import ln_silu, identity
from trellis_lm.trellis_memory import run_trellis_memory


def ref_memory(write, read, alpha, beta, gamma, phi, read_mode):
    """Exact unrolled recurrence; u carries the full du/dz (true meta-gradient)."""
    B, H, T, D = write.shape
    M = alpha.shape[-1]
    Ms = write.new_zeros(B, H, M, D)
    g = gamma.view(1, H, 1, 1)
    outs = []
    for t in range(T):
        w = write[:, :, t]            # [B,H,D]
        a = alpha[:, :, t]            # [B,H,M]
        b = beta[:, :, t]             # [B,H,1]
        z = torch.einsum("bhmd,bhd->bhm", Ms, w)            # in graph
        inner = 0.5 * ((phi(z) - a) ** 2).sum()
        (u,) = torch.autograd.grad(inner, z, create_graph=True)   # full du/dz
        outer = torch.einsum("bhm,bhd->bhmd", u, w)
        Ms = b.unsqueeze(-1) * Ms - g * outer
        r = read[:, :, t]
        if read_mode == "M_q":
            y = torch.einsum("bhmd,bhd->bhm", Ms, r)        # [B,H,M]
        else:
            y = torch.einsum("bhmd,bhm->bhd", Ms, r)        # [B,H,D]
        outs.append(y)
    return torch.stack(outs, dim=2)


def compare(phi, read_mode, exact_inner):
    torch.manual_seed(0)
    B, H, T, D, M = 2, 3, 4, 5, 7        # m=7 != D=5
    dt = torch.float64
    rd = D if read_mode == "M_q" else M
    write = torch.randn(B, H, T, D, dtype=dt, requires_grad=True)
    read = torch.randn(B, H, T, rd, dtype=dt, requires_grad=True)
    alpha = torch.randn(B, H, T, M, dtype=dt, requires_grad=True)
    beta = torch.rand(B, H, T, 1, dtype=dt, requires_grad=True)
    gamma = (torch.rand(H, dtype=dt) * 0.05 + 0.01).requires_grad_(True)
    ins = [write, read, alpha, beta, gamma]

    yref = ref_memory(write, read, alpha, beta, gamma, phi, read_mode)
    gref = torch.autograd.grad(yref.sum(), ins, retain_graph=True, allow_unused=True)
    yours = run_trellis_memory(write, read, alpha, beta, gamma, phi, read_mode,
                               training=True, exact_inner=exact_inner)
    gours = torch.autograd.grad(yours.sum(), ins, allow_unused=True)

    mode = "exact" if exact_inner else "stale"
    fwd = (yref - yours).abs().max().item()
    print(f"--- phi={phi.__name__:8s} {read_mode:5s} inner={mode:5s} | "
          f"fwd max|ref-ours|={fwd:.2e}")
    worst = 0.0
    for n, gr, go in zip(["write", "read", "alpha", "beta", "gamma"], gref, gours):
        if gr is None or go is None:
            print(f"      grad {n:6s}: None"); continue
        e = (gr - go).abs().max().item()
        rel = e / (gr.abs().max().item() + 1e-15)
        worst = max(worst, rel)
        flag = "  <-- MISMATCH" if rel > 1e-6 else ""
        print(f"      grad {n:6s}: max|diff|={e:.2e}  rel={rel:.2e}{flag}")
    return worst


def main():
    print("=== Trellis meta-gradient correctness (ground-truth autograd ref) ===")
    print("m=7 != D=5, float64, T=4. rel>1e-6 on any grad = outer backward wrong.\n")
    res = {}
    for phi in (identity, ln_silu):
        for rm in ("M_q", "M_T_r"):
            print(f"### phi={phi.__name__} read_mode={rm}")
            res[(phi.__name__, rm, "exact")] = compare(phi, rm, True)
            res[(phi.__name__, rm, "stale")] = compare(phi, rm, False)
            print()
    print("=== SUMMARY (worst relative grad error) ===")
    for k, v in res.items():
        print(f"  {k}: {v:.2e}{'  ASYMMETRIC/BUG' if v > 1e-6 else '  ok'}")
    # the diagnostic: does exact-mode match for identity but not ln_silu?
    id_ex = max(res[("identity", rm, "exact")] for rm in ("M_q", "M_T_r"))
    ln_ex = max(res[("ln_silu", rm, "exact")] for rm in ("M_q", "M_T_r"))
    print(f"\n  exact-mode: identity worst={id_ex:.2e}  ln_silu worst={ln_ex:.2e}")
    if id_ex < 1e-6 and ln_ex > 1e-6:
        print("  => CONFIRMED asymmetric: exact backward correct for identity, "
              "WRONG for nonlinear (drops Hessian) -> handicaps the nonlinear arm.")
    elif id_ex < 1e-6 and ln_ex < 1e-6:
        print("  => exact backward correct for BOTH -> the 2x gap is not a "
              "meta-gradient bug; look at fidelity (shell/phi/gamma) next.")
    else:
        print("  => exact backward wrong for identity too -> different issue.")


if __name__ == "__main__":
    main()
