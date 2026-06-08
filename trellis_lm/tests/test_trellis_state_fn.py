"""Phase-2 layer-1 grad-check: the explicit-backward TrellisStateEvolutionFn
must match autograd through the PyTorch state-evolution for all five inputs.

This is the correctness foundation the Triton/HIP fused kernels will implement.
"""

import torch

from trellis_lm.trellis_memory import (
    run_trellis_memory_chunked_state_evolution as evo,
    TrellisStateEvolutionFn,
    trellis_chunk_decay,
)
from trellis_lm.activations import ln_silu


def run():
    torch.set_default_dtype(torch.float64)
    torch.manual_seed(0)
    B, H, T, D, M, cs = 2, 3, 67, 8, 16, 16
    w0 = torch.randn(B, H, T, D)
    a0 = torch.randn(B, H, T, M)
    b = torch.sigmoid(torch.randn(B, H, T, 1))
    g0 = torch.rand(H) + 0.1
    P0, rmat0, _ = trellis_chunk_decay(b, cs)
    nC = P0.shape[2]
    c1 = torch.randn(B, H, nC, M, D)
    c2 = torch.randn(B, H, nC, cs, M)

    def grads(use_fn):
        w = w0.clone().requires_grad_()
        a = a0.clone().requires_grad_()
        P = P0.clone().requires_grad_()
        rm = rmat0.clone().requires_grad_()
        g = g0.clone().requires_grad_()
        if use_fn:
            M0s, us = TrellisStateEvolutionFn.apply(w, a, P, rm, g)
        else:
            M0s, us, _, _, _ = evo(w, a, None, g, ln_silu, cs, P=P, rmat=rm)
        ((M0s * c1).sum() + (us * c2).sum()).backward()
        return w.grad, a.grad, P.grad, rm.grad, g.grad

    ref = grads(False)
    fn = grads(True)
    ok = True
    for nm, r, f in zip(("write", "alpha", "P", "rmat", "gamma"), ref, fn):
        e = float((r - f).abs().max())
        ok &= e < 1e-9
        print("grad_%-6s %.2e %s" % (nm, e, "PASS" if e < 1e-9 else "FAIL"))
    print("ALL PASS" if ok else "SOME FAILED")
    return ok


if __name__ == "__main__":
    raise SystemExit(0 if run() else 1)
