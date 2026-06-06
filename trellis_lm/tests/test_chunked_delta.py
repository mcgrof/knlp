"""Exactness test: chunked delta rule must match the naive sequential loop."""

import torch

from trellis_lm.linear_baselines_chunked import chunked_delta_rule


def naive(q, k, v, beta, alpha=None):
    B, H, T, D = q.shape
    S = torch.zeros(B, H, D, D)
    outs = []
    for t in range(T):
        kt, vt, qt = k[:, :, t], v[:, :, t], q[:, :, t]
        bt = beta[:, :, t][..., None, None]
        if alpha is not None:
            S = alpha[:, :, t][..., None, None] * S
        pred = torch.einsum("bhij,bhj->bhi", S, kt)
        S = S + bt * torch.einsum("bhi,bhj->bhij", vt - pred, kt)
        outs.append(torch.einsum("bhij,bhj->bhi", S, qt))
    return torch.stack(outs, dim=2)


def run():
    torch.manual_seed(0)
    B, H, T, D = 2, 3, 40, 8
    q = torch.randn(B, H, T, D)
    k = torch.nn.functional.normalize(torch.randn(B, H, T, D), dim=-1)
    v = torch.randn(B, H, T, D)
    beta = torch.sigmoid(torch.randn(B, H, T))
    ok = True
    for gated in (False, True):
        alpha = torch.sigmoid(torch.randn(B, H, T)) if gated else None
        ref = naive(q, k, v, beta, alpha)
        got = chunked_delta_rule(q, k, v, beta, alpha, chunk_size=16)
        err = (ref - got).abs().max().item()
        verdict = "PASS" if err < 1e-3 else "FAIL"
        if err >= 1e-3:
            ok = False
        print(f"gated={gated}  max_abs_err={err:.2e}  {verdict}")
    print("ALL PASS" if ok else "SOME FAILED")
    return ok


if __name__ == "__main__":
    raise SystemExit(0 if run() else 1)
