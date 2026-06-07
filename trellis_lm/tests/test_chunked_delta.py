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
    B, H, T, D = 2, 3, 70, 8
    q = torch.randn(B, H, T, D)
    k = torch.nn.functional.normalize(torch.randn(B, H, T, D), dim=-1)
    v = torch.randn(B, H, T, D)
    beta = torch.sigmoid(torch.randn(B, H, T))
    # cases: plain, gated (normal gate), gated-small (gate driven small — the
    # regime that overflowed the 1/a rescale and diverged training).
    cases = [
        ("plain", None),
        ("gated", torch.sigmoid(torch.randn(B, H, T))),
        ("gated-small", torch.sigmoid(torch.randn(B, H, T) - 4.0)),
    ]
    ok = True
    for cs in (16, 32):
        for name, alpha in cases:
            ref = naive(q, k, v, beta, alpha)
            got = chunked_delta_rule(q, k, v, beta, alpha, chunk_size=cs)
            err = (ref - got).abs().max().item()
            finite = torch.isfinite(got).all().item()
            verdict = "PASS" if (err < 1e-3 and finite) else "FAIL"
            if verdict == "FAIL":
                ok = False
            print(
                f"chunk{cs:<3} {name:12s} max_abs_err={err:.2e} finite={finite}  {verdict}"
            )
    print("ALL PASS" if ok else "SOME FAILED")
    return ok


if __name__ == "__main__":
    raise SystemExit(0 if run() else 1)
