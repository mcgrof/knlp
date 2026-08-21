#!/usr/bin/env python3
"""Standalone algebra check for pre-bias FP8 key caching.

Models with a key-projection bias compute K_t = RoPE_t(x_t W_K + b).
Quantizing that finished K to FP8 with one per-tensor scale fails when
the bias dominates the scale (the FP8 failure-atlas mechanism). The
repair stores only the token-varying part in FP8 and keeps the bias
exact. Two representations are checked here:

  reconstruct: cache RoPE_t(x_t W_K) in FP8; at read time dequantize
      and add RoPE_t(b), which is computable from b and the rotary
      tables alone.
  score-correction: never rebuild K. Because RoPE_t is linear,
      q . RoPE_t(k_pre + b) = q . RoPE_t(k_pre) + q . RoPE_t(b),
      and the second term collapses per rotary frequency f to
      A_f(q) cos(t theta_f) + B_f(q) sin(t theta_f), where A and B
      depend only on q and b. Cost per query: one O(d) prep, then an
      O(d/2) dot against the cos/sin tables per key position.

Checks, in order:
  1. RoPE linearity, float64, exact.
  2. Score identity for both representations, float64, exact.
  3. FP8 end-to-end: post-RoPE-residual FP8 caching with the exact
     bias correction versus naive full-K FP8, under a Qwen-like
     bias whose magnitude dwarfs the residual. The repair must
     roughly match the no-bias FP8 error; the naive path must be
     far worse.

Pure CPU torch. No model download; the bias is synthetic with
atlas-like statistics (bias norm >> residual norm). Real-weight
validation belongs to the serving battery.
"""

import torch

torch.manual_seed(7)

D = 128  # head dim, full rotary
T = 512  # key positions
E4M3_MAX = 448.0


def rope_tables(T, D, base=1000000.0, dtype=torch.float64):
    freqs = base ** (-torch.arange(0, D, 2, dtype=dtype) / D)
    t = torch.arange(T, dtype=dtype)
    ang = torch.outer(t, freqs)  # [T, D/2]
    return torch.cos(ang), torch.sin(ang)


def rope_apply(x, cos, sin):
    """x: [..., T, D] with interleaved even/odd pairs."""
    xe, xo = x[..., 0::2], x[..., 1::2]
    ye = xe * cos - xo * sin
    yo = xe * sin + xo * cos
    y = torch.empty_like(x)
    y[..., 0::2], y[..., 1::2] = ye, yo
    return y


def quantize_e4m3(t):
    amax = t.abs().amax().clamp(min=1e-12)
    scale = (amax / E4M3_MAX).double()
    q = (t / scale).clamp(-E4M3_MAX, E4M3_MAX).to(torch.float8_e4m3fn)
    return q.double() * scale  # dequantized view


def main():
    cos, sin = rope_tables(T, D)
    k_pre = torch.randn(T, D, dtype=torch.float64)  # x_t W_K residuals
    q = torch.randn(D, dtype=torch.float64)
    # Qwen-like bias: a few huge channels, norm far above the residual
    bias = torch.randn(D, dtype=torch.float64)
    bias[torch.randperm(D)[:8]] += 40.0 * torch.sign(torch.randn(8).double())
    print(
        f"bias norm {bias.norm():.1f} vs mean residual row norm "
        f"{k_pre.norm(dim=1).mean():.1f}"
    )

    # 1. RoPE linearity, exact
    lhs = rope_apply(k_pre + bias, cos, sin)
    rhs = rope_apply(k_pre, cos, sin) + rope_apply(bias.expand(T, D), cos, sin)
    err = (lhs - rhs).abs().max().item()
    print(f"1. RoPE linearity max err        {err:.3e}")
    assert err < 1e-12

    # 2a. reconstruct representation, exact
    scores_ref = lhs @ q  # [T]
    k_resid_rot = rope_apply(k_pre, cos, sin)
    bias_rot = rope_apply(bias.expand(T, D), cos, sin)
    scores_reco = (k_resid_rot + bias_rot) @ q
    err = (scores_ref - scores_reco).abs().max().item()
    print(f"2a. reconstruct identity max err {err:.3e}")
    assert err < 1e-9

    # 2b. score-correction representation, exact.
    # q . RoPE_t(b) expands per rotary pair to
    #   (q_e b_e + q_o b_o) cos + (q_o b_e - q_e b_o) sin
    qe, qo = q[0::2], q[1::2]
    be, bo = bias[0::2], bias[1::2]
    A = qe * be + qo * bo  # [D/2]
    B = qo * be - qe * bo
    correction = cos @ A + sin @ B  # [T]
    scores_corr = k_resid_rot @ q + correction
    err = (scores_ref - scores_corr).abs().max().item()
    print(f"2b. score-corr identity max err  {err:.3e}")
    assert err < 1e-9

    # 3. FP8 accuracy: naive full-K vs pre-bias residual + exact bias.
    # Compare absolute score errors: the repaired path quantizes the
    # same tensor as the biasless floor case and adds an exact term,
    # so its absolute error must equal the floor by construction.
    naive_err = (quantize_e4m3(lhs) @ q - scores_ref).abs().mean().item()
    repaired = quantize_e4m3(k_resid_rot) @ q + correction
    repaired_err = (repaired - scores_ref).abs().mean().item()
    floor_err = (quantize_e4m3(k_resid_rot) @ q - k_resid_rot @ q).abs().mean().item()
    print(f"3. mean abs score error: naive full-K fp8   {naive_err:.4f}")
    print(f"   pre-bias residual fp8 + exact correction {repaired_err:.4f}")
    print(f"   fp8 error floor (no bias in play)        {floor_err:.4f}")
    assert abs(repaired_err - floor_err) < 1e-9, "repair must equal the floor"
    assert naive_err > 5 * repaired_err, "naive should be far worse"

    print("\nALL CHECKS PASSED")
    print(
        "The score-correction path needs no K reconstruction: per query, "
        "O(d) prep for A and B, then an O(d/2) dot with the rotary "
        "tables per key position. Partial-rotary models split the same "
        "way: the rotated dims use the correction, the pass-through "
        "dims add q . b directly."
    )


if __name__ == "__main__":
    main()
