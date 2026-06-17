"""Cumulative decode-latency assessment for the fdec bricks, added one at a time.

Decode is memory-bandwidth-bound (paper-memory-decode): per generated token the GPU
reads the model weights once, the whole KV cache, and the LM-head, so the roofline
latency is (decode bytes per token) / (HBM bandwidth). This is a byte-bound PROJECTION,
not measured wall-clock -- the real kernel speedup is the CUDA serving-stack dimension.
We add the bricks cumulatively and show the effect at several context lengths, because
which pool dominates shifts with context.

Bricks (deployable numbers, corrected across this thread):
  weights : int8 (+GPTQ), factor 0.508 (8 bits + group scale), ~near-lossless
  KV      : pre-bias K8/V8 (both 8-bit), factor 0.50 -- vs the older K16/V8 at 0.75
  LM-head : idblock certified-decode -- byte-NEUTRAL out-of-sample on Qwen (the
            in-sample ~0.25x does not generalize), so 1.0 deployable
"""

MODEL = "Qwen2.5-7B"
# per-token decode byte pools at ctx 32768, bf16 (from fdec TABLE.md, verified)
W_MB = 14141.0
KV_MB_32K = 1879.0
LMH_MB = 1090.0
HBM_GBPS = 3350.0  # H100 HBM

# brick factors (deployable)
F_W_INT8 = 0.508
F_KV_K16V8 = 0.75  # keys fp16 + values fp8
F_KV_K8V8 = 0.50  # pre-bias K8/V8
F_LMH_IDBLOCK = 1.0  # OOS byte-neutral

# measured quality (Qwen2.5-7B, this thread); composed quality confirmed separately
Q = {
    "weights int8": "top-1 0.921 / ppl x1.027 (int8); +GPTQ tightens KL",
    "KV K8/V8 pre-bias": "ppl x1.008 / KL 0.021 (FP8 pre-bias residual)",
    "LM-head idblock": "lossless (argmax), but byte-neutral OOS",
}


def kv_mb(ctx):
    return KV_MB_32K * ctx / 32768.0


def latency_ms(total_mb):
    return total_mb / HBM_GBPS  # MB / (GB/s) = ms


def assess(ctx):
    w, kv, lmh = W_MB, kv_mb(ctx), LMH_MB
    base = w + kv + lmh
    wi = w * F_W_INT8
    # the two KV rows are ALTERNATIVES (pick one), shown for comparison
    rows = [
        ("baseline (bf16)", w, kv, lmh),
        ("+ weights int8", wi, kv, lmh),
        ("+ KV K16/V8 (alt)", wi, kv * F_KV_K16V8, lmh),
        ("+ KV K8/V8 pre-bias (alt)", wi, kv * F_KV_K8V8, lmh),
        ("+ LM-head idblock", wi, kv * F_KV_K8V8, lmh * F_LMH_IDBLOCK),
    ]
    print(f"\n=== {MODEL}  ctx={ctx}  (KV pool = {kv:,.0f} MB) ===")
    print(
        f"{'cumulative brick':<27}{'W':>8}{'KV':>8}{'LMh':>7}"
        f"{'total MB':>10}{'lat ms':>8}{'vs base':>9}"
    )
    print("-" * 77)
    for name, a, b, c in rows:
        tot = a + b + c
        print(
            f"{name:<27}{a:>8.0f}{b:>8.0f}{c:>7.0f}{tot:>10.0f}"
            f"{latency_ms(tot):>8.2f}{(tot/base-1)*100:>8.1f}%"
        )
    print(
        "  (the two KV rows are alternatives; pre-bias K8/V8 saves an extra "
        f"{kv*F_KV_K16V8 - kv*F_KV_K8V8:,.0f} MB of KV vs K16/V8 here)"
    )


def main():
    print("Decode-byte roofline latency, bricks added cumulatively.")
    print("Quality (measured separately, Qwen2.5-7B):")
    for k, v in Q.items():
        print(f"  {k:<20} {v}")
    for ctx in (4096, 32768, 131072):
        assess(ctx)
    # context sweep: roofline ms (and -%) per ctx, with the K16/V8 column added
    print("\n=== context sweep (roofline ms, % vs baseline) ===")
    print(
        f"{'ctx':>8}{'baseline':>11}{'+w int8':>13}{'+KV K16/V8':>15}"
        f"{'+KV K8/V8':>15}"
    )
    print("-" * 62)
    for ctx in (4096, 32768, 131072):
        kv = kv_mb(ctx)
        base = W_MB + kv + LMH_MB
        wi = W_MB * F_W_INT8 + kv + LMH_MB
        k16 = W_MB * F_W_INT8 + kv * F_KV_K16V8 + LMH_MB
        k8 = W_MB * F_W_INT8 + kv * F_KV_K8V8 + LMH_MB

        def cell(x):
            return f"{latency_ms(x):.2f} ({(x/base-1)*100:+.0f}%)"

        print(
            f"{ctx:>8}{latency_ms(base):>10.2f} {cell(wi):>12}{cell(k16):>15}"
            f"{cell(k8):>15}"
        )
    print(
        "\nNote: roofline = bytes/HBM, a memory-bound projection. Weights dominate at "
        "short ctx (int8 is the big lever); KV grows with ctx and pre-bias K8/V8 "
        "overtakes K16/V8 as the win there; LM-head idblock is byte-neutral OOS."
    )


if __name__ == "__main__":
    main()
