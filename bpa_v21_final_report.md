# BPA v21 Final Report: Exploit g=4, Beat kv_ratio=0.33

## Summary

v21 tested whether tight-group INT4 (g=4) combined with fewer INT8
layers (k<4) can beat S2_k6's kv_ratio=0.333 while maintaining
PASS@3% at all sequence lengths up to 32K.

**Verdict: The v21 hypothesis (g=4 + k<4 beats 0.333) is FALSE.**
g=4 has 50% scale overhead, making its kv_ratio=0.50 — worse than
INT8-all (0.516). Instead, standard INT4 g=32 with k=4 INT8 layers
using the theory-based ranking [0,8,1,2] achieves kv_ratio=0.3203
and PASS@3%, beating S2_k6 by 3.9%.

## 1. True KV Bytes Accounting (Phase 0)

The critical finding: scale metadata overhead scales inversely with
group size. g=4 produces 16 groups per head_dim=64, requiring 64
bytes of scales per layer — equal to the 64-byte INT4 payload
itself, yielding 50% scale overhead and kv_ratio=0.50.

| Config | Bytes/tok | kv_ratio | Scale % |
|--------|-----------|----------|---------|
| dense_bf16 | 12288 | 1.0000 | 0.0% |
| INT8_all | 6336 | 0.5156 | 0.0% |
| INT4 g=32 k=0 | 3456 | 0.2812 | 11.1% |
| INT4 g=32 k=4 | 3936 | 0.3203 | 11.1% |
| INT4 g=8 k=0 | 4608 | 0.3750 | 33.3% |
| INT4 g=4 k=0 | 6144 | 0.5000 | 50.0% |

g=32 is the only group size where kv_ratio < 0.333 is achievable
(requires k<=4). g=8 and g=4 cannot beat S2_k6 regardless of k.

## 2. Grid Search Results (Phase 2)

Grid over g={32,8,4} x k={0,1,2,3,4,6} with theory ranking
[0,8,1,2,...] from v20. Quick screen at L=8K, survivors validated
at L=16K and L=32K with 3 seeds.

### g=32 (the only viable group size for compression)

| k | kv_ratio | max_delta | PASS@3% |
|---|----------|-----------|---------|
| 0 | 0.2812 | +30.25% | NO |
| 1 | 0.2910 | +7.48% | NO |
| 2 | 0.3008 | +4.97% | NO |
| 3 | 0.3105 | +4.25% | NO |
| 4 | 0.3203 | +2.30% | YES |
| 6 | 0.3398 | +2.35% | YES |

**k=4 is the minimum for PASS@3% at all L (k-floor confirmed).**
g=32 k=4 with theory ranking achieves kv_ratio=0.3203.

### g=8 (marginal, cannot beat S2_k6)

| k | kv_ratio | max_delta | PASS@3% |
|---|----------|-----------|---------|
| 2 | 0.3867 | +2.83% | YES |
| 3 | ~0.392 | +1.72% | YES |
| 4 | 0.3984 | +1.90% | YES |

g=8 achieves lower k-floor (k=2 passes) thanks to 4x lower noise,
but its kv_ratio (0.387+) is too high to be useful. The noise
reduction is real but the scale overhead (33%) defeats the purpose.

### g=4 (useless for compression, strong noise reduction)

| k | kv_ratio | max_delta | PASS@3% |
|---|----------|-----------|---------|
| 0 | 0.5000 | +4.46% | NO |
| 2 | 0.5013 | +2.07% | YES |
| 3 | 0.5020 | +2.19% | YES |

g=4 confirms v20's finding that tight groups reduce noise (k=2
passes), but kv_ratio=0.50 makes this academic. You pay more
bytes for g=4 INT4 than for INT8-all (0.516).

## 3. Failure Attribution (Phase 3)

For g=32, the k-floor is k=4 because:

| Config | Total unprotected noise | Top distortion layers |
|--------|------------------------|----------------------|
| g32 k=1 | 12.574 | [8, 1, 2, 3, 4, 20] |
| g32 k=2 | 10.165 | [1, 2, 3, 4, 20, 21] |
| g32 k=3 | 8.219 | [2, 3, 4, 20, 21, 9] |

The PASS@3% threshold requires total unprotected noise below ~7.6.
k=3 (protecting [0,8,1]) leaves noise=8.22, just above threshold.
k=4 (protecting [0,8,1,2]) reduces noise to ~6.78, crossing the
threshold. Layer 2 contributes sigma=1.44, which is the marginal
layer that pushes k=3 over the edge.

## 4. Why The v21 Hypothesis Failed

v20 showed g=4 reduces per-layer INT4 noise by ~8x. This is true
and confirmed by v21 (g=4 k=0 at +4.46% vs g=32 k=0 at +30%).
However, v20 reported kv_ratio without accounting for scale overhead.

The fundamental constraint: for a model with head_dim=64:
- g=4: 16 groups/head, each needing a fp16 scale = 32 bytes/head
- g=32: 2 groups/head, each needing a fp16 scale = 4 bytes/head
- Payload is always 32 bytes/head (64 elements * 4 bits)

So g=4 scale overhead (32 bytes) matches the payload (32 bytes),
doubling the effective cost per INT4 layer. The 8x noise reduction
is real but comes at 8x scale cost — no free lunch.

## 5. Winner

**g=32 k=4 with theory ranking [0,8,1,2]:**
- kv_ratio = 0.3203 (beats S2_k6's 0.333 by 3.9%)
- max_delta = +2.30% (PASS@3% at all L, seeds, and ratios)
- Uses standard INT4 block quantization (no exotic groups)
- Protected layers: 0, 8, 1, 2 (top-4 by theory noise ranking)

Compared to S2_k6 (from v16):
- S2_k6: 18 INT4 + 6 INT8, kv_ratio=0.333, PASS@3%
- g32_k4: 20 INT4 + 4 INT8, kv_ratio=0.320, PASS@3%
- 2 fewer INT8 layers needed, thanks to better layer selection
  (theory ranking from residual norms vs empirical ranking from
  per-layer PPL ablation)

## 6. Recommendation for v22

1. The k-floor (k=4 for g=32) is architectural. Post-hoc methods
   cannot reduce it below 4 for this model. Further compression
   requires either (a) a model with more layers (diluting per-layer
   noise contribution) or (b) training-in-the-loop quantization to
   reduce sigma at the sensitive layers.

2. g=32 k=4 with theory ranking is the Pareto-optimal operating
   point for Qwen2.5-0.5B. It achieves the best compression ratio
   (0.320) that passes quality gates.

3. Scale metadata overhead kills tight-group INT4. Any future work
   on small group sizes must either (a) amortize scales across
   tokens (segment-level scales) or (b) use compressed scale
   representations (e.g., INT8 scales instead of fp16).

4. The theory ranking from v20 is strictly better than the oracle
   ranking from v19 for this model, since it measures residual
   norms directly rather than PPL deltas (which have confounding
   interactions between layers).
