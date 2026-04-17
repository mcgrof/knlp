# Fused INT4 v0.0.2 — Latency Review (2026-04-04)

Retrospective latency review of fused INT4 v0.0.2 on H100. Records
the measured post-CUDA-graph-fix gap vs FP16, identifies the
sources of the residual ~2.8% ITL delta, and proposes a set of
kernel- and system-level follow-up ideas ranked by expected gain.
Paired with fused-int4-v0.0.2-20260404.md (which documents the
CUDA graph fix itself).


### Measured state after CUDA graph fix

| Cell | ITL (ms) | TPS | Graph modes |
|------|----------|-----|-------------|
| FP16 baseline | 6.47 | 153.3 | FULL+PIECEWISE |
| Fused INT4 | 6.65 | 149.3 | FULL+PIECEWISE |
| **Delta** | **+0.18 ms (2.8%)** | **-2.6%** | |

Prior to v0.0.2 the fused backend had `AttentionCGSupport.NEVER`, restricting it to PIECEWISE-only graphs while FP16 used FULL. That alone accounted for ~1.3 ms. Restoring `UNIFORM_SINGLE_TOKEN_DECODE` closed ~86% of the gap.

---

### 1. Remaining latency sources (the ~0.18 ms / 2.8%)

**A. Per-tile scale broadcast loop in the decode kernel.** The K-side scale expansion (lines 772-778) uses a `tl.static_range(NUM_GROUPS)` loop that constructs a float32 `g_mask` vector and accumulates via `+=` on every iteration. For Qwen2.5-7B with head_dim=128 and GROUP_SIZE=32, that's 4 iterations of broadcast-mask-accumulate per tile, for both K *and* V. This is pure ALU overhead that FlashAttention's FP16 path doesn't have. The `USE_GATHERED_SCALE` path (H3, line 826-831) only applies to V, not K. This loop is likely the single largest contributor to the residual gap.

**B. Nibble unpacking + signed offset arithmetic.** Every K/V load requires `& 0x0F`, `>> 4`, `- 8`, `.to(tl.float32)`, then multiply by scale. That's ~8 FP32 ops per packed byte that the FP16 path doesn't do at all. At decode-256 sequence lengths on H100, the kernel is not memory-bound enough to fully hide this compute behind memory latency.

**C. Even/odd Q split and dual accumulator.** The fused kernel splits Q into `q_even`/`q_odd` and carries two accumulators (`acc_even`, `acc_odd`) with two `tl.sum()` reductions per tile iteration. FlashAttention uses a single contiguous vector dot. The dual accumulator doubles the online-softmax rescale cost per tile (`acc * alpha` happens for both even and odd).

**D. Cache-write kernel on the decode path.** Every decode step runs `reshape_and_cache_v_only_int4` (or `reshape_and_cache_int4` for k_precision=int4) as a separate kernel launch before the decode attention kernel. This is an extra kernel that the FP16 path doesn't have — its cache write is handled by `reshape_and_cache_flash` in `do_kv_cache_update`, but the fused path additionally quantizes V. With CUDA graphs this is captured, so launch overhead is zero, but the kernel itself has nonzero execution time.

---

### 2. Kernel-level and system-level ideas

**Kernel-level:**

1. **Gathered scale for K (not just V).** The `USE_GATHERED_SCALE` code path already exists for V (H3, line 826) — it replaces the `static_range` loop with a single indexed `tl.load` using `group_idx[None, :]`. The K side (line 772) still uses the loop. Extending gathered-scale to K eliminates half the scale-broadcast overhead. This is a one-line structural change.

2. **Fuse the cache-write into the attention kernel.** Currently the decode path runs two separate kernels: quantize-and-write, then fused-decode. For decode (B=1 token per sequence), the new K/V is a single position — it could be quantized inline at the start of the decode kernel before the tile loop, avoiding a separate kernel launch entirely. This eliminates one kernel's scheduling overhead plus L2 cache pollution from the separate write pass.

3. **INT4x2 native byte load without per-element unpacking.** Instead of the current `low = (packed & 0x0F) - 8; high = (packed >> 4) - 8` pattern, precompute a 256-entry lookup table in shared memory that maps each uint8 → (low_dequant, high_dequant) pair. On H100's 228 KB shared memory per SM, a 256×2×4 = 2 KB LUT is negligible. This replaces 4 integer ops + 2 casts per byte with 1 shared-memory gather, which H100's LDS bandwidth can sustain.

4. **Halve accumulator width.** The even/odd split is needed because INT4 packing interleaves two values per byte. But the Q vector could be pre-packed to match: instead of `q_even * k_low + q_odd * k_high`, restructure Q at decode time so a single `tl.dot` (or vectorized FMA) processes the full head_dim. This requires changing how Q is laid out but eliminates the second accumulator and its `alpha` rescale.

**System-level:**

5. **Shrink BLOCK_N for short sequences.** The default `DECODE_BLOCK_N=128` means a 256-token sequence does only 2 tile iterations. If each iteration under-fills warps, a smaller `BLOCK_N=64` might pipeline better on H100 at these lengths. The existing env var makes this easy to sweep.

6. **Overlap cache-write with model compute.** If the cache-write kernel is not fused into the attention kernel (idea #2), it could be launched on a separate CUDA stream so it overlaps with the next layer's MLP. This requires changes in the vLLM scheduler but would hide the write latency entirely.

---

### 3. Is the ~2.8% gap intrinsic or worth chasing?

**Mostly intrinsic, but ~1% is recoverable.** The irreducible floor comes from the nibble unpacking + dequant math that the FP16 path simply doesn't do. At short decode lengths (256 tokens), the kernel is not bandwidth-bound enough for the 4x cache compression to help — the memory traffic savings manifest at longer contexts (the docs show 3.8-4.0x speedup at B=16, T=32K). At decode-256, the kernel is compute-bound, so extra ALU from dequant directly adds latency.

That said, the K scale loop (idea #1) and the separate cache-write kernel (ideas #2 or #6) are unnecessary overhead, not intrinsic. Fixing those could plausibly close 0.05-0.10 ms, bringing the gap to ~1.2-1.6%. Below that, you're paying for dequant ALU that can't be eliminated without hardware INT4 support.

**Verdict: the 2.8% gap is acceptable for production**. The value proposition of fused INT4 is memory capacity (2x more concurrent sequences) and long-context latency (3-4x at 32K tokens), not short-context decode speed. Chasing the last 1% at decode-256 has diminishing returns.

---

### 4. Best 3-5 next experiments

| # | Experiment | Expected impact | Effort |
|---|-----------|----------------|--------|
| 1 | **Enable gathered-scale for K** (extend H3 to K-side in decode kernel) | -0.02 to -0.05 ms, eliminates K scale loop | 30 min, ~10 lines |
| 2 | **Sweep BLOCK_N={32,64,128} × NUM_WARPS={2,4,8}** at decode-256 and decode-1024 | Find Pareto-optimal config per sequence length range | 2 hr, env vars only, no code change |
| 3 | **Measure at longer context (decode-2048, decode-4096)** | Expect fused to *beat* FP16 due to memory savings — this is the real payoff regime | 1 hr, config change only |
| 4 | **Fuse cache-write into decode kernel** (single-kernel decode for B=1) | Eliminates one kernel launch + L2 pollution, -0.02 to -0.05 ms | 4-8 hr, moderate kernel surgery |
| 5 | **Measure cache-write kernel time in isolation** (nsys profile of `reshape_and_cache_v_only_int4` during decode graph replay) | Quantifies how much of the 0.18 ms is write vs read. If write is >0.05 ms, idea #4 is justified | 1 hr, profiling only |

---

### 5. Non-obvious idea

**Pre-packed Q reordering at projection time.** The even/odd Q split (lines 719-722) and dual accumulator exist solely because INT4 packing stores two values per byte in low/high nibble order. If the model's Q projection output is permuted at the linear layer level (swap dims so even indices are contiguous in the first half, odd in the second), the decode kernel can use a single `tl.dot` on each half without interleaving. This moves zero cost to the Q projection (which is a matrix multiply — the permutation is free if baked into the weight matrix column order) and eliminates the dual-accumulator overhead from the latency-critical decode inner loop. It requires a one-time weight permutation at model load, not a runtime cost.

---

### Recommendation block

**Ship v0.0.2 as-is for production use.** The 2.8% gap at short decode is acceptable and will invert to a net win at longer contexts. For incremental improvement:

1. **Immediate (this week):** Enable gathered-scale for K (idea #1) — trivial change, measurable win.
2. **Next benchmark run:** Add decode-2048 and decode-4096 workloads to confirm the crossover point where fused beats FP16.
3. **If latency budget is tight:** Profile cache-write kernel time (idea #5) to decide whether fusing write+decode (idea #4) is worth the engineering.
4. **Skip** the LUT and Q-reordering ideas until you have evidence they matter at production batch sizes.
