# BPA v23 Final Report

## v22 Recap

v22 established amort_g8_S8_k4 as the Pareto-optimal KV cache
compression config on Qwen2.5-0.5B: kv_ratio=0.3073 with PASS@3%
across all L. The k-floor remained at k*=4 (structural). W7900 was
compute-bound at all batch sizes. On the 1.5B model, the
residual_ratio proxy failed catastrophically (+67960% PPL by ranking
layer 0 last), while the transferred 0.5B ranking nearly passed
(+3.42%). v22 concluded that v23 must do empirical oracle ranking
on 1.5B, build a throughput story for W7900, and stress-test
amortization knobs.

## v23 Goals

A) W7900-native throughput/capacity headline from KV byte cuts.
B) Replicate and stabilize on 1.5B using empirical sensitivity.
C) Stress-test amortization knobs (S, g, scale mode) and find
   Pareto-optimal configs under real byte accounting.
D) Produce H100 readiness dossier.

## Phase 0: Baseline Lock

All four baselines pass regression at L={8K, 16K, 32K} × 3 seeds:

| Config          | max_delta | kv_ratio | Status |
|-----------------|-----------|----------|--------|
| amort_g8_S8_k4  | +2.43%    | 0.3073   | PASS   |
| g32_k4          | +2.30%    | 0.3203   | PASS   |
| S2_k6           | +0.70%    | 0.3333   | PASS   |
| INT8_all        | +0.69%    | 0.5156   | PASS   |

## Phase 1: Throughput Benchmark (W7900 Value Story)

Tested 4 methods across L={8K, 16K, 32K} × batch={1, 4, 8, 16}.

**Latency**: W7900 remains compute-bound. No per-token latency
improvement from KV compression. The amortized backend is ~8%
*slower* than dense due to Python-level quant/dequant overhead
(22.9 vs 21.9 ms/tok at L=8K; 75.9 vs 68.0 ms/tok at L=32K).
Latency is flat across batch sizes for all methods — the GPU is
compute-saturated regardless of KV cache size.

**Capacity headline**: KV compression enables dramatically more
concurrent sequences. The capacity gain comes from reduced KV
memory per sequence, freeing VRAM for additional sequences.

| L     | dense | g32_k4 | amort  | amort vs dense |
|-------|-------|--------|--------|----------------|
| 8192  | 494   | 1543   | 1608   | **3.26x**      |
| 16384 | 247   | 771    | 804    | **3.26x**      |
| 32768 | 123   | 385    | 402    | **3.27x**      |

Model memory: 982 MB. GPU VRAM: 48.3 GB. The remaining ~47 GB
is available for KV cache. At L=32K, dense KV costs ~384 MB/seq,
while amort_g8_S8_k4 costs ~118 MB/seq — a 3.3x reduction.

## Phase 2: Amortization Parameter Sweep

Swept 12 configs: g={8,4}, S={1,2,4,8,16}, scale_mode={post_rope,
pre_rope, norm_invariant}, all at k=4.

**Key findings**:

1. **g8_S16 fails** (+5.08%): S=16 is too aggressive for g=8.
   The scale drifts too much across 16 tokens.

2. **Scale mode is irrelevant for g=8**: post_rope, pre_rope,
   and norm_invariant all produce identical results. This is
   because g=8 groups 8 consecutive elements of head_dim=64,
   and the scale variation within a group is similar regardless
   of RoPE phase. This simplifies deployment — use post_rope.

3. **g=4 shows RoPE drift** at small S (S=1, S=2, S=4): delta
   grows with L. g=4 S=1 goes from 0.65% at 8K to 1.17% at 32K.
   This is non-catastrophic but warns against g=4 at very long
   context.

4. **g8_S8_k3 breaks the 0.30 wall**: kv_ratio=0.2969 with
   max_delta=+2.38%, PASS@3%. This is the first sub-0.30 config
   on 0.5B that passes quality gates.

**Pareto frontier** (all PASS@3%):

| Config              | kv_ratio | max_delta |
|---------------------|----------|-----------|
| amort_g8_S8_k3      | **0.2969** | +2.38%  |
| amort_g8_S8_k4      | 0.3073   | +2.43%    |
| amort_g8_S4_k3      | 0.3105   | +2.05%    |
| amort_g8_S4_k4      | 0.3203   | +2.06%    |
| amort_g8_S2_k3      | 0.3379   | +1.73%    |
| amort_g8_S2_k4      | 0.3464   | +1.63%    |

The g8_S8_k3 config is the new Pareto-optimal on 0.5B.
No RoPE drift detected (delta decreases: 1.61% → 1.35% → 1.30%).

## Phase 3: 1.5B Oracle Sensitivity + k-sweep

### Oracle Ranking

Empirical per-layer INT4 ablation on Qwen2.5-1.5B (28 layers,
head_dim=128, 2 KV heads). Each layer tested individually at INT4
while all others remain INT8. Two-stage refinement: 1 seed for all
28 layers, then 3 seeds for the top-8.

**Final ranking** (top-8): [0, 15, 1, 19, 25, 23, 12, 9]

| Layer | max_delta (3-seed) | Note |
|-------|--------------------|------|
| 0     | +824.55%           | Attention sink (extreme) |
| 15    | +3.20%             | Only other >1% layer |
| 1     | +0.79%             | |
| 19    | +0.43%             | |
| 25    | +0.23%             | |
| 23    | +0.20%             | |
| 12    | +0.16%             | |
| 9     | +0.12%             | |

Layer 0 sensitivity on 1.5B (824.6%) is 5x worse than on 0.5B
(161.8%). This confirms the attention sink mechanism strengthens
with model size. Layer 15 (+3.2%) is the only other layer
exceeding 1% — all remaining 26 layers are under 0.8%.

### k-sweep Results

| Config            | k | kv_ratio | max_delta | Status |
|-------------------|---|----------|-----------|--------|
| g32_k2_oracle     | 2 | **0.2974** | +1.05%  | PASS   |
| g32_k3_oracle     | 3 | 0.3055   | +0.73%    | PASS   |
| g32_k4_oracle     | 4 | 0.3136   | +0.47%    | PASS   |
| g32_k6_oracle     | 6 | 0.3298   | +0.71%    | PASS   |
| g32_k8_oracle     | 8 | 0.3460   | +0.82%    | PASS   |
| amort_g8_S8_k4    | 4 | 0.3002   | +0.82%    | PASS   |

**k\* drops from 4 (0.5B) to 2 (1.5B)**. Only layers 0 and 15
need INT8 protection. This validates the k/D scaling hypothesis:
as model depth D increases, the fraction of sensitive layers
shrinks, improving kv_ratio.

**k/D comparison**:
- 0.5B: k*=4, D=24, k/D=0.167, g32_k4 ratio=0.3203
- 1.5B: k*=2, D=28, k/D=0.071, g32_k2 ratio=0.2974

The 1.5B achieves sub-0.30 kv_ratio with just g32 (no amortization
needed), purely from the k/D improvement.

### INT8-all Baseline on 1.5B

INT8_all: max_delta=+0.26%, PASS. Essentially lossless, consistent
with 0.5B findings.

## What Remains Blocked Until H100

The W7900 is compute-bound at all batch sizes tested (1-16) and
all sequence lengths (8K-32K). Per-token latency does not improve
when kv_ratio drops. The capacity story (3.26x more concurrent
sequences) is real but doesn't demonstrate latency gains.

Blocked experiments:
1. Latency improvement from KV compression in bandwidth-bound
   regime (requires H100 or consumer GPU with narrower bus)
2. Fused INT4 attention kernel performance
3. 7B/8B model with full oracle ranking + long context
4. Cost-per-token analysis in production setting

See artifacts/v23/h100_readiness.md for the detailed run list.

## Summary

| Metric | v22 Best | v23 Best (0.5B) | v23 Best (1.5B) |
|--------|----------|-----------------|-----------------|
| kv_ratio | 0.3073 | **0.2969** | **0.2974** |
| Config | amort_g8_S8_k4 | amort_g8_S8_k3 | g32_k2_oracle |
| k* | 4 | 3 (amort) | 2 |
| Capacity gain | 3.26x | 3.26x | N/A (not measured) |
| Latency gain | None (compute-bound) | None | None |
| PASS@3% | Yes | Yes | Yes |

v23 achieves two sub-0.30 PASS configurations:
1. amort_g8_S8_k3 on 0.5B (ratio=0.2969, +2.38%)
2. g32_k2_oracle on 1.5B (ratio=0.2974, +1.05%)

The k/D scaling hypothesis is confirmed: k* shrinks with model
depth, making INT4 KV compression increasingly attractive for
larger models. Extrapolating: a 70B model (D~56) might need only
k*~2, yielding kv_ratio~0.286 with simple g32 quantization.
