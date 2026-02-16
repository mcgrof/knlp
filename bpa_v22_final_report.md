# BPA v22 Final Report: Scale Overhead Engineering + Larger Model + Bandwidth

## Executive Summary

v22 tested three paths to improve on v21's best (g32_k4, kv_ratio=0.3203):
(A) amortize scale metadata to make tight groups viable,
(B) replicate the schedule on Qwen2.5-1.5B, and
(C) prove bandwidth-bound latency gains.

**Result A succeeds.** Token-window scale amortization with g=8, S=8
achieves kv_ratio=0.3073 PASS@3% — a 4.1% byte reduction vs v21's
best and 7.7% vs S2_k6. This is the new Pareto-optimal for
Qwen2.5-0.5B.

Result B partially confirms: g32_k4 ratio improves from 0.3203 to
0.3136 on 1.5B (more layers dilute INT8 overhead). However, the
residual_ratio proxy catastrophically fails on 1.5B (+67960% PPL),
and transferred 0.5B ranking barely misses (+3.42%). Model-specific
empirical sensitivity is required.

Result C negative: W7900 remains compute-bound at all batch sizes
(1, 4, 8) at L=16384. No latency benefit from KV byte reduction.

## Phase 0: Regression Guard

All baselines pass:

| Config    | kv_ratio | max_delta | PASS@3% |
|-----------|----------|-----------|---------|
| g32_k4    | 0.3203   | +2.30%    | all     |
| S2_k6     | 0.333    | +2.35%    | all     |
| INT8_all  | 0.5156   | +0.69%    | all     |

## Phase 1: Scale Overhead Engineering

### Byte Accounting

The scale overhead problem: g=4 requires 16 groups/head with fp16
scales, giving 50% overhead (kv_ratio=0.50). Token-window amortization
shares one scale across S consecutive tokens, reducing scale cost by
factor S.

Effective overhead at head_dim=64, 2 KV heads:

| g   | S   | scale overhead | INT4 bytes/layer | kv_ratio (k=4) |
|-----|-----|----------------|------------------|-----------------|
| 32  | 1   | 11.1%          | 144              | 0.3203          |
| 8   | 1   | 33.3%          | 192              | 0.3984          |
| 8   | 2   | 20.0%          | 160              | 0.3464          |
| 8   | 4   | 11.1%          | 144              | 0.3203          |
| 8   | 8   | 5.9%           | 136              | 0.3073          |
| 4   | 8   | 11.1%          | 144              | 0.3203          |
| 4   | 16  | 5.9%           | 136              | 0.3073          |

### Strategy Results

Three strategies tested:

**Strategy 1 (Token-window amortization):** Works. Sharing scales
across S=8 tokens introduces moderate noise from scale drift but
stays within tolerance for g=8. The best config amort_g8_S8_k4
achieves kv_ratio=0.3073, max_delta=+2.43%, PASS@3%.

**Strategy 2 (INT8 log-scale):** Failed with infinite PPL due to a
dimension mismatch bug in the log-scale reconstruction. After fixing,
the byte savings are marginal (INT8 vs FP16 scales saves ~0.6% on
already-small overhead). Not pursued further.

**Strategy 3 (Cross-head sharing):** Limited applicability. Qwen2.5-0.5B
has only 2 KV heads, so sharing across heads saves nothing (scales
already per-group within each head; sharing across 2 heads doesn't
reduce count meaningfully). g=8 variant fails (+3.84%).

### Validated Configs (PASS@3% at all L)

| Config              | g  | S  | k | kv_ratio | max_delta | vs v21 best |
|---------------------|----|----|---|----------|-----------|-------------|
| amort_g8_S8_k4      | 8  | 8  | 4 | 0.3073   | +2.43%    | -4.1%       |
| amort_g8_S4_k4      | 8  | 4  | 4 | 0.3203   | +2.06%    | tied        |
| amort_g4_S8_k4      | 4  | 8  | 4 | 0.3203   | +2.99%    | tied        |
| amort_g8_S2_k4      | 8  | 2  | 4 | 0.3464   | +1.63%    | -           |
| amort_g4_S2_k4      | 4  | 2  | 4 | 0.3984   | +2.05%    | -           |
| amort_g8_S1_k4      | 8  | 1  | 4 | 0.3984   | +1.90%    | -           |
| amort_g4_S1_k4      | 4  | 1  | 4 | 0.5026   | +2.27%    | -           |
| sharedscale_g4_k4   | 4  | -  | 4 | 0.5026   | +2.95%    | -           |

Failed (exceed 3%):
- amort_g4_S16_k4: +3.89% (S=16 too aggressive for g=4)
- amort_g4_S4_k4: +3.11%
- sharedscale_g8_k4: +3.84%

### Key Insight

g=8 tolerates amortization better than g=4. At S=8, g=8 has 8
values per group — the max-abs scale drifts less across consecutive
tokens because each group captures more of the value range. g=4
with its 4 values per group is more sensitive to scale mismatch,
and only S<=2 stays safe.

## Phase 2: Larger Model (1.5B)

Qwen2.5-1.5B: 28 layers, head_dim=128, 2 KV heads.

### Byte Accounting Improvement

g32_k4 ratio: 0.3136 (vs 0.3203 on 0.5B). The improvement comes
from k/D=4/28=0.143 vs 4/24=0.167 — fewer layers need INT8
protection relative to total.

### Sensitivity Proxy Failure

The residual_ratio proxy produces ranking [3,2,9,4,11,10] for 1.5B.
This ranking causes catastrophic failure (+67960% PPL) because it
completely misses layer 0 (attention sink), which has residual_ratio
of only 0.064 — the LOWEST of all 28 layers. The proxy measures
reconstruction error magnitude, but for the sink layer the absolute
values are small while the functional importance is enormous.

Transferring the 0.5B ranking [0,8,1,2] nearly passes (+3.42%),
confirming that early layers (especially layer 0) remain critical,
but the specific ordering may differ across model sizes.

### Results

| Config           | k | ranking        | max_delta   | PASS@3% |
|------------------|---|----------------|-------------|---------|
| g32_k4_residual  | 4 | [3,2,9,4]      | +67960%     | FAIL    |
| g32_k6_residual  | 6 | [3,2,9,4,11,10]| +67917%     | FAIL    |
| g32_k4_transfer  | 4 | [0,8,1,2]      | +3.42%      | FAIL    |
| g32_k6_transfer  | 6 | [0,8,1,2,3,4]  | +3.44%      | FAIL    |
| INT8_all         |28 | all            | +0.26%      | PASS    |

The transferred ranking at k=4 hits +3.42% — very close to the 3%
threshold. This suggests k=5 or k=6 with a proper model-specific
sensitivity sweep would pass, giving kv_ratio around 0.33 or better.

## Phase 3: Bandwidth-bound Regime

Tested at L=16384 with batch sizes {1, 4, 8, 16}.

| Batch | dense ms | g32_k4 ms | speedup | bandwidth_bound |
|-------|----------|-----------|---------|-----------------|
| 1     | 36.52    | 36.61     | 0.998x  | NO              |
| 4     | 49.70    | 49.87     | 0.997x  | NO              |
| 8     | 72.85    | 72.85     | 1.000x  | NO              |
| 16    | OOM      | OOM       | -       | -               |

W7900 remains compute-bound. The attention kernel is dominated by
GEMM computation, not KV memory bandwidth. At batch=8, the GPU uses
42.3GB of 48.3GB VRAM but still cannot fill the memory bandwidth.
batch=16 OOMs.

This confirms the v16-v21 finding: bandwidth-bound latency gains
require either (a) much larger batch sizes (impossible on 48GB for
this model at L=16K), (b) a bandwidth-limited GPU (e.g., consumer
cards with narrower bus), or (c) a larger model where the KV cache
dominates total memory.

## Phase 4: Theory Update

Per-layer INT4 sigma (RMS quantization error) comparison:

| Layer | sigma g=32 | sigma g=4 | reduction |
|-------|------------|-----------|-----------|
| 0     | 4.306      | 1.702     | 60.5%     |
| 8     | 2.409      | 1.612     | 33.1%     |
| 1     | 1.947      | 0.874     | 55.1%     |
| 2     | 1.550      | 0.621     | 59.9%     |
| avg   | 0.703      | 0.326     | 53.7%     |

Total sigma: g=32=16.88, g=4=7.82 (53.7% reduction).

g=4 reduces noise by ~2x across all layers, which is why g=4 k=0
nearly passes (+4.5% vs g=32 k=0 at +30%). But the scale overhead
makes g=4 useless in byte terms unless amortized. The amortized
g=8 S=8 achieves a middle ground: less noise reduction than g=4
(g=8 sigma is intermediate) but viable scale overhead.

## Pareto Frontier Summary

| Config              | kv_ratio | max_delta | PASS@3% | notes          |
|---------------------|----------|-----------|---------|----------------|
| S2_k6 (v16)        | 0.333    | +2.35%    | all     | prior best     |
| g32_k4 (v21)       | 0.3203   | +2.30%    | all     | v21 best       |
| amort_g8_S8_k4 (v22)| 0.3073  | +2.43%    | all     | NEW BEST       |
| amort_g8_S4_k4      | 0.3203   | +2.06%    | all     | lower delta    |

The new champion amort_g8_S8_k4 achieves 69.3% KV compression
(1 - 0.3073) while maintaining PASS@3%.

## Answers to v22 Questions

1. **Did we get kv_ratio < 0.30?** No. Best is 0.3073. To go below
   0.30 requires k<=3, which still fails even with amortization.

2. **Did k* drop below 4?** No. k*=4 remains the floor for all group
   sizes tested. The amortization reduces scale overhead but does not
   change the fundamental noise accumulation that drives k*.

3. **Did larger model improve kv_ratio via k/D shrinkage?** Partially.
   The theoretical ratio improves (0.3136 vs 0.3203), but the
   sensitivity proxy fails catastrophically and transferred ranking
   barely misses. Model-specific empirical sensitivity is needed.

4. **Are we bandwidth-bound?** No. W7900 is compute-bound at all
   tested batch sizes. No latency gains from KV byte reduction.

5. **v23 recommendation:** See branch tree.
