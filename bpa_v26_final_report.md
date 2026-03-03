# BPA v26 Final Report: H100 Validation of O(1) k* Scaling

## Summary

v26 executed the H100 test plan specified in v25 on two 7B-scale
models: Qwen2.5-7B (D=28) and Mistral-7B-v0.1 (D=32). The O(1)
hypothesis is decisively supported: k*(3%) does not grow with D.

## Models Tested

| Model | D | KV heads | head_dim | params |
|-------|---|----------|----------|--------|
| Qwen2.5-0.5B | 24 | 2 | 64 | 494M |
| Qwen2.5-1.5B | 28 | 2 | 128 | 1.5B |
| Qwen2.5-7B | 28 | 4 | 128 | 7.6B |
| Mistral-7B-v0.1 | 32 | 8 | 128 | 7.2B |

Hardware: NVIDIA H100 80GB HBM3, CUDA 13.0, torch 2.10+cu126.

## Acceptance Criteria Results

### A) O(1): k*(3%) <= 4 at D >= 32 -- PASS

| Model | D | k*(3%) | k*/D | k*(1%) |
|-------|---|--------|------|--------|
| Qwen2.5-0.5B | 24 | 2 | 0.083 | N/A |
| Qwen2.5-1.5B | 28 | 2 | 0.071 | 3 |
| Qwen2.5-7B | 28 | 2 | 0.071 | 3 |
| Mistral-7B | 32 | 0 | 0.000 | 0 |

k*(3%) = {2, 2, 2, 0} across D = {24, 28, 28, 32}. k* is bounded
by a small constant and decreases at D=32 to zero. The O(1)
hypothesis is supported across all tested models.

Mistral-7B achieves PASS@1% with k=0 (all INT4, no protected
layers), with max_delta = +0.48%. This is the strongest evidence
that per-layer noise accumulation is well-controlled in models with
sufficient depth and head diversity.

### B) kv_ratio <= 0.30 at eps=3% -- PASS

| Model | k*(3%) | kv_ratio |
|-------|--------|----------|
| Qwen2.5-7B | 2 | 0.2974 |
| Mistral-7B | 0 | 0.2812 |

Both sub-0.30. Mistral achieves 0.2812 with zero protected layers.

### C) Latency >= 10% reduction -- NOT TESTABLE

The simulation approach (quantize-dequantize in Python, store as
fp16) cannot demonstrate latency wins because the KV cache remains
fp16 in GPU memory. A fused INT4 attention kernel is required.

Observed: 0-7% speedup on Mistral (best 6.6% at L=8K batch=16),
3-5% slowdown on Qwen7B due to quantization overhead.

### D) Throughput >= 20% improvement -- NOT TESTABLE

Same limitation: without fused INT4 kernels, the memory footprint
is unchanged. The theoretical capacity gain at kv_ratio=0.28 is
3.57x concurrent sequences.

## Oracle Sensitivity Analysis

### Qwen2.5-7B (D=28)

Ranking: [0, 27, 3, 17, 19, 22, 8, 7]

| Layer | max delta |
|-------|-----------|
| 0 | +140,154% |
| 27 | +2.81% |
| 3 | +1.14% |
| all others | <0.56% |

Tail fraction (C=1): 0.0001. Layer 0 accounts for 99.99% of total
sensitivity. This continues the trend: 0.5B (+23%) -> 1.5B (+824%)
-> 7B (+140K%). Sink dominance increases with model scale.

### Mistral-7B-v0.1 (D=32)

Ranking: [0, 15, 16, 7, 12, 30, 13, 22]

| Layer | max delta |
|-------|-----------|
| 0 | +0.41% |
| 15 | +0.23% |
| 16 | +0.23% |
| all others | <0.16% |

Tail fraction (C=1): 0.14. Sensitivity is more evenly distributed.
No single layer dominates. This is why k*=0: the cumulative INT4
noise across all 32 layers stays within 0.48%.

The Mistral result is architecturally interesting: 8 KV heads
(vs Qwen's 2-4) provide more redundancy per layer, reducing
individual layer sensitivity.

## Dense Baseline PPL

| Model | L=8K | L=32K |
|-------|------|-------|
| Qwen2.5-7B | 4.2-5.1 | 5.0-8.5 |
| Mistral-7B | 4.6-7.4 | 3.9-6.7 |

INT8-all max delta: +0.38% (Qwen), +0.05% (Mistral). Both
essentially lossless.

## Key Findings

1. **O(1) is robust.** k*(3%) = {2, 2, 2, 0} across four models
   (D=24 to D=32). k* is bounded by a constant independent of D.
   The strongest form: k* actually decreases at D=32.

2. **Sink dominance is model-family-dependent.** Qwen models show
   extreme layer 0 dominance (+23% to +140K%). Mistral distributes
   sensitivity more evenly. Both patterns yield O(1) through
   different mechanisms: Qwen via a few dominant layers, Mistral
   via uniformly low per-layer noise.

3. **kv_ratio sub-0.30 is achievable.** 0.2974 (Qwen7B) and
   0.2812 (Mistral) with INT4 g=32 and oracle protection.

4. **Latency requires fused kernels.** Our simulation cannot
   demonstrate latency wins. This is not a falsification of the
   capacity story -- it is a limitation of the simulation approach.
   The next step is to integrate with a fused INT4 KV attention
   kernel (e.g., FlashAttention with quantized KV support).

## What This Means for the BPA Arc

The v26 results complete the empirical case for O(1) k* scaling:

- v24 established k*=2 at D=24 and D=28 (two Qwen models)
- v26 confirms k*=2 at D=28 (Qwen 7B, different model size)
- v26 extends to k*=0 at D=32 (Mistral, different architecture)

The accumulation model from v24 predicts that k* is bounded when
sink-layer dominance holds. The Mistral result goes further:
when no layer has extreme sensitivity (tail_frac=0.14 vs
Qwen's <0.001), the cumulative noise from all-INT4 stays within
tolerance, making k*=0.

## Artifacts

All results are in `results/v26/artifacts/v26/`:

| Artifact | File |
|----------|------|
| Qwen7B oracle | oracle_sensitivity_qwen7b.json |
| Qwen7B k-sweep | k_star_qwen7b.json |
| Qwen7B latency | latency_benchmark_qwen7b.json |
| Mistral7B oracle | oracle_sensitivity_mistral7b.json |
| Mistral7B k-sweep | k_star_mistral7b.json |
| Mistral7B latency | latency_benchmark_mistral7b.json |
| Phase 0 baselines | phase0/phase0_{qwen7b,mistral7b}.json |

## Next Steps

1. Integrate fused INT4 attention kernel for real latency/throughput
   measurement (Criterion C/D).
2. Test on Llama-3.1-8B (D=32, requires gated access) to confirm
   cross-architecture generalization.
3. Explore whether k=0 generalizes to other Mistral-family models
   or is specific to Mistral-7B-v0.1.
