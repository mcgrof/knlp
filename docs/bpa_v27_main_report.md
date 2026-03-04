# BPA v27: Bounded Protection in Mixed-Precision KV Caches

## 1. Executive Summary

Under mixed-precision KV cache quantization, the number of
transformer layers requiring high-precision protection is bounded
by a small constant independent of model depth. We observe
k*(3%) = {2, 2, 2, 0, 0} across D = {24, 28, 28, 32, 32},
spanning five models, three architecture families, and two GPU
platforms. This yields effective KV compression ratios of
0.28-0.30, corresponding to a theoretical 3.3-3.6x increase in
concurrent serving capacity.

Two distinct mechanisms underlie this result. In Qwen-family models,
layer 0 (the attention sink) dominates INT4 sensitivity, accounting
for 67-99.99% of total per-layer error. Protecting 1-2 sink layers
is sufficient. In Mistral (8 KV heads) and Llama-2 (32 KV heads,
MHA), per-layer sensitivity is uniformly low (max 0.48% and 1.01%
respectively), and all-INT4 quantization stays within tolerance.

The result is empirical. We do not claim asymptotic O(1) scaling
but report that k* does not grow across the tested depth range.
Latency gains require fused INT4 attention kernels not yet available
in our evaluation framework.

## 2. Problem Statement

The KV cache is the dominant memory consumer for long-context LLM
inference. At batch size B, context length L, with D layers, n_kv
KV heads, and head dimension d_h, the cache requires:

    memory = 2 * B * L * D * n_kv * d_h * sizeof(dtype)

For a 7B model at L=32K with fp16, this is approximately 4-8 GB per
sequence. At batch sizes needed for high throughput, KV cache memory
exceeds model weight memory.

Mixed-precision quantization offers a path: compress the KV cache
to INT4 (4 bits per value) for most layers while keeping a few
critical layers at INT8 or fp16. The open question is: how many
layers need protection?

If the answer is k* = O(D), then mixed-precision offers only a
constant factor improvement. If k* = O(1), the effective compression
ratio approaches the quantization floor as models grow deeper, and
the protected overhead becomes negligible.

## 3. Definitions

**D**: Number of transformer layers.

**k**: Number of layers kept at INT8 (high precision). The remaining
D-k layers use INT4 with group size g=32.

**k*(eps)**: The minimum k such that max |delta_pct| <= eps across
all tested (context length, random seed) pairs, where delta_pct is
the relative PPL change from dense baseline.

**kv_ratio**: Total compressed KV bytes per token divided by dense
fp16 bytes per token, including quantization scale metadata.

**PASS_eps**: A configuration passes if k*(eps) exists and is at
most D. PASS_3% means eps=3%.

**Empirical lower boundary (ELB)**: The lowest kv_ratio achieving
PASS_eps across tested configurations. This is not an
information-theoretic bound but the best ratio achieved under our
protocol.

**Oracle ranking**: Per-layer sensitivity ranking determined by
single-layer INT4 ablation (one forward pass per layer).

## 4. Experimental Arc

The BPA (Bounded Protection Allocation) series spans v8-v26 over
approximately 5 months of experimentation. The arc progressed
through several falsified hypotheses before converging on the
current result.

**v8-v13**: Established tiered KV management basics on Qwen2.5-0.5B.
Simple tiering schemes (copy to slower storage) worked at 3% PPL
tolerance. Learned compression methods (MLA, KVSplice) failed due
to lossy random projections destroying position information.

**v14b-v15**: Investigated compression fidelity. Only INT8 passed
universally. Discovered that K values are compressible (RoPE
concentrates energy in few directions) but V values are not.
rope_complex (complex-plane K grouping) achieved full pass by
preserving RoPE phase while compressing magnitude.

**v16**: Broke the kv_ratio=0.5 barrier with mixed INT4/INT8.
Established that 6 out of 24 layers need INT8 protection (the
"S2" configuration). Sensitivity-guided layer assignment solved
the cumulative INT4 degradation problem.

**v18-v19**: Searched for cheap proxy signals to replace oracle
ranking. Adam v-hat bit allocation (v18) and a 95-variant signal
bakeoff (v19) both fell short. C3_residual_ratio was the best proxy
(rho=0.394 with oracle) but none matched oracle quality.

**v20-v21**: Attempted to break the k-floor. Tight-group INT4
(g=4) reduced per-layer noise 8x but incurred 50% scale overhead.
Standard g=32 with oracle ranking achieved ratio=0.3203 at k=4.

**v22-v23**: Scale amortization and model scaling. Amortized-scale
g=8 S=8 achieved ratio=0.2969 (Pareto-optimal for 0.5B). Scaling
to 1.5B showed k* remained at 2. Two sub-0.30 passing configs.

**v24**: Formalized the accumulation theory. Error propagation
through the residual stream: e_D = sum of amplified per-layer noise.
Derived the k(D,eps) lower bound. The O(1) condition requires
bounded sink count plus bounded tail. Supported by data: k*=2 at
both D=24 and D=28.

**v26**: H100 validation at 7B scale. Qwen2.5-7B confirmed k*=2
(same as smaller Qwen models). Mistral-7B-v0.1 achieved k*=0
(all-INT4 passes at <0.5% max delta). Two architecturally distinct
mechanisms, both yielding bounded k*.

Key falsified assumptions along the way:
- MLA/KVSplice compression would work (falsified v13-v14)
- V values could be compressed (falsified v14b)
- A cheap proxy signal could replace oracle ranking (falsified v18-v19)
- g=4 would beat g=32 on kv_ratio (falsified v21: scale overhead)
- k* is an architectural constant (refined v20→v24: depends on ranking quality)

## 5. Main Empirical Result

### Table 1: Canonical k* Results

| Model | Arch | D | n_kv | k*(3%) | k*/D | kv_ratio | max_delta |
|-------|------|---|------|--------|------|----------|-----------|
| Qwen2.5-0.5B | Qwen2 | 24 | 2 | 2 | 0.083 | 0.3008 | 2.85% |
| Qwen2.5-1.5B | Qwen2 | 28 | 2 | 2 | 0.071 | 0.2974 | 1.05% |
| Qwen2.5-7B | Qwen2 | 28 | 4 | 2 | 0.071 | 0.2974 | 1.05% |
| Mistral-7B | Mistral | 32 | 8 | 0 | 0.000 | 0.2812 | 0.48% |
| Llama-2-7b | Llama | 32 | 32 | 0 | 0.000 | 0.2812 | 1.01% |

All results use: INT4 g=32 symmetric / INT8, oracle ranking, 3
seeds, wikitext-103-raw-v1, 64 decode tokens. L in {8K, 32K} for
models with sufficient context; L in {2K, 4K} for Llama-2-7b
(max_position_embeddings=4096).

k* does not increase with D. Within the Qwen family (D=24 to D=28),
k* is constant at 2. Across architectures (D=32), both Mistral and
Llama-2 achieve k*=0. The protected fraction k*/D decreases:
0.083, 0.071, 0.071, 0.000, 0.000.

### Effective compression

At k*=2 (Qwen-7B): kv_ratio = 0.2974, meaning the compressed KV
cache is 29.7% of the dense size. At k*=0 (Mistral-7B):
kv_ratio = 0.2812, approaching the theoretical INT4-all floor of
0.28125 for g=32 head_dim=128.

## 6. Mechanistic Interpretation

### Sink dominance (Qwen family)

Qwen models exhibit extreme layer 0 sensitivity that increases
with model scale:

| Model | Layer 0 delta | Next layer | Tail fraction |
|-------|---------------|------------|---------------|
| Qwen2.5-0.5B | +23.5% | +1.34% (layer 2) | 33.6% |
| Qwen2.5-1.5B | +824.6% | +3.20% (layer 15) | 0.85% |
| Qwen2.5-7B | +140,154% | +2.81% (layer 27) | 0.01% |

Layer 0 serves as the primary attention sink. INT4 quantization of
its K values corrupts the RoPE embeddings that encode absolute
position, causing catastrophic degradation. The remaining D-1 layers
have uniformly low sensitivity because (a) RoPE rotation has already
been applied and positional information is distributed, and (b) the
quantization noise per layer is small relative to the signal.

The tail fraction decreases with model scale, meaning the gap
between the dominant layer(s) and the rest widens. This is
consistent with O(1) scaling: as D increases, the number of
dominant layers stays constant while the tail becomes relatively
smaller.

### Uniform robustness (Mistral, Llama-2)

Mistral-7B-v0.1 and Llama-2-7b show a fundamentally different
pattern from Qwen. Sensitivity is distributed relatively evenly
across all 32 layers with no dominant sink:

| Model | n_kv_heads | max layer delta | k*(3%) |
|-------|-----------|-----------------|--------|
| Mistral-7B | 8 | 0.41% (layer 0) | 0 |
| Llama-2-7b | 32 | 1.18% (layer 3) | 0 |

Notably, Llama-2-7b with MHA (32 KV heads = 32 attention heads) has
no special sensitivity at layer 0 (delta=0.76%, ranked 22nd). The
most sensitive layer is layer 3 at 1.18%, and the top-8 deltas span
only 0.86-1.18% — remarkably uniform.

The likely cause is head redundancy. With more KV heads, each head
carries less information, providing redundancy that absorbs
quantization noise. Even the cumulative INT4 error across all 32
layers stays within tolerance (0.48% for Mistral, 1.01% for Llama).

### Why both yield O(1)

Both mechanisms produce bounded k* through different paths:

1. **Sink path**: A constant number C of layers (typically 1-2)
   have sensitivity >> eps. Protecting these C layers is sufficient
   because the remaining D-C layers have bounded cumulative error.
   k* = C = O(1).

2. **Uniform path**: No layer has sensitivity > eps even
   individually. The cumulative error across all D layers stays
   below eps because per-layer noise is O(1/sqrt(n_kv_heads)) or
   similar. k* = 0.

## 7. Practical Implications

### Capacity gain

| Model | kv_ratio | Capacity multiplier |
|-------|----------|---------------------|
| Qwen2.5-0.5B | 0.3008 | 3.32x |
| Qwen2.5-1.5B | 0.2974 | 3.36x |
| Qwen2.5-7B | 0.2974 | 3.36x |
| Mistral-7B | 0.2812 | 3.56x |
| Llama-2-7b | 0.2812 | 3.56x |

These are theoretical upper bounds assuming KV cache is the sole
memory bottleneck. In practice, model weights, activations, and
system overhead reduce the effective gain.

### What is and is not proven for latency

**Measured**: The simulation approach (quantize-dequantize in Python,
store as fp16) shows 0-7% speedup on Mistral and 3-5% slowdown on
Qwen7B due to quantization overhead. This is not meaningful because
the KV cache remains fp16 in GPU memory.

**Not measured**: Actual decode latency with a fused INT4 attention
kernel. Such a kernel would load 4x less data from the KV cache,
which should translate to proportional speedup in the
bandwidth-limited decode regime. This is the gap between "capacity"
(measured) and "latency" (not measured).

### Deployment recipe

For a system builder, the practical takeaway:

1. Run oracle ranking (D forward passes, one-time calibration)
2. Set top-k* layers to INT8, rest to INT4 g=32
3. Always protect sink tokens (first 4) at fp16
4. Keep a near-window (last 1024 tokens) at fp16

Cost: <3% PPL, ~3.3x KV memory reduction. The oracle calibration
is a one-time cost per model.

## 8. Relation to L-squared-M

The Learned Long-context Memory (L-squared-M) framework motivates
the question: what is the minimum KV cache footprint for a given
quality level? Our work provides an empirical route toward this
bound within the mixed-precision INT4/INT8 family.

**What ELB means**: The empirical lower boundary (0.28-0.30) is
the best kv_ratio we achieve under our protocol. It is not an
information-theoretic lower bound.

**What it does not mean**: We do not claim this is the minimum
possible ratio. Lower bitwidths (INT3, INT2), different quantization
schemes (NF4, GPTQ), learned compression, or token-level eviction
could all potentially achieve lower ratios.

**The asymptotic argument**: If k* remains O(1) as D increases,
then kv_ratio approaches the INT4-all floor (0.28125 for g=32,
hd=128). The gap between achieved ratio and floor is
O(k*/D) * (int8_layer - int4_layer), which vanishes if k* is
constant and D grows.

This is suggestive but not proven. It requires O(1) to hold at
arbitrary D, which we cannot verify from four data points.

## 9. Limitations

1. **D range**: 24-32 (three distinct values). Proving O(1)
   requires larger D (48, 64, 80+).

2. **Oracle ranking**: All headline results use oracle ranking
   (D forward passes per model). No proxy achieves the same k*.
   Transfer ranking (apply 0.5B ranking to 1.5B) nearly works
   (+3.42%) but does not match oracle.

3. **No fused INT4 kernel**: Latency criterion not met. Our
   simulation quantizes and dequantizes in Python, storing fp16.
   Real latency gains require hardware-aware kernels.

4. **Three architecture families**: Qwen2 (GQA, 2-4 KV heads),
   Mistral (GQA, 8 KV heads), and Llama-2 (MHA, 32 KV heads).
   No models above 7.6B. Llama-2 limited to L<=4K context.

5. **Wikitext only**: PPL on wikitext-2-raw-v1. No downstream task
   evaluation (MMLU, HumanEval, etc.).

6. **Fixed quantization**: g=32 symmetric INT4 only. Other group
   sizes, asymmetric quantization, and per-channel schemes may yield
   different k*.

7. **Greedy decoding**: 64-token continuations with greedy decoding.
   Beam search or sampling not tested.

8. **No gradient-based analysis**: The accumulation theory uses
   empirical noise measurements, not formal gradient analysis.

## 10. H100 Next Steps

### COMPLETED: Third architecture confirmation

Llama-2-7b-hf (D=32, MHA, 32 KV heads) confirmed k*(3%)=0 with
max_delta=1.01%. Uniform robustness pattern: no dominant sink layer,
sensitivity distributed across all 32 layers (max per-layer
delta=1.18%). This is the third architecture family confirming
O(1) bounded k*.

### Priority 1: Larger D

Test on a 13B+ model (D >= 40) to extend the D range. Candidates:
- Qwen2.5-14B (D=48 if accessible)
- Llama-2-13b (D=40, MHA, but limited to L=4K)

### Priority 2: Fused kernel integration

Integrate with a fused INT4 attention kernel to measure actual
latency. Candidates: FlashInfer with INT4 KV, vLLM quantized
cache, or custom CUDA kernel. The latency measurement is the
primary gap between "interesting empirical study" and "systems
paper."

### Acceptance criteria

A) ACHIEVED: k*(3%)=0 on Llama-2-7b (third architecture).
B) ACHIEVED: kv_ratio=0.2812 on Llama-2-7b.
C) Latency claim requires >= 10% ms/token improvement in
   bandwidth-bound regime — NOT YET MEASURED.
D) Absent C, report capacity/throughput only (current status).
