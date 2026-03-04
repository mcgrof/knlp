# Version Reconciliation: BPA v8-v26

## Purpose

This document maps the evolution of BPA experiments across versions,
identifies apparent contradictions, and explains why they are (or
are not) real inconsistencies.

## Version Map

| Version | Focus | GPU | Models | Key result |
|---------|-------|-----|--------|------------|
| v8-v13 | Tiered KV basics | W7900 | Qwen2.5-0.5B | bitter1-3 PASS@3%, MLA/splice FAIL |
| v14b | Compression fidelity | W7900 | Qwen2.5-0.5B | Only INT8 passes, V not compressible |
| v15 | Structural bottleneck | W7900 | Qwen2.5-0.5B | rope_complex FULL PASS@1% |
| v16 | Breaking the 0.5 wall | W7900 | Qwen2.5-0.5B | Mixed S2 (18xINT4+6xINT8) PASS@1% |
| v18 | Adam v-hat proxy | W7900 | Qwen2.5-0.5B,7B | Cheap heuristic, inferior to oracle |
| v19 | Signal bakeoff | W7900 | Qwen2.5-0.5B | C3_residual best proxy (rho=0.394) |
| v20 | Break k-floor | W7900 | Qwen2.5-0.5B | k*=4 architectural; g=4 tight-group |
| v21 | Exploit g=4 | W7900 | Qwen2.5-0.5B | g=4 50% scale overhead; g32 k=4 ratio=0.3203 |
| v22 | Scale amortization | W7900 | Qwen2.5-0.5B,1.5B | amort_g8_S8_k3 ratio=0.2969 Pareto |
| v23 | Max-out + oracle | W7900 | Qwen2.5-0.5B,1.5B | Sub-0.30: 0.2969 (0.5B), 0.2974 (1.5B) |
| v24 | Accumulation theory | W7900 | Qwen2.5-0.5B,1.5B | O(1) supported, theory derived |
| v25 | Technical narrative | — | — | Paper-ready arc documentation |
| v26 | H100 validation | H100 | Qwen2.5-7B, Mistral-7B | O(1) confirmed: k*={2,2,2,0} |

## Reconciliation of k* Values

### Issue 1: Different k* under different quant backends

The k* value depends on the quantization backend:

| Backend | k* (0.5B) | k* (1.5B) | Notes |
|---------|-----------|-----------|-------|
| Standard g32 INT4/INT8 | 2 | 2 | Canonical (v24) |
| Amortized g8 S8 | 3 | — | v22 Pareto config |
| Amortized g4 S2 | — | — | v22 marginal |
| Standard g32 (v20) | 4 | — | Earlier ranking method |

**Explanation**: The v20 k*=4 used a different oracle ranking than
v24. In v20, the ranking was [0, 8, 1, 2]; in v24, the ranking was
updated to [0, 2, 11, 16] using the same protocol but different
random seeds. The v24 ranking is canonical. Under the v24 ranking,
k*=2 for Qwen2.5-0.5B.

The amortized variants have k*=3 (higher) because the scale
amortization introduces additional error that requires one more
protected layer to compensate.

**Resolution**: All headline claims use the standard g32 INT4/INT8
backend with oracle ranking from the canonical protocol. This is
the simplest and most comparable setup.

### Issue 2: k-floor under standard INT4 vs amortized variants

v20 established k*=4 as an "architectural" floor. v24 then found
k*=2 with a better oracle ranking. This apparent contradiction
exists because:

1. v20 used a 1-seed oracle at L=8K only; v24 used 3-seed oracle.
2. The sensitivity landscape of Qwen2.5-0.5B is relatively flat
   beyond layer 0 (tail fraction 33.6%), so ranking quality matters.
3. k*=4 was the floor *for that particular ranking*, not an
   absolute architectural property.

**Resolution**: k* is ranking-dependent. The oracle ranking at 3
seeds is the canonical ranking. Under this ranking, k*=2 for all
Qwen models. The earlier k*=4 result from v20 is superseded.

### Issue 3: O(1) claim across different D

The O(1) claim rests on:

| D | k*(3%) | Model |
|---|--------|-------|
| 24 | 2 | Qwen2.5-0.5B |
| 28 | 2 | Qwen2.5-1.5B |
| 28 | 2 | Qwen2.5-7B |
| 32 | 0 | Mistral-7B |

The D=28 data point appears twice (1.5B and 7B) because both
models have 28 layers. This is not a contradiction but reduces the
effective D range: we have 3 distinct D values {24, 28, 32}, not 4.

The Mistral k*=0 result is architecturally different (8 KV heads
vs 2-4 in Qwen). Comparing across architectures is informative
but not a controlled experiment. The O(1) claim is supported by:
- Within Qwen family: k*=2 at D=24 and D=28 (constant)
- Across architectures: k*=0 at D=32 (bounded)

**Resolution**: The O(1) claim is empirical, not proven. It holds
over the tested range D in {24, 28, 32} under the canonical
protocol. We do not extrapolate to arbitrary D.

### Issue 4: Sink dominance vs uniform robustness

Qwen models exhibit extreme layer 0 sensitivity (23% to 140K%),
while Mistral shows uniform sensitivity (max 0.41%). Both achieve
low k*. This is not a contradiction: they represent two different
mechanisms that both yield bounded k*.

- Qwen path: few dominant layers + bounded tail implies small k*
- Mistral path: all layers uniformly robust implies k*=0

Both are consistent with the accumulation theory from v24: k* is
bounded when either (a) sink dominance confines error to a few
layers, or (b) per-layer noise is small enough that cumulative
error stays within tolerance even at k=0.

### Issue 5: Amortized-scale Pareto-optimality

v22 found amort_g8_S8_k3 at kv_ratio=0.2969 as Pareto-optimal
for Qwen2.5-0.5B. v24 found standard g32_k2 at kv_ratio=0.3008
for the same model. Which is the "best" result?

- amort_g8_S8_k3: lower kv_ratio (0.2969) but uses scale
  amortization (extra hyperparameter, not standard INT4)
- g32_k2: slightly higher kv_ratio (0.3008) but uses only
  standard INT4/INT8 (simpler, more reproducible)

**Resolution**: The headline claim uses g32_k2 for simplicity. The
amortized variant is mentioned as a further optimization pathway but
is not part of the central claim. The empirical lower boundary is
reported as the g32 result.

### Issue 6: GPU differences

v24 results were on W7900 (AMD, ROCm). v26 results were on H100
(NVIDIA, CUDA). The quantization is simulated (quantize-dequantize
in Python), so GPU type does not affect the numerical outcome. The
same model, same data, same protocol produces the same PPL.

The only GPU-dependent metrics are latency and throughput, which are
reported separately and explicitly labeled by GPU.

**Resolution**: No reconciliation needed for PPL-based claims. GPU
is recorded in the canonical table for provenance.

## Summary of Canonical Choices

1. Quantization: standard g32 INT4/INT8 (symmetric, no amortization)
2. Ranking: oracle (per-layer ablation, 3 seeds, L=8K)
3. Evaluation: L in {8192, 32768}, 3 seeds, 64 continuation tokens
4. Pass criterion: max |delta_pct| <= eps across all (L, seed)
5. Cache: W_sink=4, W_min=1024, far region compressed
6. Dataset: wikitext-2-raw-v1
