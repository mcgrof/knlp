# BPA v27: Central Claim and Scope

## Claim Box

Under mixed-precision KV cache quantization (INT4/INT8, group
size 32, symmetric), the number of transformer layers requiring
high-precision (INT8) protection to maintain perplexity within
epsilon=3% of the dense baseline is bounded by a small constant k*
that does not grow with model depth D.

Observed:

    k*(3%) = {2, 2, 2, 0, 0}
    D      = {24, 28, 28, 32, 32}

across 5 models (Qwen2.5-0.5B/1.5B/7B, Mistral-7B-v0.1, Llama-2-7b),
3 architecture families (Qwen2, Mistral, Llama), and 2 GPU platforms
(AMD W7900, NVIDIA H100).

This yields kv_ratio in [0.28, 0.30] and a theoretical 3.3-3.6x
concurrent sequence capacity at <3% PPL cost, assuming KV cache
is the memory bottleneck.

Two distinct mechanisms underlie the bounded k*:

1. **Sink dominance** (Qwen family): Layer 0 accumulates 67-99.99%
   of total INT4 sensitivity. Protecting 1-2 sink layers is
   sufficient. The tail of D-2 unprotected layers contributes
   <3% total error.

2. **Uniform robustness** (Mistral, Llama-2): Per-layer INT4
   sensitivity is uniformly low (max 0.48% Mistral, 1.18% Llama).
   Even all-INT4 (k=0) accumulates <1.1% total error. No
   protection needed. Llama-2-7b with MHA (32 KV heads) shows the
   same uniform pattern with no dominant sink layer.

## Limitations Box

1. **Finite D range**: D in {24, 28, 32}. We do not claim O(1)
   asymptotically. The claim is empirical over the tested range.

2. **Oracle ranking only**: k* values use a per-model oracle
   ranking (D forward passes per model). No proxy ranking achieves
   the same k*. Practical deployment requires either oracle
   calibration (one-time cost) or a transfer procedure.

3. **No fused INT4 kernel**: All results use simulate-quantize-
   dequantize in Python with fp16 storage. Actual latency savings
   require a fused INT4 attention kernel (e.g., FlashAttention
   with INT4 KV support). Latency criterion C (>=10% speedup) is
   NOT MET.

4. **Two architecture families**: Qwen2 (GQA, 2-4 KV heads) and
   Mistral (GQA, 8 KV heads). No Llama, no MHA architectures,
   no models above 7.6B parameters.

5. **Wikitext evaluation only**: All PPL measurements use
   wikitext-2-raw-v1. Task-specific evaluation (MMLU, HumanEval,
   etc.) is not included.

6. **Fixed group size**: g=32 only. Smaller groups (g=4, g=8)
   have different scale overhead and may yield different k*.

7. **Greedy decoding**: All evaluations use greedy decoding with
   64 continuation tokens. Beam search or sampling may expose
   different sensitivity patterns.

## The L-squared-M Empirical Lower Boundary

### Definition

The **empirical lower boundary** (ELB) is the lowest kv_ratio
achieved under PASS_eps across all tested configurations, within a
defined family of mixed-precision mechanisms, on a specific model
under the canonical evaluation protocol.

    ELB(model, eps) = min_{k, ranking, scheme} kv_ratio
                      s.t. PASS_eps(model, k, ranking, scheme)

It is NOT an information-theoretic lower bound. It is the best
ratio we have achieved, not the best ratio that can be achieved.

### Observed ELB Values

| Model | eps | ELB (g32 oracle) | Scheme |
|-------|-----|-------------------|--------|
| Qwen2.5-0.5B | 3% | 0.3008 | g32 k=2 |
| Qwen2.5-1.5B | 3% | 0.2974 | g32 k=2 |
| Qwen2.5-7B | 3% | 0.2974 | g32 k=2 |
| Mistral-7B | 3% | 0.2812 | g32 k=0 |

### Connection to Asymptotic Behavior

If k* remains bounded while D increases:
- The protected fraction k*/D approaches zero.
- The kv_ratio approaches the all-INT4 floor:
  kv_ratio_floor = int4_layer / dense_layer (independent of D).
- For g=32, head_dim=128: kv_ratio_floor = 0.28125.

The empirical lower boundary is therefore approaching the all-INT4
floor from above. The gap between ELB and floor is determined by
the number of protected layers and the scale overhead.

This is an empirical route toward an L-squared-M-style lower-bound
story: if mixed-precision with bounded protection is the mechanism,
and if the protection count does not grow with depth, then the
effective KV ratio has a hard floor determined solely by the
quantization bitwidth and group size.

### What This Does Not Mean

- It does not mean kv_ratio=0.28 is the minimum possible. Smaller
  group sizes, non-symmetric quantization, learned quantization, or
  entirely different compression families (e.g., MLA) could achieve
  lower ratios.
- It does not mean INT4 is optimal. Other bitwidths (INT3, INT2)
  or floating-point formats (FP4, NF4) are not tested.
- It does not mean the O(1) property extends to arbitrary D. We
  observe it over D in {24, 28, 32} and hypothesize it continues,
  but this is not proven.
