# Bandwidth-Proportional Attention (BPA)

Bandwidth-Proportional Attention (BPA) explores transformer inference
architectures where the amount of KV cache memory accessed per token
scales with available memory bandwidth rather than full context length.

This is an active research program within knlp, not a finalized
architecture.

## Motivation

Autoregressive transformer decode repeatedly reads the entire KV cache
to generate each output token. Decode latency therefore scales linearly
with KV cache size, which grows with context length and batch size.

Measurements across three accelerator architectures (AMD W7900, NVIDIA
H100, NVIDIA B200) spanning a 7.6x bandwidth range confirm that decode
performance is governed by memory bandwidth, not compute capacity.
Throughput at batch size 1 scales roughly proportionally to peak memory
bandwidth across all three GPUs.

Compression (INT4 quantization) halves the KV traffic and delivers
2.7-4.8x decode speedup when fused into the attention kernel. But
compression alone cannot eliminate the fundamental linear scaling of
decode cost with context length. As context windows grow to 128K+
tokens, even compressed KV caches saturate available bandwidth.

The BPA research direction asks: instead of reading less data per KV
entry (compression), can we read fewer KV entries per decode step?

## Research Questions

* Can the number of KV entries read per decode step be reduced without
  harming model quality?
* Can KV memory be tiered across precision levels, with bandwidth
  allocated proportionally to layer sensitivity?
* Can attention operate within a fixed bandwidth budget that adapts to
  the hardware rather than the context length?
* Does selective KV access (reading a subset of layers or tokens)
  preserve the information needed for accurate generation?
* How does the minimum number of protected high-precision layers (k*)
  scale with model depth?

## Current Experiments in knlp

### KV Bandwidth Scaling

Decode throughput and latency measurements across batch sizes and
context lengths on W7900, H100, and B200. Confirms the memory-bound
regime and fits a Hill-type saturation model.

### KV Cache Quantization

INT4 quantization of KV cache with per-group scales. Fused Triton
kernel eliminates intermediate buffer writes. Non-fused pipelines
are counterproductive (slower than FP16 baseline).

### Selective Layer Protection

Empirical identification of k* (minimum INT8-protected layers for
quality preservation). Key finding: k* is bounded by a small
constant independent of model depth (O(1) scaling hypothesis).
Layer 0 (attention sink) is consistently the most sensitive.

### KV Tiering

Mixed-precision KV cache where sensitive layers use INT8 and
remaining layers use INT4. Achieves sub-0.30 KV ratios while
maintaining quality within 3% of FP16 baseline.

### Extreme Context Scaling

Context length push experiments up to 384K tokens on B200 (178 GB
HBM). Establishes HBM capacity as the binding constraint for
long-context inference.

## Experiment Data

Interactive visualizations of BPA experiment results are available at:

[KV Bandwidth Visualization](https://mcgrof.github.io/knlp/kv_bandwidth_visualization.html)

Raw experiment data is archived in the knlp-key-results repository
under `bpa/b200_campaign/`.

## Status

BPA is ongoing research. Results are preliminary and subject to
revision as experiments continue across additional model architectures
and hardware platforms.
