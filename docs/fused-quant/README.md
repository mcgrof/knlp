# Fused KV-cache quantization

This directory covers two implementation lines:

- a custom Triton INT4 decode kernel that reconstructs packed K/V tiles inside
  attention; and
- independent K/V cache dtypes in FlashInfer, including K16/V8 and the
  symmetric-FP8 kernel work that follows from the FP8 failure atlas.

Use the custom INT4 line to measure the value of fusion. Use the FlashInfer
line to compare production paged-cache formats and tile schedules.

## Table of contents

- [Custom Triton INT4 line](#custom-triton-int4-line)
- [Latency analysis](#latency-analysis)
- [FP8 hardware and tile path](#fp8-hardware-and-tile-path)
- [FlashInfer K16/V8](#flashinfer-k16v8)
- [Bias-aware symmetric FP8](#bias-aware-symmetric-fp8)
- [Measurement rules](#measurement-rules)
- [Paper and source pointers](#paper-and-source-pointers)

## Custom Triton INT4 line

The fused Triton kernel stores K and V in packed INT4, with two values per
byte and per-group FP16 scales. It reconstructs each active tile inside the
attention loop and avoids a full FP16 KV intermediate in global memory.

A matched non-fused path writes an FP16 intermediate before standard
attention and runs slower than FP16 at the same shape. The fused path reaches
2.7–4.8x speedup over PyTorch SDPA on H100 and 1.6–7.2x on W7900 across the
reported sweep. Treat this as the fusion control: compression helps only when
the implementation preserves the saved memory traffic.

Versioned kernels live under [`fused-quant/`](../../fused-quant/):

- `fused_int4.v0.0.1.py` — canonical baseline;
- `fused_int4.v0.0.2.py` — CUDA-graph support through
  `UNIFORM_SINGLE_TOKEN_DECODE`; and
- `fused_int4.v0.1.0.py` — experimental K/V precision splits.

`fused_int4.py` points to v0.1.0.

## Latency analysis

[`latency.md`](latency.md) decomposes the H100 decode-256 gap into:

- K-side scale broadcast;
- nibble unpacking;
- the even/odd Q split;
- accumulator pressure; and
- the separate cache-write kernel.

Use batch and context sweeps to find the crossover where cache-traffic savings
outweigh launch, preparation, and occupancy costs.

## FP8 hardware and tile path

[`fp8-attention-hardware-notes.md`](fp8-attention-hardware-notes.md) maps the
relevant Hopper, Blackwell, and CDNA 3 instruction paths.

Apply these facts:

- keep compressed K/V in HBM and reconstruct active tiles on chip;
- advance online softmax tile by tile;
- complete K preparation before QK for each tile;
- complete V preparation before PV for each tile;
- measure overlap for the selected kernel and shape;
- treat CDNA 3 FP8 MFMA scale placement as software work around an instruction
  with no scale operands; and
- use Blackwell block-scaled MMA with the narrow Q/K and scale layouts required
  by the hardware.

The K8/V8 versus K16/V8 measurement plan summarized in the
[`bias-aware KV deployment policy`](../bias-aware-kv-quantization.html) has
been run. The answer depends on the kernel family. On the CUDA-core
decode kernel, conversion cost dominates: every FP8 cell is slower than
BF16 there, and K16/V8 beats K8/V8. On the SM90 Tensor Core kernel with
the split-dtype prefill kernels, bytes dominate: symmetric K8/V8 wins by
16-19% over K16/V8 at bandwidth-bound shapes, and both beat BF16
(K16/V8 by 1.19-1.28x; tile-path result commit f75960a6). FP8 loses
only at batch 1.
Any speed claim about these formats must name the kernel it was measured
on.

## FlashInfer K16/V8

The FlashInfer dtype-split work stores K in the model's native 16-bit type and
V in FP8 E4M3. This produces 1.33x KV-cache capacity relative to 16-bit K/V.

Use K16/V8 as:

- the quality-safe fallback for hostile key distributions (measured in
  serving: symmetric FP8 collapses Qwen2.5-7B GSM8K from 90.5% to 2.0%,
  while K16/V8 measures 90.0% against the 90.5% FP16 baseline — n=200,
  8-shot, a screen that did not resolve a difference);
- the no-K-transform control; and
- the capacity and quality operating point on the CUDA-core decode
  path, where it beats K8/V8 — though every FP8 cell, K16/V8 included,
  trails BF16 on that kernel.

On the Tensor Core kernel K16/V8 is not the fast FP8 option: symmetric
K8/V8 beats it by 16-19% at bandwidth-bound shapes. Its remaining role
there is quality, and it still beats BF16 while carrying that safety.
Historical note: early serving numbers showed asymmetric decode slower
than BF16 because a dispatch guard barred asymmetric caches from the
Tensor Core kernel; the guard lift routed asymmetric decode onto Tensor
Cores and the kernel then beat BF16 by about 28%, serving at BF16
parity with 1.33x cache capacity.

Public source lives at <https://github.com/mcgrof/flashinfer>. The
upstream-oriented decode branch is
`20260702-asym-k16v8-decode-upstream` (branch tip 2f3f0f6d). The
corresponding vLLM branch is
<https://github.com/mcgrof/vllm/tree/20260702-k16fp8> (branch tip
758786e23). The validated serving-gate stack is pinned at vLLM fork
commit 8a1714108 and FlashInfer fork commit 6dfdc833.

## Bias-aware symmetric FP8

The [FP8 KV-cache failure atlas](../fp8-kv-failure-atlas.html) identifies a
Qwen key failure mechanism: the key-projection bias captures the FP8 scale and
crushes the token-varying residual.

Implement symmetric K8/V8 for this class by:

1. quantizing the pre-bias K residual;
2. keeping the fixed K bias exact;
3. reconstructing the bias before QK or applying the equivalent score-space
   correction; and
4. validating long-context autoregressive quality.

The tested dynamic and static scale layouts make this representation
quality-admissible on Qwen2.5-7B. The serving stakes are now measured on
both sides: symmetric K8/V8 is the fastest cache on the Tensor Core
kernel, and ordinary post-bias symmetric FP8 destroys Qwen2.5 reasoning
in serving, while on a biasless model no adverse quality effect was
measured (Qwen3-8B needle 1.00; its GSM8K screen was uninformative for
prompt-format reasons and cannot establish equivalence).

The pre-bias representation would let biased-K models use the fastest
cache, and its serving cost is measured; the question is closed. The
kernel costs 1.69x K16/V8 latency at 4K context and 1.11x at 32K (five
processes, Latin square, CIs exclude parity), and the equal-memory
serving gate failed at 0.654x completed req/s and 2.056x p95 latency
against frozen bounds of 1.20x and 1.25x (pre-bias at 48 requests vs
K16/V8 at 32, 30,720 in / 128 out, Qwen2.5-7B, H100; result commit
85fa7b2d, countersigned c82c5db9). The pre-bias kernel-optimization line
is permanently closed. Pre-bias remains a documented capacity option
only (1.50x K16/V8 capacity, 1,893,440 tokens measured), never the
serving default — see the bias-aware KV deployment policy.

The measured cost comprised:

- per-tile K operand preparation;
- bias reconstruction or score correction;
- RoPE placement;
- scale storage and broadcast;
- cache layout conversion; and
- occupancy.

Retain K16/V8 for partial-RoPE and other key distributions that fail the
symmetric quality gate.

## Measurement rules

Run the four dtype controls:

```text
K16/V16
K8/V16
K16/V8
K8/V8
```

For Blackwell BF16-Q/FP8-KV, compare:

```text
Full
KOnly
SeparateKv
```

Note: on B200 trtllm-gen the k_only mode returns wrong output —
validate outputs before timing it — and fp8 there measured nearly flat
(~1.05x).

Add FP8-Q × FP8-K and MXFP8 QK as native narrow controls. Run L2-warm and
HBM-cold variants. Sweep batch, context, head dimension, GQA ratio, and page
size.

Record:

- exact FlashInfer commit;
- JIT URI and cubin hash;
- selected transform mode;
- kernel latency and tile interval;
- DRAM and L2 traffic;
- conversion and matrix instruction counts;
- dependency, barrier, and scoreboard stalls;
- register and shared-memory use;
- achieved occupancy;
- ITL and throughput;
- KV capacity and admitted concurrency; and
- quality metrics for every cache representation.

Choose the serving format from quality, kernel time, capacity, and goodput.

## Paper and source pointers

- [`bias-aware-kv-quantization.html`](../bias-aware-kv-quantization.html) —
  deployment policy, pre-bias conclusion, and kernel-family boundary.
- [`kv-compression-frontier.html`](../kv-compression-frontier.html) — measured
  quality, capacity, and serving comparison across KV formats.
- [`fp8-attention-hardware-notes.md`](fp8-attention-hardware-notes.md) — ISA
  and kernel-path details.
- [`latency.md`](latency.md) — custom INT4 latency decomposition.
- [`lineage/`](lineage/) — dated kernel provenance.
- [`fp8-kv-failure-atlas.html`](../fp8-kv-failure-atlas.html) — numerical
  failure mechanisms and repair gates.
- [`paper_memory_decode.html`](../paper_memory_decode.html) —
  *Memory-Traffic Saturation in Autoregressive Transformer Decode*.
