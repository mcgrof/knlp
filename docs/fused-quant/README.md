# Fused INT4 KV-cache attention (Triton)

A fused Triton decode kernel that stores keys and values in packed INT4 (two
values per byte) with per-group FP16 scales, and reconstructs them inside the
attention tile loop. No full FP16 KV intermediate is written to global memory.

The kernel is a controlled experiment that separates fusion from bit-width
reduction. A non-fused INT4 path that writes a temporary FP16 buffer before
calling standard attention is slower than FP16 at the same shape. The fused
path reaches 2.7–4.8x speedup over PyTorch SDPA on H100 and 1.6–7.2x on W7900.
The result is that bit-width reduction only improves decode when the kernel
does not recreate the saved traffic in a larger intermediate format.

The production follow-on lives in FlashInfer as independent key/value dtype
support. See [FlashInfer asymmetric path](#flashinfer-asymmetric-path).

## Table of contents

- [Status and versions](#status-and-versions)
- [Latency analysis](#latency-analysis)
- [FP8 hardware and tile-path notes](#fp8-hardware-and-tile-path-notes)
- [FlashInfer asymmetric path](#flashinfer-asymmetric-path)
- [Paper reference](#paper-reference)
- [Collaboration pointers](#collaboration-pointers)
- [Lineage](#lineage)

## Status and versions

Three versioned kernel files live under [`fused-quant/`](../../fused-quant/):

- `fused_int4.v0.0.1.py` — first canonical version that ran the corrected
  Marin 8B LongBench rerun cleanly.
- `fused_int4.v0.0.2.py` — CUDA-graph support fix. The fused backend had
  `AttentionCGSupport.NEVER`, restricting it to PIECEWISE capture while FP16
  used FULL+PIECEWISE. Restoring `UNIFORM_SINGLE_TOKEN_DECODE` closed about
  86% of the resulting decode-heavy latency gap. Production-validated.
- `fused_int4.v0.1.0.py` — experimental K/V precision-split paths, gated by
  environment flags. The base INT4/INT4 path is unchanged. This branch was
  superseded by FP16-K / FP8-V work in FlashInfer.

`fused_int4.py` points to the current default, v0.1.0.

## Latency analysis

[`latency.md`](latency.md) is the H100 latency review of v0.0.2. It decomposes
the remaining decode-256 gap into K-side scale broadcast, nibble unpacking,
the even/odd Q split, and the separate cache-write kernel. It also records the
important crossover: the custom fused path is not the fastest choice at every
batch and context, but becomes valuable once cache traffic is large enough to
amortize its fixed and on-chip costs.

## FP8 hardware and tile-path notes

[`fp8-attention-hardware-notes.md`](fp8-attention-hardware-notes.md) describes
what Hopper, Blackwell, and CDNA 3 actually expose to an FP8 attention kernel.
The current summary is:

- fused kernels keep the compressed cache in HBM; they do not need to write a
  dequantized cache back to HBM;
- online softmax works tile by tile, so there is no single all-score barrier;
- K operand preparation must finish before that tile's QK and softmax update;
- V preparation may overlap with the PV pipeline but is not free;
- CDNA 3 FP8 MFMA has no scale operands; current AITER applies scale outside
  the MFMA; and
- Blackwell hardware block scaling is real, but uses narrow/block-scaled Q and
  K rather than ordinary BF16 Q with a plain E4M3 K cache.

The companion
[`fp8-attention-tile-path.html`](../fp8-attention-tile-path.html) states the
performance hypothesis and the microtests needed to prove or reject it. The
central comparison is simple: does K8/V8 save more HBM time than it adds in
unhidden K-tile preparation and pipeline pressure, relative to K16/V8?

## FlashInfer asymmetric path

A separate line of work modifies FlashInfer to accept independent key and
value dtypes in paged decode and prefill kernels. K16/V8 stores K in the
model's native 16-bit type and V in FP8 E4M3.

On the measured H100 workloads, K16/V8 matches FP16 quality and can outperform
ordinary symmetric K8/V8. The tile-path hypothesis is that K16/V8 trades one
extra K byte per element for removal of K8 operand preparation before QK.
That explanation is plausible and consistent with the result, but it is not
yet a controlled causal measurement. V conversion also remains real work,
even when a kernel hides much of it behind the PV pipeline.

The public FlashInfer work is at <https://github.com/mcgrof/flashinfer>. The
latest upstream-oriented branches referenced by the paper are:

- `20260702-asym-k16v8-decode-upstream` — decode-only K16/V8 dtype split,
  rebased for an upstream PR series;
- older development branches such as `asym-prefill-refactor-stage` remain
  useful for reproducing the original full-stack result.

The fused INT4 Triton kernel and the FlashInfer branch answer different
questions. The Triton kernel proves that avoiding a global-memory intermediate
matters. The FlashInfer work asks which K/V storage and tile schedule wins in a
production paged-attention stack.

## Symmetric FP8 is back on the table

The [FP8 KV-cache failure atlas](../fp8-kv-failure-atlas.html) root-caused the
Qwen K8 failure. The key-projection bias captures the scale and crushes the
token-varying residual. Quantizing the pre-bias residual while keeping the
fixed bias exact makes symmetric K8/V8 near-lossless in the tested dynamic and
static scale layouts.

That changes the engineering question. K16/V8 is no longer the only plausible
quality-safe FP8 design for biased Qwen models. A bias-aware K8/V8 path can win
on capacity, and may win on speed if its tile preparation is fixed or moved to
a native block-scaled path. The remaining work is a real serving-kernel test,
not another argument about whether ordinary post-bias K8 fails. It does.

## Paper reference

The result set appears in *Memory-Traffic Saturation in Autoregressive
Transformer Decode*. The custom INT4 kernel is the fusion-control experiment.
The FlashInfer K16/V8 path is the practical serving result. The new tile-path
page narrows the next claim to a falsifiable kernel hypothesis rather than the
old, incorrect slogan that K dequant is globally serial and V dequant is free.

## Collaboration pointers

Useful cross-vendor work should compare actual instruction and kernel paths:

- Hopper BF16-Q/plain-FP8-K tile transformation;
- Blackwell BF16-Q/FP8-K transform modes;
- Blackwell MXFP8 block-scaled QK;
- CDNA 3 FP8-by-FP8 MFMA with software scale placement; and
- K16/V8 as the no-K-transform control.

AMD's native FP8 MFMA is relevant, but it should not be described as accepting
FP8 scale operands. NVIDIA's block-scaled Blackwell path does accept hardware
scale factors, but requires the corresponding narrow/block-scaled operand and
scale layouts. These details are precisely why comparing vendor marketing
labels is less useful than comparing the selected kernels and profiler traces.

## Lineage

[`lineage/`](lineage/) preserves dated notes from v0.0.1 through v0.1.0. The
current summaries live here, in [`latency.md`](latency.md), and in the
[tile-path hypothesis page](../fp8-attention-tile-path.html).
