# Fused INT4 KV-cache attention (Triton)

A fused Triton decode kernel that stores keys and values in packed
INT4 (two values per uint8 byte) with per-group FP16 scales, and
dequantizes in-register inside the attention loop so no full FP16
KV intermediate is ever materialized.  It exists as a controlled
experiment that isolates the effect of kernel fusion from the
effect of bit-width reduction: a non-fused INT4 path that writes
a temporary FP16 buffer before calling standard attention is
*slower* than FP16 at the same shape, while the fused kernel
achieves 2.7–4.8× speedup over PyTorch SDPA on H100 and
1.6–7.2× on W7900.  That result grounds the broader claim that
memory-traffic reduction is what matters for decode, not
bit-width per se.

The work is one lineage inside a larger KV-cache-quantisation
programme.  The follow-on production path lives in FlashInfer as
independent branches — see [FlashInfer asymmetric FP16-K /
FP8-V](#flashinfer-asymmetric-path) below.

## Table of contents

- [Status and versions](#status-and-versions)
- [Latency analysis](#latency-analysis)
- [FlashInfer asymmetric path](#flashinfer-asymmetric-path)
- [Paper reference](#paper-reference)
- [Collaboration pointers](#collaboration-pointers)
- [Lineage](#lineage)

## Status and versions

Three versioned kernel files live at the repo root under
[`fused-quant/`](../../fused-quant/):

- `fused_int4.v0.0.1.py` — first canonical version that ran the
  corrected Marin 8B LongBench rerun cleanly.
- `fused_int4.v0.0.2.py` — CUDA-graph support fix: the fused
  backend previously had `AttentionCGSupport.NEVER`, restricting it
  to PIECEWISE-only capture while FP16 used FULL+PIECEWISE.
  Restoring `UNIFORM_SINGLE_TOKEN_DECODE` closed ~86% of the
  resulting decode-heavy latency gap.  Production-validated.
- `fused_int4.v0.1.0.py` — experimental asymmetric K/V precision-
  split paths (INT8-K + INT4-V, INT8-K + INT8-V, FP16-K + INT8-V),
  gated by environment flags, base INT4/INT4 path unchanged.  No
  production validation archived; the precision-split idea was
  eventually superseded by FP16-K / FP8-V via FlashInfer.

`fused_int4.py` is a symlink to the current default (v0.1.0).

## Latency analysis

[`latency.md`](latency.md) is the detailed H100 latency review
of v0.0.2.  It decomposes the residual ~2.8% ITL gap vs FP16 at
decode-256 into four contributors (K-side scale broadcast loop,
nibble unpacking arithmetic, even/odd Q split with dual
accumulator, separate cache-write kernel), lists six follow-up
ideas ranked by expected gain, and records the verdict that the
residual gap is acceptable for production because fused INT4's
value proposition is memory capacity (2× more concurrent
sequences) and long-context latency (3–4× at 32K tokens), not
short-context decode speed.

## FlashInfer asymmetric path

A separate line of work modifies
FlashInfer~([ye2024flashinfer](https://arxiv.org/abs/2501.01005))
to accept independent key and value data types on the paged
decode and prefill kernels.  With K stored in FP16 and V in
FP8-e4m3, the softmax serial critical path avoids K
dequantization entirely while V dequant overlaps with the
reduction pipeline.  On H100 vLLM at batch 32, asymmetric K16/V8
matches or exceeds FP16 decode throughput on every tested model
and reaches 1.38× FP16 on Qwen2.5-7B, while providing 1.33×
KV cache capacity.  It also rescues fragile-key models where
symmetric FP8 collapses: Qwen2.5-7B NIAH retrieval at 32K drops
from 0.84 (FP16) to 0.00 under symmetric FP8 and recovers to
0.89 under asymmetric K16/V8.

That work lives in its own FlashInfer fork rather than in this
repo, as two topic branches:

- `asymmetric-kv-dtype` — the FP16-K / FP8-V production path.
- `asymmetric-kv-int4v` — later evolution adding INT4-V on top
  of the FP16-K path.

Both branches are intended for public release.  The fused INT4
Triton kernel documented here and the FlashInfer branches are
peers in the same programme: the Triton kernel proves the
fusion mechanism on a custom path; FlashInfer applies the same
principle in a production stack.

## Paper reference

The result set described here is published in *Memory-Traffic
Physics of Autoregressive Decode*.  This kernel appears as the
controlled mechanism proof (fused-vs-non-fused INT4 on identical
data and scales); the FlashInfer asymmetric path is the paper's
main practical contribution.

## Collaboration pointers

This work naturally pairs with vendor-level optimisation of
FP8 attention.  AMD's AITER backend already implements the
fusion principle at hardware level via native FP8×FP8 WMMA
instructions, which absorb the FP8 scales into softmax
normalisation and eliminate K dequantisation from the critical
path entirely — a hardware-level solution to the same
bottleneck the FlashInfer asymmetric branch solves in software.
A joint evaluation on MI300X / MI350 against the NVIDIA
reference results is a natural follow-up.

## Lineage

[`lineage/`](lineage/) archives the dated provenance notes
from the development of v0.0.1 → v0.1.0 (kernel import from the
H100 execution lane, CUDA graph fix, residual-gap profiling
runs).  They are preserved for traceability; the current
summary lives in this README and in [`latency.md`](latency.md).
