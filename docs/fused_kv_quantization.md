# Fused KV Quantization

> **This page has moved.** The canonical entry point for this work is
> now [`docs/fused-quant/README.md`](fused-quant/README.md), which
> covers the Triton fused INT4 kernel lineage, the FlashInfer
> asymmetric FP16-K / FP8-V production path, and the FP8 Tensor Core /
> matrix-core ISA constraints that shape both.

## Where to find things now

The fused KV quantization work lives across a repository and two
document areas:

**Code (Triton kernel line)**
- [`fused-quant/fused_int4.v0.0.1.py`](../fused-quant/fused_int4.v0.0.1.py)
  — first canonical version
- [`fused-quant/fused_int4.v0.0.2.py`](../fused-quant/fused_int4.v0.0.2.py)
  — CUDA-graph support fix, production-validated
- [`fused-quant/fused_int4.v0.1.0.py`](../fused-quant/fused_int4.v0.1.0.py)
  — experimental asymmetric K/V precision-split paths

**Code (FlashInfer asymmetric, production path)**
- <https://github.com/mcgrof/flashinfer> — public fork with two
  branches:
  - [`asymmetric-kv-dtype`](https://github.com/mcgrof/flashinfer/tree/asymmetric-kv-dtype)
    — FP16-K / FP8-V, matches or beats FP16 decode throughput on
    every tested model, rescues fragile-key models where symmetric
    FP8 collapses
  - [`asymmetric-kv-int4v`](https://github.com/mcgrof/flashinfer/tree/asymmetric-kv-int4v)
    — later evolution with INT4-V on top of the FP16-K path

**Documentation**
- [`docs/fused-quant/README.md`](fused-quant/README.md)
  — top-level entry describing both lineages and how they relate
- [`docs/fused-quant/latency.md`](fused-quant/latency.md)
  — H100 v0.0.2 latency deep-dive
- [`docs/fused-quant/fp8-attention-hardware-notes.md`](fused-quant/fp8-attention-hardware-notes.md)
  — Tensor Core / matrix-core FP8 ISA constraints across NVIDIA
  Hopper, NVIDIA Blackwell, and AMD CDNA 3
- [`docs/fused-quant/lineage/`](fused-quant/lineage/)
  — dated provenance notes

**Paper**
- [`knlp.io/decode`](https://knlp.io/decode) — *Memory-Traffic
  Saturation in Autoregressive Transformer Decode* — interactive
  landing page with the main findings

## Why this page was retired

The previous version of this document dated from 2026-03-29 and
predated the FlashInfer asymmetric FP16-K / FP8-V production path
that is now the paper's main result.  It also mixed paper-summary
material, fused-INT4 Triton kernel implementation notes, H100
dispatch policies, vLLM branch status, and an A100 serving
debugging log into one file, which made it hard for a reader to
find the current state.  Splitting the material into the
purpose-specific locations above means each reader lands closer to
the content they actually want.
