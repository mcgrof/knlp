# Fused KV Quantization

> **This page has moved.** The canonical entry point is now
> [`docs/fused-quant/README.md`](fused-quant/README.md).

## Current entry points

**Code, custom Triton line**

- [`fused-quant/fused_int4.v0.0.1.py`](../fused-quant/fused_int4.v0.0.1.py)
- [`fused-quant/fused_int4.v0.0.2.py`](../fused-quant/fused_int4.v0.0.2.py)
- [`fused-quant/fused_int4.v0.1.0.py`](../fused-quant/fused_int4.v0.1.0.py)

**Code, FlashInfer K/V dtype split**

- <https://github.com/mcgrof/flashinfer>
- <https://github.com/mcgrof/vllm/tree/20260702-k16fp8>

**Documentation**

- [`docs/fused-quant/README.md`](fused-quant/README.md) — current overview
- [`docs/fused-quant/latency.md`](fused-quant/latency.md) — H100 INT4 latency
- [`docs/fused-quant/fp8-attention-hardware-notes.md`](fused-quant/fp8-attention-hardware-notes.md)
  — corrected Hopper, Blackwell, and CDNA 3 hardware notes
- [`docs/fp8-attention-tile-path.html`](fp8-attention-tile-path.html) — the
  K8/V8 versus K16/V8 performance hypothesis and target microtests
- [`docs/fp8-kv-failure-atlas.html`](fp8-kv-failure-atlas.html) — the numerical
  failure mechanisms and the pre-bias symmetric-FP8 repair
- [`docs/fused-quant/lineage/`](fused-quant/lineage/) — dated provenance notes

**Paper**

- [`knlp.io/decode`](https://knlp.io/decode) — *Memory-Traffic Saturation in
  Autoregressive Transformer Decode*

## Why the old page was retired

The previous version mixed the custom INT4 experiment, serving-stack status,
paper claims, and hardware interpretation into one long document. The split
above keeps three different questions separate:

1. Does fusion avoid a global-memory round trip?
2. Is a K/V representation numerically safe?
3. Does the selected tile schedule beat the bytes it saves?

Those questions interact, but they are not interchangeable. Keeping them
separate prevents one measurement from being attributed to the wrong
mechanism.
