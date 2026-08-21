# Fused KV quantization index

Use this page to navigate the fused KV-cache quantization work.

## Code

### Custom Triton kernels

- [`fused-quant/fused_int4.v0.0.1.py`](../fused-quant/fused_int4.v0.0.1.py)
- [`fused-quant/fused_int4.v0.0.2.py`](../fused-quant/fused_int4.v0.0.2.py)
- [`fused-quant/fused_int4.v0.1.0.py`](../fused-quant/fused_int4.v0.1.0.py)

### FlashInfer and vLLM K/V dtype split

- <https://github.com/mcgrof/flashinfer>
- <https://github.com/mcgrof/vllm/tree/20260702-k16fp8>

## Documentation

- [`docs/fused-quant/README.md`](fused-quant/README.md) — implementation
  overview and measurement rules
- [`docs/fused-quant/latency.md`](fused-quant/latency.md) — H100 INT4
  latency decomposition
- [`docs/fused-quant/fp8-attention-hardware-notes.md`](fused-quant/fp8-attention-hardware-notes.md)
  — Hopper, Blackwell, and CDNA 3 instruction and tile paths
- [`docs/fp8-attention-tile-path.html`](fp8-attention-tile-path.html) — K8/V8
  versus K16/V8 microtests, profiler matrix, and decision rules
- [`docs/fp8-kv-failure-atlas.html`](fp8-kv-failure-atlas.html) — numerical
  failure mechanisms and bias-aware symmetric FP8
- [`docs/fused-quant/lineage/`](fused-quant/lineage/) — dated provenance

## Paper

- [`knlp.io/decode`](https://knlp.io/decode) — *Memory-Traffic Saturation in
  Autoregressive Transformer Decode*

## Keep the three gates separate

1. **Fusion gate:** verify that the kernel avoids a global-memory
   reconstruction round trip.
2. **Quality gate:** verify that the K/V representation preserves model
   behavior.
3. **Schedule gate:** verify that tile preparation and occupancy cost less than
   the HBM traffic saved.

Require all three gates before selecting a serving format.
