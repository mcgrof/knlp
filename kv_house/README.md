# kv_house

Prefix-stable sequence-axis transform coding of sealed KV-cache
blocks. The research motivation, prior art, lanes, hypotheses, and
milestone gates are in [docs/kv-house/](../docs/kv-house/); this
file is the operational map. Milestone 1 ran and returned NO_GO —
the measured reasons are in
[docs/kv-house/results-summary.md](../docs/kv-house/results-summary.md);
the package remains useful for its sealed-block codec, PIA
adapter, and temporal-structure measurement tooling.

## Layout

- `transforms.py` — temporal transforms along the token axis of a
  [B, d] block: identity, CacheGen-style anchor-delta, DC /
  random / power-iteration Householder reflectors, DCT-II, Haar,
  flat Hadamard, per-block PCA oracle, corpus KLT.
- `quant.py` — per-mode symmetric quantizer and byte accounting
  (payload + scales + transform metadata + header).
- `temporal_stats.py` — token-axis structure metrics with
  shuffled-token controls.
- `attention_sim.py` — attention-damage metrics from swapping a
  reconstructed block into a full-precision cache.
- `prefix_codec.py` — the sealed-block codec: deterministic,
  suffix-independent, randomly accessible artifacts whose cache
  key includes the codec configuration.
- `pia_adapter.py` — Prefix Integrity Analysis adapter
  (`--algorithm kv_house.pia_adapter:make`).

## Tests

Gate-0 math and contract tests run on CPU with no model:

```bash
python3 -m pytest tests/kv_house/ -q
```

They cover Householder construction, the transformed-attention
identity for every orthogonal lane, lossless inversion, encode
determinism, suffix independence, raw partial tails, and block
random access.

## Experiments

```bash
python3 experiments/kv_house/capture_kv.py --out-dir CAP ...
python3 experiments/kv_house/measure_temporal_structure.py \
    --capture-dir CAP --out structure.csv
python3 experiments/kv_house/run_transform_ablation.py \
    --capture-dir CAP --out ablation.jsonl
python3 experiments/kv_house/run_milestone1.py \
    --structure-csv structure.csv --ablation-jsonl ablation.jsonl \
    --capture-dir CAP --out verdict.json
```

Capture records post-RoPE K/V, strided post-RoPE Q probes, and
pre-RoPE K per layer, using the runtime attention discovery in
`tools/kv/k_bias_common.py` (requires `attn_implementation="sdpa"`).
