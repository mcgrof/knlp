# BPA v28: H100 Appendix

## Hardware Configuration

All headline results were produced on a single NVIDIA H100
80GB HBM3 GPU (hostname: rqv).

| Parameter | Value |
|-----------|-------|
| GPU | NVIDIA H100 80GB HBM3 |
| VRAM | 81,559 MiB |
| Driver | 580.126.09 |
| CUDA | 12.6 |
| PyTorch | 2.10.0+cu126 |
| Triton | 3.6.0 |
| Transformers | 5.2.0 |
| Python | 3.12 |

## Models Evaluated

| Model | HF Identifier | Params | D | n_kv | head_dim | Context |
|-------|--------------|--------|---|------|----------|---------|
| Qwen2.5-7B | Qwen/Qwen2.5-7B | 7.6B | 28 | 4 | 128 | 131072 |
| Mistral-7B | mistralai/Mistral-7B-v0.1 | 7.2B | 32 | 8 | 128 | 32768 |
| Llama-2-7b | NousResearch/Llama-2-7b-hf | 6.7B | 32 | 32 | 128 | 4096 |
| Qwen2.5-14B | Qwen/Qwen2.5-14B | 14.7B | 48 | 8 | 128 | 131072 |

All models loaded in fp16 (torch.float16) without weight
quantization.

## Memory Usage

| Model | Model Weights (GB) | Peak @ L=32K (GB) |
|-------|--------------------|--------------------|
| Qwen2.5-7B | 15.2 | ~45 |
| Mistral-7B | 14.4 | ~42 |
| Llama-2-7b | 13.5 | ~40 |
| Qwen2.5-14B | 29.7 | ~65 |

Qwen2.5-14B fits on a single H100 80GB at L=32K without
weight quantization.

## Runtime

| Phase | Experiment | Time |
|-------|-----------|------|
| Phase 0 | Canonical eval (3 models) | ~4h |
| Phase 0 | Qwen-7B extended k-sweep | ~1h |
| Phase 1 | Depth extension (Qwen2.5-14B) | ~5 min |
| Phase 2 | Current path benchmark | ~30 min |
| Phase 2 | Triton kernel benchmark | ~10 min |

Phase 1 completed quickly because the 14B model has very
uniform sensitivity (no layer requires more than 1 seed to
characterize).

## Reproducibility

All results can be reproduced using:

```bash
# On rqv (H100)
export RESULTS_ROOT=/mnt/tmpfs/knlp/results/v28

# Phase 0: Canonical evaluation
python3 scripts/v28_canonical_eval.py

# Phase 1: Depth extension
python3 scripts/v28_depth_extension.py

# Phase 2: Current path benchmark
python3 scripts/v28_benchmark_current_path.py

# Phase 2: Triton kernel benchmark
python3 scripts/v28_triton_int4_dequant.py
```

Python environment: `/home/mcgrof/bpa_env/bin/python3`

## DynamicCache Compatibility

Transformers 5.x changed the KV cache API. The scripts use
the `DynamicCache` interface:

```python
# Access cache
past.layers[i].keys    # shape: [B, n_kv, T, head_dim]
past.layers[i].values  # shape: [B, n_kv, T, head_dim]

# Modify cache
past.layers[i].keys = new_keys
past.layers[i].values = new_values
```

The old tuple-based API (`past[i][0]`, `past[i][1]`) is no
longer supported. All v28 scripts include compatibility checks
via `hasattr(past, "layers")`.
