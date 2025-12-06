# KV Cache Compression Plugin

Low-rank projection for KV cache memory reduction.

## What This Is

A research tool for exploring KV cache compression via orthogonal projection.
Useful for understanding compression dynamics and architectural differences
between attention mechanisms.

**Not recommended for production.** The throughput cost exceeds the memory benefit.

## Measured Results (B200, Qwen2.5-7B)

Throughput = tokens generated per second. Higher is better.

| Configuration | Memory | Throughput | Quality (PPL) |
|---------------|--------|------------|---------------|
| Baseline | 56 MB | 110 tok/s | 7.88 |
| FP16 Low-Rank (r120) | 54 MB (-3%) | 105 tok/s (-5%) | 8.35 (+6%) |
| INT8 Low-Rank (r120) | 41 MB (-27%) | 87 tok/s (-21%) | 8.34 (+6%) |
| FP16 Low-Rank (r96) | 49 MB (-12%) | 106 tok/s (-4%) | 10.6 (+35%) |

Task accuracy (HellaSwag, ARC-Easy, WinoGrande, PIQA) showed 0% degradation
at rank 120 despite +6% PPL increase.

## Research Value

### K vs V Compression Dynamics

The plugin proved that K and V have fundamentally different compressibility:

| Component | Variance Spread | Interpretation |
|-----------|-----------------|----------------|
| K | 132x | Few dimensions dominate - fragile |
| V | 4x | Uniform importance - robust |

**K compression fails catastrophically.** A few dimensions carry almost all
attention pattern information. Compressing K destroys these critical dimensions.

**V compression is tolerable.** All dimensions contribute roughly equally,
so compression removes less critical information.

This finding applies to standard attention (Qwen, Llama, Mistral). MLA
architectures (DeepSeek-V2) equalize K/V through their shared latent space.

### Calibration vs Random Projection

PCA-calibrated projections outperform random orthogonal by ~8%:

| Method | PPL at Rank 112 |
|--------|-----------------|
| Random | +15.2% |
| Calibrated PCA | +14.0% |

Modest improvement. Calibration helps but isn't transformative.

### Scaling with Model Size

Larger models tolerate compression slightly better:

| Model | Rank 120 PPL | Rank 112 PPL |
|-------|--------------|--------------|
| 7B | +5.9% | +14.0% |
| 72B | +4.6% | +10.6% |

~20% relative improvement at 72B vs 7B. Real but modest.

## When to Use

Use for **research and experimentation** only:

- Understanding attention compression dynamics
- Comparing standard attention vs MLA architectures
- Benchmarking compression quality/speed tradeoffs
- Memory profiling experiments

Do not use for production inference. Better alternatives exist.

## Quick Start

```python
import torch
from transformers import AutoModelForCausalLM
from gpt2.compression.compressed_cache import (
    CompressedDynamicCache,
    CalibratedCompressor,
    IdentityCompressor,
)

# Calibrate first
# python scripts/calibrate_kv_lowrank.py --model Qwen/Qwen2.5-7B --rank 120

# Load calibration
calib = torch.load("key_results/kv_calib_qwen7b_r120.pt")

model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-7B", device_map="cuda")
num_layers = model.config.num_hidden_layers

# V-only compression (K uses identity)
k_compressors = [IdentityCompressor() for _ in range(num_layers)]
v_compressors = [
    CalibratedCompressor(layer["V"]["U"], layer["V"]["mean"])
    for layer in calib["layers"]
]

cache = CompressedDynamicCache(k_compressors, v_compressors, num_layers)
outputs = model.generate(input_ids, past_key_values=cache, max_new_tokens=100)
```

## INT8 Quantization

INT8 reduces memory further but costs more throughput:

| vs Baseline | Memory | Throughput |
|-------------|--------|------------|
| FP16 Low-Rank | -3% | -5% |
| INT8 Low-Rank | -27% | -21% |

Use `QuantizedCalibratedCompressor` instead of `CalibratedCompressor`.

## Benchmarking

```bash
# Full benchmark (memory, throughput, quality)
python scripts/benchmark_kv_comprehensive.py \
    --model Qwen/Qwen2.5-7B \
    --rank 120 \
    --calibration key_results/kv_calib_qwen7b_r120.pt

# Quantization comparison
python scripts/benchmark_kv_quantized.py \
    --model Qwen/Qwen2.5-7B \
    --rank 120 \
    --calibration key_results/kv_calib_qwen7b_r120.pt
```

## Better Alternatives for Production

1. **KV Cache Quantization** - INT8/INT4 KV cache in frameworks like vLLM.
   2x memory reduction with minimal quality loss and no throughput penalty.

2. **Grouped-Query Attention** - Architecture change that reduces KV cache
   at the model level. No runtime overhead.

3. **Sliding Window Attention** - Fixed memory regardless of sequence length.

4. **PagedAttention** - Better memory utilization without compression overhead.

## Hardware Tested

- 4x NVIDIA B200 (766 GB total VRAM)
- Model: Qwen2.5-7B, Qwen2.5-72B
- Framework: PyTorch 2.x, Transformers 4.x

## Files

- `README.md` - This file
- `calibration_guide.md` - How to create calibration files
- `architecture_analysis.md` - K vs V compression dynamics
- `quality_results.md` - Benchmark data
