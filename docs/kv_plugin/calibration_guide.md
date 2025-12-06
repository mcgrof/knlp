# Calibration Guide

How to create compression presets for custom models.

## Overview

Calibration extracts PCA projection matrices from sample text. These matrices
capture the principal directions of variance in K and V, enabling efficient
low-rank compression.

## Quick Calibration

```bash
python scripts/auto_tune_kv_compression.py \
  --model Qwen/Qwen2.5-7B \
  --target-ppl-delta 0.05 \
  --output kv_preset_qwen2.5-7b.json
```

This automatically:
1. Runs calibration text through the model
2. Extracts K/V projections per layer
3. Computes PCA components
4. Searches for optimal rank within PPL budget
5. Saves preset JSON

## Manual Calibration

For more control:

```python
from gpt2.compression.compressed_cache import calibrate_kv_compression

calibration = calibrate_kv_compression(
    model=model,
    tokenizer=tokenizer,
    texts=calibration_texts,      # List of diverse sample texts
    rank=96,                       # Target rank (128 - rank = compression)
    device="cuda",
)

# Save calibration
torch.save(calibration, "kv_calibration.pt")
```

## Calibration Data Requirements

- **Diversity**: Use varied text (prose, code, dialogue, technical)
- **Length**: At least 256 tokens per sample
- **Quantity**: 5-10 samples is usually sufficient
- **Quality**: Representative of deployment distribution

Example calibration texts:

```python
calibration_texts = [
    "The transformer architecture revolutionized NLP...",
    "def fibonacci(n):\n    if n <= 1: return n...",
    "User: What is machine learning?\nAssistant: Machine learning is...",
    "∫ x² dx = x³/3 + C. The fundamental theorem states...",
    "In a shocking turn of events, the stock market...",
]
```

## Rank Selection

**Measured on Qwen2.5-7B with V-only calibrated PCA (head_dim=128):**

| Rank | Compression | Measured PPL Impact |
|------|-------------|---------------------|
| 128 | 1.00x (none) | 0% |
| 120 | 1.03x | +6% |
| 112 | 1.07x | +14% |
| 96 | 1.14x | +35% |
| 64 | 1.33x | +647% (broken) |

**Reality check**: Aggressive compression does not work. Even with
calibrated PCA and V-only compression, 1.14x (rank 96) causes +35%
PPL degradation. The practical limit is 1.03-1.07x compression with
6-14% quality loss.

## Architecture-Specific Notes

### Standard Attention (Qwen, Mistral, Llama)

- Calibrate V only (set K compressor to `IdentityCompressor`)
- K calibration wastes compute and risks quality

### MLA (DeepSeek-V2)

- Calibrate the shared KV latent
- Both K and V projections can be compressed equally

## Validating Calibration

After calibration, validate with PPL measurement:

```bash
python scripts/benchmark_kv_compression_quality.py \
  --model Qwen/Qwen2.5-7B \
  --ranks 120 \
  --calibration key_results/kv_calib_qwen7b_r120.pt \
  --num-samples 50
```

Expected output (conservative rank 120):
```
Baseline PPL: 7.88
Compressed PPL: 8.35
Delta: +5.9%
```

Note: Do not expect <1% PPL impact. That was never achievable.

## Troubleshooting

**High PPL degradation (>5%)**
- Reduce rank (try 112 instead of 96)
- Check calibration text diversity
- Verify model architecture (MLA vs standard?)

**OOM during calibration**
- Reduce calibration text length
- Use fewer samples
- Run on CPU with `--device cpu`

**Inconsistent results**
- Use fixed random seed
- Increase calibration samples
- Check for NaN/Inf in hidden states
