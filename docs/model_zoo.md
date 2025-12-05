# KV Plugin Model Zoo

Supported models with validated compression presets.

## Summary Table

| Model | Size | Compression | PPL Delta | Needle | Preset File |
|-------|------|-------------|-----------|--------|-------------|
| Qwen2.5-7B | 7B | **2.67x** | +0.99% | 100% | `kv_preset_qwen-qwen2.5-7b_v9.json` |
| Mistral-7B | 7B | **2.42x** | +2.30% | 100%* | `kv_preset_mistralai-mistral-7b-v0.1_v9.json` |
| Qwen2-1.5B | 1.5B | **2.29x** | +3.06% | - | `kv_preset_qwen-qwen2-1.5b_v9.json` |
| Qwen2.5-0.5B | 0.5B | **2.29x** | +4.06% | 100% | `kv_preset_qwen-qwen2.5-0.5b_v9.json` |

*Needle test matches baseline behavior exactly.

## Key Finding

**Larger models are more compressible.** The 7B models achieve better compression
ratios with lower quality loss than smaller models:

- Qwen2.5-7B: 2.67x @ +0.99% (best)
- Mistral-7B: 2.42x @ +2.30%
- Qwen2.5-0.5B: 2.29x @ +4.06%

## Model Details

### Qwen2.5-7B (Recommended)

Best compression-to-quality ratio in the zoo.

```json
{
  "model": "Qwen/Qwen2.5-7B",
  "rank": 96,
  "target": "v",
  "bits": 8,
  "compression_ratio": 1.33,
  "total_compression": 2.67,
  "ppl_delta": 0.99%
}
```

**Usage:**
```python
from gpt2.compression.compressed_cache import load_preset_cache

model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-7B", ...)
cache = load_preset_cache("kv_preset_qwen-qwen2.5-7b_v9.json", device="cuda")
outputs = model.generate(input_ids, past_key_values=cache, ...)
```

### Mistral-7B-v0.1

Solid compression on the popular Mistral architecture.

```json
{
  "model": "mistralai/Mistral-7B-v0.1",
  "rank": 106,
  "target": "v",
  "bits": 8,
  "compression_ratio": 1.21,
  "total_compression": 2.42,
  "ppl_delta": 2.30%
}
```

**Architecture notes:**
- 32 layers (vs 28 for Qwen)
- head_dim=128 (same as Qwen)
- Slightly less compressible than Qwen at same size

### Qwen2-1.5B

Mid-size model for resource-constrained environments.

```json
{
  "model": "Qwen/Qwen2-1.5B",
  "rank": 112,
  "target": "v",
  "bits": 8,
  "compression_ratio": 1.14,
  "total_compression": 2.29,
  "ppl_delta": 3.06%
}
```

### Qwen2.5-0.5B

Smallest tested model. Higher relative PPL impact due to less redundancy.

```json
{
  "model": "Qwen/Qwen2.5-0.5B",
  "rank": 112,
  "target": "v",
  "bits": 8,
  "compression_ratio": 1.14,
  "total_compression": 2.29,
  "ppl_delta": 4.06%
}
```

## Adding New Models

To add support for a new model:

1. **Run auto-tuner:**
```bash
python scripts/auto_tune_kv_compression.py \
    --model <model_name> \
    --target-ppl-delta 0.05 \
    --device cuda
```

2. **Verify with needle test:**
```bash
python scripts/eval_long_context_needle.py \
    --model <model_name> \
    --preset kv_preset_<model>_v9.json \
    --context-lengths 512,1024,2048
```

3. **Add entry to this table**

## Compression Strategy

All presets use the same validated strategy:

- **Target:** V-only compression (K is more sensitive to degradation)
- **Method:** PCA-calibrated low-rank projection
- **Quantization:** int8 on compressed representations
- **Calibration:** 1024 tokens of representative text

This achieves 2-3x memory reduction with <5% PPL degradation across all models.

## Planned Models

Future models to add (v11+):

- [ ] LLaMA-3-8B
- [ ] Phi-3-medium
- [ ] Gemma-2B/7B
- [ ] CodeLlama variants

## Version History

| Version | Models Added |
|---------|-------------|
| v9 | Qwen2.5-0.5B, Qwen2.5-7B |
| v10 | Stability verification |
| v11 | Mistral-7B, Qwen2-1.5B |
