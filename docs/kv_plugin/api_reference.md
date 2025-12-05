# API Reference

## Core Classes

### `CompressedDynamicCache`

The main cache class that wraps HuggingFace's DynamicCache with compression.

```python
from gpt2.compression.compressed_cache import CompressedDynamicCache

cache = CompressedDynamicCache(
    k_compressors=k_compressors,      # List of K compressors per layer
    v_compressors=v_compressors,      # List of V compressors per layer
    num_layers=32,                     # Number of transformer layers
    compress_start_len=512,            # Start compression after N tokens (adaptive)
    uncompressed_tail=256,             # Keep last N tokens uncompressed (adaptive)
)
```

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `k_compressors` | `List[nn.Module]` | Required | Compressor for K per layer |
| `v_compressors` | `List[nn.Module]` | Required | Compressor for V per layer |
| `num_layers` | `int` | Required | Number of transformer layers |
| `compress_start_len` | `int` | `0` | Enable adaptive mode: compress after N tokens |
| `uncompressed_tail` | `int` | `0` | Keep last N tokens uncompressed |

#### Methods

- `reset()` - Clear the cache for a new sequence
- `get_memory_stats()` - Return dict with memory usage statistics

---

### `load_preset_cache`

Convenience function to load a pre-calibrated cache configuration.

```python
from gpt2.compression.compressed_cache import load_preset_cache

cache = load_preset_cache(
    preset_path="kv_preset_qwen2.5-7b.json",
    num_layers=32,
    device="cuda",
    adaptive=True,
    compress_after=512,
    keep_recent=256,
)
```

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `preset_path` | `str` | Required | Path to preset JSON file |
| `num_layers` | `int` | Required | Number of transformer layers |
| `device` | `str` | `"cuda"` | Device to load tensors |
| `adaptive` | `bool` | `False` | Enable adaptive compression mode |
| `compress_after` | `int` | `512` | Adaptive: compress after N tokens |
| `keep_recent` | `int` | `256` | Adaptive: keep last N uncompressed |

---

### Compressor Classes

#### `IdentityCompressor`

No-op compressor (pass-through). Used for K when doing V-only compression.

```python
from gpt2.compression.compressed_cache import IdentityCompressor

k_comp = IdentityCompressor()
```

#### `QuantizedCalibratedCompressor`

PCA-based low-rank compression with int8 quantization.

```python
from gpt2.compression.compressed_cache import QuantizedCalibratedCompressor

v_comp = QuantizedCalibratedCompressor(
    U=projection_matrix,    # [head_dim, rank] orthonormal matrix
    mean=mean_vector,       # [head_dim] centering vector
    bits=8,                 # Quantization bits (8 or 16)
    dtype=torch.float16,    # Compute dtype
)
```

---

## Preset File Format

Preset files are JSON with calibration data:

```json
{
  "model": "Qwen/Qwen2.5-7B",
  "rank": 96,
  "head_dim": 128,
  "n_layers": 32,
  "n_heads": 28,
  "compression_ratio": 2.67,
  "layers": [
    {
      "K": {"U": [...], "mean": [...]},
      "V": {"U": [...], "mean": [...]}
    },
    ...
  ]
}
```

---

## Calibration

Generate a preset for a new model:

```bash
python scripts/auto_tune_kv_compression.py \
  --model Qwen/Qwen2.5-7B \
  --target-ppl-delta 0.05 \
  --output my_preset.json
```

See [Calibration Guide](calibration_guide.md) for details.
