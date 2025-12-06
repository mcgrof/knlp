# KV Plugin Integration Guide

How to add KV cache compression to any HuggingFace model.

## Prerequisites

1. **HuggingFace model** with standard `past_key_values` / cache API
2. **Calibration data**: A few samples (~100 sequences) from your target domain
3. **GPU**: CUDA-capable GPU with sufficient memory for calibration

## Quick Start (5 minutes)

For supported models (Qwen2.5-0.5B, Qwen2.5-7B), use pre-built presets:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from gpt2.compression.presets import get_preset_info, create_cache_from_preset

# Load model
model_name = "Qwen/Qwen2.5-7B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name, torch_dtype=torch.float16, device_map="cuda"
)

# Get conservative preset (1.03x compression, +6% PPL)
preset = get_preset_info(model_name, "r120-vonly")

# Create compressed cache
cache = create_cache_from_preset(preset)

# Generate with compression
outputs = model.generate(
    tokenizer("Hello, my name is", return_tensors="pt")["input_ids"].cuda(),
    max_new_tokens=64,
    past_key_values=cache,
    use_cache=True,
)
print(tokenizer.decode(outputs[0]))
```

## Full Pipeline (New Models)

### Step 1: Calibration

Run PCA calibration to learn optimal projection matrices:

```bash
python scripts/calibrate_kv_lowrank.py \
    --model YourOrg/YourModel \
    --rank 96 \
    --samples 100 \
    --output kv_lowrank_calib_yourmodel_r96.pt
```

**Parameters:**
- `--rank`: Target rank for low-rank projection. Start with `head_dim * 0.75`
- `--samples`: Number of calibration sequences (100 is usually sufficient)
- `--output`: Output calibration file

**Tips:**
- For models with `head_dim=64`: try ranks 60, 56, 48
- For models with `head_dim=128`: try ranks 96, 80, 64

### Step 2: Auto-Tune (Recommended)

Let the auto-tuner find the best configuration under a PPL budget:

```bash
python scripts/auto_tune_kv_compression.py \
    --model YourOrg/YourModel \
    --target-ppl-delta 0.05 \
    --output kv_preset_yourmodel_v9.json
```

**Parameters:**
- `--target-ppl-delta`: Maximum acceptable PPL degradation (default: 5%)
- `--target`: Which to compress: `v` (recommended), `k`, or `kv`
- `--rank-grid`: Comma-separated ranks to try (auto-computed if omitted)

The auto-tuner will:
1. Run calibration for each rank (if not already done)
2. Evaluate all (rank, bits) combinations
3. Select the config with **maximum compression** meeting the PPL budget
4. Output a preset JSON file

### Step 3: Validate Long-Context

Run needle-in-haystack test to verify compression doesn't hurt retrieval:

```bash
python scripts/eval_long_context_needle.py \
    --model YourOrg/YourModel \
    --preset kv_preset_yourmodel_v9.json \
    --context-lengths 512,1024,2048 \
    --num-trials 5
```

**Expected result:** Compressed should match or exceed baseline accuracy.

### Step 4: Use in Production

```python
import json
from gpt2.compression.presets import create_cache_from_preset

# Load your preset
with open("kv_preset_yourmodel_v9.json") as f:
    preset = json.load(f)

# Create cache
cache = create_cache_from_preset(preset, device="cuda")

# Use with model.generate()
outputs = model.generate(
    input_ids,
    past_key_values=cache,
    use_cache=True,
    max_new_tokens=256,
)
```

## Configuration Options

### Target Selection

| Target | What's Compressed | Typical Compression | Notes |
|--------|-------------------|---------------------|-------|
| `v` | V cache only | 1.03-1.14x | **Recommended**. K preserves attention patterns. |
| `k` | K cache only | N/A | **Do not use.** Causes catastrophic quality loss. |
| `kv` | Both K and V | N/A | **Do not use.** K compression destroys quality. |

### Quantization

| Bits | Memory vs FP16 | Throughput Cost | PPL Impact |
|------|----------------|-----------------|------------|
| 16 | Baseline | -5% | +6% |
| 8 | -24% | -21% | +6% |

**Trade-off:** INT8 saves 24% memory but costs 21% throughput vs baseline.
Not free - use only if memory-constrained.

### Rank Selection

| Model Size | head_dim | Recommended Rank | PPL Impact |
|------------|----------|------------------|------------|
| 7B | 128 | 120 | +6% |
| 7B | 128 | 112 | +14% |
| 72B | 128 | 120 | +5% |

**Conservative approach:** Use rank 120 (head_dim - 8) for ~6% PPL impact.
Lower ranks cause rapidly increasing quality degradation.

## Troubleshooting

### "Calibration file not found"

Ensure the calibration file is in the current directory or provide the full path:

```python
preset["calibration_file"] = "/full/path/to/kv_lowrank_calib_model_r96.pt"
```

### "PPL degradation too high"

1. Increase rank (less compression, better quality)
2. Use V-only compression instead of K+V
3. Use int8 instead of int4 quantization

### "OOM during calibration"

1. Reduce `--samples` to 50
2. Use `--seq-len 512` instead of default 1024
3. Use gradient checkpointing: `--gradient-checkpointing`

### "Model doesn't have past_key_values"

The model must use HuggingFace's standard KV cache API. Check if:
- `model.config.use_cache` is supported
- `outputs.past_key_values` is returned when `use_cache=True`

## Per-Layer Compression (Advanced)

For models where different layers have different compressibility:

```python
from gpt2.compression.compressed_cache import (
    CompressedDynamicCache,
    IdentityCompressor,
    load_calibrated_compressors,
)

# Load calibrated compressors
k_comp, v_comp, metadata = load_calibrated_compressors(
    "kv_lowrank_calib_model_r96.pt",
    device=torch.device("cuda"),
    quantize_bits=8,
)

# Make early layers use identity (no compression)
for i in range(4):  # First 4 layers uncompressed
    k_comp[i] = IdentityCompressor()
    v_comp[i] = IdentityCompressor()

# Create cache with mixed compression
cache = CompressedDynamicCache(k_comp, v_comp, num_layers=28)
```

## Example: Full Workflow for Mistral-7B

```bash
# Step 1: Calibrate with multiple ranks
python scripts/calibrate_kv_lowrank.py \
    --model mistralai/Mistral-7B-v0.1 --rank 96 --output kv_calib_mistral_r96.pt
python scripts/calibrate_kv_lowrank.py \
    --model mistralai/Mistral-7B-v0.1 --rank 80 --output kv_calib_mistral_r80.pt

# Step 2: Auto-tune with 5% PPL budget
python scripts/auto_tune_kv_compression.py \
    --model mistralai/Mistral-7B-v0.1 \
    --target-ppl-delta 0.05 \
    --rank-grid 96,80 \
    --output kv_preset_mistral_v9.json

# Step 3: Validate
python scripts/eval_long_context_needle.py \
    --model mistralai/Mistral-7B-v0.1 \
    --preset kv_preset_mistral_v9.json

# Step 4: Run stability check
python scripts/eval_stability.py \
    --model mistralai/Mistral-7B-v0.1 \
    --preset kv_preset_mistral_v9.json \
    --num-runs 3
```

## Available Scripts Reference

| Script | Purpose |
|--------|---------|
| `calibrate_kv_lowrank.py` | Generate PCA calibrations |
| `auto_tune_kv_compression.py` | Auto-find best config |
| `eval_long_context_needle.py` | Long-context retrieval test |
| `eval_stability.py` | Multi-run stability evaluation |
| `profile_kv_performance.py` | Latency/throughput profiling |
| `validate_ppl_postbugfix.py` | PPL ablation testing |

## Preset JSON Format

```json
{
  "model": "Qwen/Qwen2.5-7B",
  "rank": 120,
  "target": "v",
  "bits": 16,
  "compression_ratio": 1.03,
  "baseline_ppl": 7.88,
  "compressed_ppl": 8.35,
  "ppl_delta": 0.06,
  "calibration_file": "kv_calib_qwen7b_r120.pt",
  "head_dim": 128,
  "num_layers": 28
}
```

## LayerNorm Nullspace (LN-NS)

LayerNorm nullspace compression exploits the geometry of LN outputs (mean ~0)
to achieve "free" rank-1 reduction: d dimensions → d-1 dimensions.

**Pros:**
- +0.8-1.6% extra KV memory savings (V-only or K+V)
- Deterministic, no calibration required
- Composable with existing PCA + quantization

**Cons:**
- ~7% slower in real generation due to compress/decompress overhead
- K-side nullspace has higher PPL impact (+0.92%) than V-side (+0.22%)

**Recommendation:** Enable only in memory-constrained setups (e.g., long contexts
or many concurrent users hitting a VRAM ceiling). Not recommended for
latency-sensitive workloads.

### Usage

```python
from gpt2.compression.kv_policies import (
    KVPolicyName,
    choose_policy_for_run,
    build_compressor_config_for_policy,
)

# Memory-emergency policy enables V-only LN nullspace
config = build_compressor_config_for_policy(
    model.config,
    model_id="Qwen/Qwen2.5-7B",
    policy_name=KVPolicyName.MEMORY_EMERGENCY,
)
# config["use_ln_nullspace_v"] == True
```

### Policy Tiers

| Policy | LN Nullspace | Use Case |
|--------|--------------|----------|
| `default` | Off | Standard production, good speed/quality/memory tradeoff |
| `high_compression` | Off | More aggressive compression, +3% PPL budget |
| `memory_emergency` | V-only On | VRAM-constrained, accept ~7% slower for memory savings |

## Tips for Best Results

1. **Only compress V** - K compression causes catastrophic quality loss
2. **Start with rank 120** - Conservative setting with ~6% PPL impact
3. **INT8 trades throughput for memory** - 24% memory savings but 21% slower
4. **Run needle test** - Verify long-context retrieval before deployment
5. **Profile at target context lengths** - Overhead is relatively constant
