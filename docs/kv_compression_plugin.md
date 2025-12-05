# KV Cache Compression Plugin System

Post-hoc KV cache compression for HuggingFace models without retraining.

## Overview

This system enables inference-time KV cache compression of frozen models using:
- **FIM-guided tiering**: Use Fisher Information metrics to identify compression targets
- **Pluggable backends**: KVSplice (learned) or PCA (variance-based)
- **Per-head heterogeneous compression**: Different ranks per layer/head
- **Fast calibration**: Hours instead of days of retraining

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│ 1. FIM Analysis (Optional but Recommended)                  │
│    Extract FIM metrics from trained model → compression cfg │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ 2. Compression Config Generation                            │
│    FIM metrics → Per-head rank recommendations (JSON)       │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ 3. Calibration                                              │
│    Collect KV samples → Fit compressor → Save state         │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ 4. Deployment                                               │
│    Load compressed model → Inference with reduced memory    │
└─────────────────────────────────────────────────────────────┘
```

## Quick Start

### Step 1: Generate Compression Config (with FIM)

If you have FIM metrics from training:

```bash
python scripts/analyze_fim_metrics.py \
  --entity mcgrof-citizen \
  --project gpt2-ra-v2-h100 \
  --generate-compression-config \
  --compression-config-output compression_config.json \
  --output-dir ./
```

This generates:
- `compression_config.json`: Machine-readable config
- `compression_config_summary.txt`: Human-readable summary

**Strategy applied**:
- `trace > 0.95`: Critical heads → No compression
- `cond > 1e7, trace < 0.90`: Excellent targets → rank=8 (88% compression)
- `cond > 1e6`: Good targets → rank=16 (75% compression)
- Otherwise: Moderate → rank=32 (50% compression)

### Step 2: Calibrate Compressor

```bash
python scripts/calibrate_kv_compressor.py \
  --model openai-community/gpt2 \
  --compressor kvsplice \
  --config compression_config.json \
  --calibration-dataset wikitext \
  --calibration-samples 1000 \
  --eval-samples 100 \
  --output-dir compression_results
```

**Output**:
```
Baseline perplexity:    29.20
Compressed perplexity:  30.15
Perplexity degradation: +3.25%
Memory savings:         52.3%
Calibration time:       147.2s
```

Results saved to:
- `compression_results/calibration_results_kvsplice.json`
- `compression_results/compressor_kvsplice.pt`

### Step 3: Use Compressed Model

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from gpt2.compression import KVSpliceCompressor, CompressedKVModelWrapper

# Load model
model = AutoModelForCausalLM.from_pretrained("gpt2")
tokenizer = AutoTokenizer.from_pretrained("gpt2")

# Load calibrated compressor
compressor = KVSpliceCompressor(config)
compressor.load_state_dict(torch.load("compression_results/compressor_kvsplice.pt"))

# Wrap model
wrapped = CompressedKVModelWrapper(model, compressor)

# Use normally
inputs = tokenizer("The future of AI is", return_tensors="pt")
outputs = wrapped.generate(**inputs, max_length=50, use_cache=True)
print(tokenizer.decode(outputs[0]))
```

## Compression Backends

### KVSplice (Learned)

Learns linear projections during calibration:
- Minimizes reconstruction loss: `||K - K_hat||² + ||V - V_hat||²`
- Optional scale/shift parameters for latent space
- Better quality than variance-based methods
- Requires calibration (10-20 minutes for GPT-2)

### PCA (Variance-Based)

Fast baseline using principal components:
- No training, only statistics collection
- Good for quick experimentation
- Baseline for comparison against learned methods
- Calibration in ~5 minutes

```bash
# Use PCA instead of KVSplice
python scripts/calibrate_kv_compressor.py \
  --compressor pca \
  --config compression_config.json
```

## Manual Config (Without FIM)

If FIM metrics are unavailable, create config manually:

```json
{
  "global": {
    "target_memory_reduction": 0.5,
    "d_head": 64,
    "algo_default": "kvsplice"
  },
  "per_layer_head": {
    "0/0": {"enabled": true, "rank": 32, "algo": "kvsplice", "d_k": 64, "d_v": 64},
    "0/1": {"enabled": true, "rank": 32, "algo": "kvsplice", "d_k": 64, "d_v": 64}
  }
}
```

## Expected Results

Based on ChatGPT's design and our analysis:

| Compression | Memory Savings | Quality Loss (PPL) | Calibration Time |
|-------------|----------------|-------------------|------------------|
| Uniform 50% | ~50% | 3-5% | 10-20 min |
| FIM-guided heterogeneous | 60-70% | <2% | 10-20 min |
| PCA baseline | ~50% | 5-8% | 5 min |

## Implementation Details

### Monkey-Patching Strategy

The wrapper monkey-patches attention forward() at runtime:

1. Intercepts KV computation in attention layers
2. Compresses per-head KV: `Z_k, Z_v = compress(K, V)`
3. Stores compressed cache: `past_key_values = (Z_k, Z_v)`
4. Decompresses for attention: `K_hat, V_hat = decompress(Z_k, Z_v)`
5. Computes attention with decompressed KV

### Calibration Workflow

**Phase 0**: Statistics collection
- Run model on calibration data
- Collect KV samples per layer/head
- Downsample to keep memory bounded (~10k samples per head)

**Phase 1**: Parameter fitting
- KVSplice: Mini-batch gradient descent on reconstruction loss
- PCA: SVD to find principal components
- ~10 epochs, learning rate 1e-3

**Phase 2**: Validation
- Evaluate perplexity on held-out data
- Compare baseline vs compressed

## Troubleshooting

### ImportError: transformers not found

```bash
pip install transformers datasets torch
```

### CUDA out of memory during calibration

Reduce batch size or calibration samples:

```bash
python scripts/calibrate_kv_compressor.py \
  --calibration-samples 500 \
  --batch-size 2
```

### High perplexity degradation

- Use FIM-guided config (protects critical heads)
- Reduce compression ratio (higher rank values)
- Try PCA baseline to validate reconstruction quality

## Future Extensions

From `docs/fim_compression_comparison.md`:

1. **Spline nonlinearity** for heavy-tailed distributions
2. **Top-eigenvector projection** for high-eigmax heads
3. **Adaptive precision** (FP8/FP4) for high-cond heads
4. **Task-aware fine-tuning** (Phase 2 calibration)
5. **Production deployment** with latency benchmarks

## References

- Design: `/data/plug/design.txt` (ChatGPT's pluggable architecture)
- Analysis: `docs/fim_compression_comparison.md`
- FIM Guide: `docs/FIM.md`
- Implementation: `gpt2/compression/`
