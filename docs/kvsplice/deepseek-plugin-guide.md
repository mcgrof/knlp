# KVSplice Plug-in for DeepSeek Models

This guide shows how to test KVSplice compression on pretrained DeepSeek models without retraining.

## Overview

KVSplice is a learned compression layer that reduces KV cache memory usage during inference. DeepSeek-V2 already uses MLA (Multi-head Latent Attention) for 6x compression. KVSplice adds an additional 2x compression on top of MLA, achieving 12x total compression.

**Key insight**: Since we currently observe KVSplice working as a low-rank projection (transform parameters at initialization), we can test it zero-shot on pretrained models to see if the compression generalizes without fine-tuning.

## Files

```
scripts/
├── deepseek_kvsplice_plugin.py      # Core plug-in module
├── benchmark_deepseek_kvsplice.py   # Inference throughput benchmark
├── eval_deepseek_quality.py         # Perplexity evaluation
└── test_deepseek_kvsplice.sh        # Quick test script (runs both)
```

## Quick Start

### Prerequisites

```bash
# DeepSeek models require transformers >= 4.36.0
pip install 'transformers>=4.36.0' torch datasets accelerate

# Or upgrade if already installed
pip install --upgrade 'transformers>=4.36.0'
```

### Run All Tests

```bash
./scripts/test_deepseek_kvsplice.sh
```

This runs both benchmark and quality evaluation on DeepSeek-V2-Lite.

## Individual Scripts

### 1. Plug-in Module

**`deepseek_kvsplice_plugin.py`** - Core KVSplice implementation

Key functions:
- `LearnedKVSplice`: Compression layer (same as our GPT-2 implementation)
- `KVSpliceWrapper`: Wrapper that intercepts attention KV cache
- `patch_model_with_kvsplice()`: Adds KVSplice to pretrained model
- `get_kv_cache_size()`: Estimates memory savings

Example usage:
```python
from transformers import AutoModelForCausalLM
from deepseek_kvsplice_plugin import patch_model_with_kvsplice

# Load pretrained model
model = AutoModelForCausalLM.from_pretrained(
    "deepseek-ai/DeepSeek-V2-Lite",
    torch_dtype=torch.float16,
    device_map="auto",
)

# Add KVSplice compression (2x on top of MLA)
patch_model_with_kvsplice(model, compression_ratio=0.5)

# Model now uses compressed KV cache during generation
outputs = model.generate(input_ids, max_new_tokens=100)
```

### 2. Inference Benchmark

**`benchmark_deepseek_kvsplice.py`** - Measures throughput and memory

Compares original model vs KVSplice across different batch sizes.

Usage:
```bash
python scripts/benchmark_deepseek_kvsplice.py \
    --model deepseek-ai/DeepSeek-V2-Lite \
    --compression-ratio 0.5 \
    --batch-sizes 1,4,8,16 \
    --seq-len 512 \
    --trials 3
```

Output shows:
- Tokens/second for each configuration
- GPU memory usage
- Throughput comparison (speedup ratio)
- Memory savings from compressed cache

Expected results:
- **Throughput**: ~0.89-0.92x (11% slower due to compress/expand overhead)
- **Memory**: 50% KV cache reduction at compression_ratio=0.5

### 3. Quality Evaluation

**`eval_deepseek_quality.py`** - Measures perplexity impact

Tests whether untrained KVSplice degrades model quality.

Usage:
```bash
python scripts/eval_deepseek_quality.py \
    --model deepseek-ai/DeepSeek-V2-Lite \
    --compression-ratio 0.5 \
    --dataset wikitext \
    --samples 1000 \
    --batch-size 4
```

Output shows:
- Original model perplexity
- KVSplice model perplexity
- Quality degradation percentage

Expected results based on TinyStories experiments:
- **Degradation**: 1-3% perplexity increase
- Zero-shot compression works reasonably well
- Can be improved with fine-tuning

## Model Support

### DeepSeek-V2-Lite (Recommended for Testing)

```bash
--model deepseek-ai/DeepSeek-V2-Lite
```

- Size: ~8GB
- Parameters: ~16B
- Already uses MLA compression
- Good for quick experiments on H100

### DeepSeek-V2 (Full Model)

```bash
--model deepseek-ai/DeepSeek-V2
```

- Size: ~236B parameters
- Requires multiple GPUs or quantization
- Tests compression at production scale

### Detection Logic

The plug-in automatically detects:
- MLA latent dimension from `model.config.kv_lora_rank`
- Transformer layers from `model.model.layers`
- Attention modules from `layer.self_attn`

If your model has a different structure, adjust `layer_pattern` argument:
```python
patch_model_with_kvsplice(model, layer_pattern="transformer.h")
```

## Compression Ratios

Test different compression ratios to find optimal trade-off:

```bash
# Aggressive (4x compression on MLA)
--compression-ratio 0.25

# Balanced (2x compression on MLA) - recommended
--compression-ratio 0.5

# Conservative (1.4x compression on MLA)
--compression-ratio 0.7
```

Lower ratio = more compression = more memory savings = potentially lower quality

## Expected Results

### Memory Savings

At `compression_ratio=0.5` (our experiments on GPT-2):
- KV cache: 50% reduction
- Total model memory: Depends on batch size and sequence length
- Benefit increases with longer sequences and larger batches

### Throughput Cost

Based on our TinyStories measurements:
- Throughput: 0.89-0.92x of original (11% slower)
- Caused by compress/expand layer computation
- Trade-off: accept speed cost for memory savings

### Quality Impact

Based on our ablation studies (H100, W7900, A100):
- Perplexity increase: 1-3%
- Validation loss increase: 0.5-1.4%
- Works zero-shot without fine-tuning
- Transform parameters remain at initialization (identity transform)
- Compression happens via low-rank projection only

## When KVSplice Helps

**Use KVSplice when:**
- Memory-constrained scenarios (8-16GB GPUs)
- Long sequences (2K+ tokens) where cache dominates memory
- High concurrency serving (fit more requests)
- Willing to trade 11% speed for 2x capacity

**Skip KVSplice when:**
- Large GPUs with abundant memory (80GB H100)
- Latency-critical applications
- Short sequences where cache is small anyway
- Need maximum throughput

## Limitations

### Current Implementation

1. **Untrained compression**: Parameters initialized to identity transform
2. **No gradient flow**: DeepSeek model frozen, only KVSplice layers added
3. **Transform not learning**: Based on our findings, scale/shift stay at init values
4. **Works via projection**: Low-rank autoencoder does the heavy lifting

### To Improve Quality

If quality degradation is too high:

1. **Fine-tune KVSplice layers**:
   ```python
   # Freeze base model, train only KVSplice
   for param in model.parameters():
       param.requires_grad = False

   for layer in model.model.layers:
       if hasattr(layer.self_attn, 'kvsplice'):
           for param in layer.self_attn.kvsplice.parameters():
               param.requires_grad = True

   # Train on target domain
   optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()))
   ```

2. **Try higher compression ratio** (less aggressive):
   ```bash
   --compression-ratio 0.7  # 1.4x instead of 2x
   ```

3. **Add LayerNorm** (already enabled by default):
   ```python
   patch_model_with_kvsplice(model, use_layernorm=True)
   ```

4. **Test different initialization**:
   Modify `LearnedKVSplice.__init__()` to use different SVD initialization

## Research Questions

Testing on DeepSeek helps answer:

1. **Does low-rank projection generalize?**
   - Do our learned weights from TinyStories transfer to DeepSeek?
   - Or does each model need custom training?

2. **What's the real quality cost?**
   - TinyStories: 1-3% degradation
   - Real model: ???

3. **Does transform learning matter?**
   - Currently: identity transform (scale≈1, shift=0)
   - If we fine-tune: do scale/shift start learning?
   - Does LayerNorm help?

4. **Is this production-ready?**
   - Can we deploy 2x cache reduction zero-shot?
   - Or do we need per-model fine-tuning?

## Next Steps

1. **Run initial tests**:
   ```bash
   ./scripts/test_deepseek_kvsplice.sh
   ```

2. **Analyze results**:
   - Check perplexity degradation
   - Verify memory savings
   - Measure throughput cost

3. **If quality is good** (< 5% degradation):
   - Test on production use cases
   - Compare to quantization approaches
   - Consider deploying zero-shot

4. **If quality is poor** (> 5% degradation):
   - Fine-tune KVSplice layers on target domain
   - Try different compression ratios
   - Investigate why projection doesn't transfer

5. **Report findings**:
   - Add results to docs/kvsplice.md
   - Compare to our TinyStories experiments
   - Determine if approach generalizes

## Troubleshooting

### AttributeError: 'DynamicCache' object has no attribute 'seen_tokens'

This can happen for two reasons:

**1. Transformers too old** - DeepSeek requires >= 4.36.0:
```bash
pip install --upgrade 'transformers>=4.36.0'
```

**2. Stale cached model code** - Even with latest transformers, cached model code from `trust_remote_code` can be incompatible:
```bash
# Clear cached DeepSeek model code
rm -rf ~/.cache/huggingface/hub/modules--transformers_modules/deepseek*
# Or if cache is elsewhere:
rm -rf /home/ubuntu/hg-cache/modules/transformers_modules/deepseek*
```

The scripts now use `code_revision="main"` to force latest model code, but if you still see errors, manually clearing the cache fixes it.

### OOM during benchmark

Reduce batch size:
```bash
--batch-sizes 1,2,4  # Instead of 1,4,8,16
```

### Model download fails

Check HuggingFace authentication:
```bash
huggingface-cli login
```

### Quality evaluation too slow

Reduce samples:
```bash
--samples 100  # Instead of 1000
```

### Import errors

Install dependencies:
```bash
pip install transformers torch datasets accelerate
```

## References

- DeepSeek-V2: https://github.com/deepseek-ai/DeepSeek-V2
- Our KVSplice experiments: docs/kvsplice.md
- Transform parameter analysis: docs/kvsplice/gpu-comparison-summary.md
