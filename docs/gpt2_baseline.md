# GPT-2 Training Baseline Configuration

## Historical Baseline (September 2025)

### Hardware
- **GPU**: AMD Radeon Pro W7900 (48GB VRAM)
- **System**: PCIe Gen 4
- **ROCm**: Older version (pre-TheRock)
- **PyTorch**: Pre-2.10 (exact version unknown)

### Configuration
```
Model: GPT-2 124M
Dataset: FineWebEdu
Block size: 1024 tokens
Batch size: 32
Gradient accumulation: 4
Effective batch size: 128 (32 × 4)

Optimizer: AdamW / AdamWSPAM / AdamWPrune
Learning rate: 6e-4
Weight decay: 0.1
Warmup steps: 2000
LR decay: Cosine to 6e-5

Max iterations: 50000
Eval interval: 500
Mixed precision: bfloat16
torch.compile(): ENABLED ✓
```

### Evidence of torch.compile() Working

From `gpt2_adamwprune_bitter0_state_50/output.log`:

```
Compiling model with torch.compile()...
Batch size: 32, Gradient accumulation: 4
Effective batch size: 128

Iter     0 | 931.7ms/iter   ← Initial iteration
Iter    10 | 7169.3ms/iter  ← Compilation happening (7.7x slower)
Iter    20 | 2754.5ms/iter  ← Compiled code active
Iter    30 | 2748.2ms/iter  ← Stabilized
Iter    40+ | ~2740ms/iter  ← Consistent 62% speedup vs compilation
```

**Classic torch.compile() pattern**: Massive slowdown at ~10 iterations for graph compilation, then consistent speedup.

### Results

**AdamWSPAM + magnitude pruning @ 50% sparsity:**
- **Perplexity: 42.82** @ 10k iterations (reference baseline)

**GPU Memory Usage:**
- Mean: 25.3 GiB
- Max: 25.3 GiB
- Training memory: ~3.03x model weights (includes optimizer states)

---

## Current System (October 2025)

### Hardware
- **GPU**: AMD Radeon Pro W7900 (48GB VRAM)
- **System**: PCIe Gen 5 (higher bandwidth)
- **CPU Memory**: Increased capacity
- **ROCm**: TheRock (custom build)
- **PyTorch**: 2.10+

### Known Issues

**torch.compile() OOM on PyTorch 2.10:**
```
torch.OutOfMemoryError: HIP out of memory. Tried to allocate 3.07 GiB.
GPU 0 has a total capacity of 44.98 GiB of which 0 bytes is free.
Of the allocated memory 41.74 GiB is allocated by PyTorch
```

**Error occurs in**: Backward pass during inductor compilation
**Location**: `torch._inductor.output_code.py:617`
**Buffer allocation**: `empty_strided_cuda((32, 1024, 50304), ...)`

**Status**: Potential TheRock regression or PyTorch 2.10 memory behavior change

---

## Recommended Configuration (Current System)

### Option 1: Match Historical Baseline (Preferred if working)
```bash
CONFIG_BATCH_SIZE=32
CONFIG_GPT2_GRADIENT_ACCUMULATION=4
CONFIG_GPT2_COMPILE=y
CONFIG_COMPILE_MODEL=y
```

**If this works**: Great! TheRock is compatible.
**If this OOMs**: File TheRock regression, use Option 2.

### Option 2: Fallback for TheRock Issues
```bash
CONFIG_BATCH_SIZE=32
CONFIG_GPT2_GRADIENT_ACCUMULATION=4
CONFIG_GPT2_COMPILE=n
CONFIG_COMPILE_MODEL=n
```

**Trade-off**: ~60% slower iteration time, but guaranteed to work.
**Use case**: Temporary workaround while TheRock torch.compile() is investigated.

### Option 3: Reduce Batch Size (Last Resort)
```bash
CONFIG_BATCH_SIZE=16
CONFIG_GPT2_GRADIENT_ACCUMULATION=8  # Keep effective batch = 128
CONFIG_GPT2_COMPILE=n
```

**Only use if**: Both Options 1 and 2 fail (unlikely).

---

## Kconfig Constraints

**CRITICAL**: `/data/AdamWPrune/Kconfig` was changed from:
```
config BATCH_SIZE
    range 32 2048  # OLD - enforced minimum of 32
```

To:
```
config BATCH_SIZE
    range 1 2048   # NEW - allows any positive batch size
```

**Why**: The old range silently clamped `CONFIG_BATCH_SIZE=16` to 32, causing confusion.

**Historical impact**: All test runs used batch_size=32 regardless of what defconfig specified, because Kconfig enforced the minimum.

---

## Defconfig Files

### Vanilla Baseline
**Path**: `defconfigs/gpt2-vanilla-baseline`
**Purpose**: Control experiment for MLA comparison
**Config**: Matches historical baseline exactly
**Compile**: Enabled (matches old system)

### MLA Baseline
**Path**: `defconfigs/gpt2-ra-mla-baseline`
**Purpose**: Test MLA compression vs vanilla
**Config**: Same as vanilla except:
- `CONFIG_ENABLE_RA_MLA=y`
- `CONFIG_RA_MLA_LATENT_DIM=128`
- `CONFIG_RA_MLA_RA_ALPHA="0.0"` (pure MLA, no reciprocal attention)
- `CONFIG_GPT2_COMPILE=n` (inductor bug with RA+MLA)

---

## RA+MLA Baseline (Experimental)

### Configuration
```
RA+MLA enabled: yes
Latent dimension: 128 (less aggressive compression than 64)
RA alpha: 0.0 (pure MLA, no reciprocal attention)
RA window: 64
Per-head Q latent: yes
Per-head V up-projection: yes
```

### Why alpha=0.0?

Previous test with alpha=0.5 (reciprocal attention) diverged catastrophically:
```
Iter   0 | loss 3.2
Iter 300 | loss 5.9  ← Training collapsed
```

Pure MLA (alpha=0.0) removes the unstable reciprocal attention component.

### Why latent_dim=128?

- **128**: Less aggressive compression, reduces information loss
- **64**: More aggressive compression (tested with Shakespeare, caused overfitting)

---

## Verification Checklist

Before running any GPT-2 experiment:

- [ ] Verify `CONFIG_BATCH_SIZE=32` in config.py
- [ ] Verify `CONFIG_GPT2_GRADIENT_ACCUMULATION=4` in config.py
- [ ] Check effective batch size in logs should be **128**
- [ ] For vanilla: `CONFIG_GPT2_COMPILE` should match system capability
- [ ] For MLA: `CONFIG_GPT2_COMPILE=n` (inductor bug)
- [ ] Dataset: `CONFIG_GPT2_DATASET="finewebedu"` (not Shakespeare!)
- [ ] Block size: `CONFIG_GPT2_BLOCK_SIZE=1024`
- [ ] Max iters: `CONFIG_GPT2_MAX_ITERS=50000`

---

## Common Pitfalls

### 1. Batch Size Mismatch
**Symptom**: Logs show batch_size=32 even though defconfig says 16
**Cause**: Kconfig range constraint (fixed as of October 2025)
**Solution**: Check actual config.py, not just defconfig

### 2. Dataset Confusion
**Symptom**: Fast convergence but terrible validation loss
**Cause**: Training on Shakespeare (tiny) instead of FineWebEdu
**Solution**: Verify `CONFIG_GPT2_DATASET="finewebedu"`

### 3. torch.compile() Silent Failure
**Symptom**: Can't find "Compiling model..." in logs
**Cause**: Logs not preserved, or compile silently disabled
**Solution**: Check iteration timings for compilation pattern (7x slower iteration around iter 10)

### 4. MLA Training Divergence
**Symptom**: Loss explodes from 3.x to 5.x+
**Cause**: `ra_alpha > 0` (reciprocal attention is unstable)
**Solution**: Use `CONFIG_RA_MLA_RA_ALPHA="0.0"` for pure MLA

---

## Historical Test Matrix Results

**Location**: [knlp-key-results repository](https://github.com/mcgrof/knlp-key-results/tree/master/key_results/test_matrix_results_20250923_010926/)

**Tests run**:
- `gpt2_adamwprune_bitter0_state_50`
- `gpt2_adamwprune_bitter1_state_50`
- `gpt2_adamwprune_bitter2_state_50`
- `gpt2_adamwspam_magnitude_50`

**Configuration used** (from config.txt):
```
CONFIG_BATCH_SIZE=32
CONFIG_GPT2_GRADIENT_ACCUMULATION=4
CONFIG_GPT2_COMPILE=y
CONFIG_COMPILE_MODEL=y
```

**GPU Memory**: ~25.3 GiB per run

---

## Target Performance

### Baseline to Beat
**AdamWSPAM + magnitude pruning:**
- Perplexity: **42.82** @ 10k iterations
- Sparsity: 50%
- This is the reference for all GPT-2 experiments

### MLA Goals
- Match or beat 42.82 perplexity
- Demonstrate latent compression benefits
- Stable training (no divergence)
- Fair comparison (same hyperparameters as vanilla)

---

## Future Work

### If torch.compile() OOMs on TheRock
1. File issue with PyTorch 2.10 memory regression
2. Test on older PyTorch to verify
3. Profile memory usage during compilation
4. Consider alternative compilation strategies

### Bitter10 Integration
**Status**: Code ready, in test queue after RA+MLA
**Blocker**: Memory violation (3x overhead)
**Alternative**: bitter10z (zero-memory variant)

---

## References

- Historical results: [knlp-key-results repository](https://github.com/mcgrof/knlp-key-results)
- Old baseline config: `gpt2/defconfigs/gpt2-finewebedu-w7900`
- New baselines: `defconfigs/gpt2-vanilla-baseline`, `defconfigs/gpt2-ra-mla-baseline`
- RA+MLA integration: `docs/bitter10_integration.md`
- Kconfig: `Kconfig` (batch_size range fixed October 2025)

---

**Last updated**: October 19, 2025
**System**: TheRock (PyTorch 2.10+, ROCm latest, PCIe Gen 5)
**Status**: Baseline configuration documented, ready for testing
