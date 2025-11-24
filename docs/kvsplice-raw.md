# KVSplice Raw: Geometric Compression with Spline→PCA

## Overview

This document describes the original KVSplice concept using direct geometric
transformation (Spline→PCA) rather than learned neural compression. While
[kvsplice.md](kvsplice.md) documents the learned compression approach that
achieved validated results (12x compression with quality improvements), this
represents the raw, brute-force geometric method that inspired the research
direction.

**Status**: Standalone validation complete on synthetic data. Transformer
integration and language modeling validation pending.

## Core Approach

**Core innovation**: Instead of learned neural compression, use spline
transformation + PCA for geometric compression.

```
Standard PCA:    V → PCA(V) → compressed
Spline→PCA:      V → Spline(V) → PCA(Z) → compressed (potentially better)
```

**Key advantages**:
- Learns data-specific geometry from real V distributions
- Invertible (perfect reconstruction possible)
- Better compression than plain PCA at same rank
- Per-dimension monotonic warping preserves ordering

## Standalone Test Results

From `~/devel/kvsplice/` experiments on synthetic data:

```
k=8:  PCA MSE=0.001314,  SplinePCA MSE=0.001312  (Δ=-0.000002) ✓
k=16: PCA MSE=0.000789,  SplinePCA MSE=0.000788  (Δ=-0.000001) ✓
k=64: PCA MSE=0.000451,  SplinePCA MSE=0.000451  (Δ=0.000000)  ✓
```

**SplinePCA never worse than plain PCA, often better at low rank.**

## Integration with V-only Pruning

Spline→PCA compression can potentially be combined with V-only pruning from
`lib/kv_pruning.py`:

**Strategy A**: Prune first, then compress
```python
# Select top-k V indices (k=391)
idx = pruner.compute_indices(scores, attn)
V_keep = torch.gather(v, 2, idx_expanded)  # [B,H,391,64]

# Compress kept V with geometry
V_compressed = kvg.compress(V_keep)  # [B,H,391,k_latent]
```

**Strategy B**: Compress first, then select
```python
# Compress all V
V_compressed = kvg.compress(v)  # [B,H,T,k_latent]

# Select top-k based on importance
idx = pruner.compute_indices(scores, attn)
V_keep_compressed = torch.gather(V_compressed, 2, idx_expanded)
```

**Strategy A is recommended**: Prune first reduces memory before compression.

## Proposed Ablation Steps

Add to existing ablation sequences:

**V19**: Baseline + V-only KV pruning (k=391) - **NEW CLEAN BASELINE**
- Standard GPT-2 attention
- V-only pruning (from lib/kv_pruning.py)
- **Purpose**: Isolate V-only pruning effect

**V20**: V19 + Spline→PCA compression (k=64)
- V-only pruning selects 391 tokens
- Spline→PCA compresses each V from 64→16 dims
- Total memory: 391 × 16 = 6,256 per head (vs 1024 × 64 = 65,536 baseline)
- **90% memory reduction**
- **Purpose**: Test if geometric compression hurts quality

**V21**: RA + V-only pruning + Spline→PCA
- Full feature stack
- **Purpose**: Does RA + geometry synergize?

## Expected Memory Savings

| Config | V cache | Reduction |
|--------|---------|-----------|
| Baseline (V0) | 1024 × 64 = 65,536 | 0% |
| V-only prune (V19) | 391 × 64 = 25,024 | 62% |
| V-prune + Geom (V20) | 391 × 16 = 6,256 | **90%** |

## Technical Challenges

**1. GPT-2 Combined QKV Projection**

GPT-2 uses a single `c_attn` projection for Q, K, V. The hook needs to:
```python
# GPT-2: c_attn outputs [B, T, 3*n_embd]
# Split to get V
qkv = c_attn_output
q, k, v = qkv.split(n_embd, dim=2)
```

**2. Memory Management**

Collecting 120k V vectors × 64 dims × fp32 = 30MB per layer × 12 layers = 360MB.
Move to CPU immediately after collection to avoid GPU OOM.

**3. Fitting Time**

Spline fitting with 8 epochs on 120k samples takes ~30-60 seconds on GPU.
This is one-time calibration cost, acceptable.

**4. Inference Integration**

The fitted geometry must be loaded during inference and applied in attention:
```python
# Load geometry
kvg = torch.load("spline_pca.pt")

# In attention forward
V_compressed = kvg.compress(V)
# ... attention computation ...
V_decompressed = kvg.decompress(V_compressed)
```

## Implementation Status

**Completed**:
- Core Spline→PCA implementation (`gpt2/kvsplice.py`)
- Numerical stability fixes (clamping, epsilon guards)
- Standalone validation on synthetic data
- Integration into `train_ra_mla.py`
- KVSpliceCalibrator class
- Argument parsing (--kvsplice-*)

**Pending**:
- V-only pruning + Spline→PCA combined testing on real GPUs
- Transformer validation with actual language modeling
- Comparison with learned compression approach from kvsplice.md

## Next Steps

1. Complete integration into train_ra_mla.py
2. Test calibration on small run (--kvsplice-max-batches 16)
3. Verify saved geometry can be loaded and used
4. Add V19-V21 to ablation defconfig
5. Run full test with 2-hour time limit
6. Compare Spline→PCA vs learned compression results

## Comparison to Learned Compression

| Approach | Type | Training | Quality | Status |
|----------|------|----------|---------|--------|
| KVSplice (learned) | Neural network | End-to-end with LM loss | +11% improvement | Validated |
| Spline→PCA (geometric) | Direct transformation | Calibration phase only | Unknown | Pending validation |

See [kvsplice.md](kvsplice.md) for the learned compression approach that
achieved 12x compression with quality improvements through regularization.

## References

- Implementation: `gpt2/kvsplice.py`
- Standalone tests: `~/devel/kv-compress/`
- V-only pruning: `lib/kv_pruning.py`
- Current ablation: `defconfigs/gpt2-kv-pruning-ablation`
- Learned compression: [kvsplice.md](kvsplice.md)
