# Architecture Analysis: K/V Compression Dynamics

This document explains why compression strategy must match your model's attention architecture.

## The Core Finding

**"Don't mess with K"** - but only for standard attention.

![K/V Variance Comparison](../../plots/overview/kv_variance_comparison.png)

## Standard Attention: K is Fragile

In standard multi-head attention (Qwen, Mistral, Llama, GPT-2), K and V are computed
independently:

```
hidden → W_k → K    (independent projection)
hidden → W_v → V    (independent projection)
```

### Variance Spread Analysis

We measure "variance spread" as the ratio of maximum to minimum per-dimension
standard deviation:

```python
variance_spread = max(per_dim_std) / min(per_dim_std)
```

**Results for Qwen2.5-0.5B (24 layers):**

| Component | Variance Spread | Interpretation |
|-----------|-----------------|----------------|
| **K** | **132x** | Few dimensions dominate - fragile |
| **V** | 4x | Uniform importance - robust |
| **K/V Ratio** | **34:1** | K is 34x more sensitive |

### Why K is Fragile

K has extreme variance concentration:

```
K dimension importance (sorted):
dim 0:  ████████████████████████████████████████  (dominant)
dim 1:  ████████████████████
dim 2:  ████████████
dim 3:  ████████
  ...
dim 127: █  (negligible)
```

PCA compression removes the "least important" dimensions. But in K, those dominant
dimensions carry critical attention pattern information. Removing them collapses
the attention mechanism.

V has uniform importance:

```
V dimension importance (sorted):
dim 0:  ████████
dim 1:  ████████
dim 2:  ████████
  ...
dim 127: ████████
```

Any dimension can be compressed with similar impact.

### Compression Safety

| Target | Safe Compression | PPL Impact |
|--------|------------------|------------|
| V-only | **2.67x** | +0.99% |
| K-only | ~1.2x | +3-5% |
| K+V | ~1.5x | +2-4% |

**Recommendation**: Always use V-only compression for standard attention.

---

## MLA: K and V Are Equal

MLA (Multi-head Latent Attention) architectures like DeepSeek-V2 fundamentally
change the compression dynamics.

### Architecture Difference

```
Standard:
  hidden → W_k → K       (128 dims per head)
  hidden → W_v → V       (128 dims per head)
  Cache: K + V separately

MLA (DeepSeek-V2):
  hidden → kv_a_proj → shared_latent (512 dims) → kv_b_proj → [K, V]
  Cache: shared_latent only (4x built-in compression!)
```

MLA compresses K and V to a **shared latent space**, then expands via a learned
projection. This equalization happens during training.

### Variance Spread in MLA

**Results for DeepSeek-V2-Lite (27 layers):**

| Component | Variance Spread | Interpretation |
|-----------|-----------------|----------------|
| K | 1.4x | Uniform - robust |
| V | 1.6x | Uniform - robust |
| K/V Ratio | 0.9:1 | **Equal sensitivity** |

Both K and V have low, nearly identical variance spread.

### Why MLA Equalizes K/V

The learned expansion layer (`kv_b_proj`) produces K and V from the same source.
During training, the model learns to distribute information evenly across both
projections to minimize loss. This naturally equalizes their variance structure.

### Implications for MLA Compression

| Strategy | Standard Attention | MLA |
|----------|-------------------|-----|
| V-only | ✅ Optimal | ❌ No advantage |
| K-only | ❌ Risky | ✅ Equally safe |
| K+V | ⚠️ Caution | ✅ Both safe |
| Compress latent | N/A | ✅ **Best approach** |

**Recommendation for MLA**: Compress the shared KV latent directly. Don't use
V-only strategy - it provides no benefit over K compression.

---

## Analysis Script

Run the analysis yourself:

```bash
# Analyze MLA model (DeepSeek-V2)
python scripts/analyze_mla_kv_structure.py --model deepseek-ai/DeepSeek-V2-Lite

# Compare with standard attention
python scripts/analyze_kv_variance.py --model Qwen/Qwen2.5-0.5B
```

---

## Summary

| Architecture | K Spread | V Spread | Strategy |
|--------------|----------|----------|----------|
| Standard (Qwen, Mistral, Llama) | 132x | 4x | **V-only** |
| MLA (DeepSeek-V2) | 1.4x | 1.6x | **Compress latent** |

The "Don't mess with K" principle is architecture-specific. Always check your
model's attention type before choosing a compression strategy.
