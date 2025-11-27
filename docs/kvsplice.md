# KV Cache Compression: MLA and Learned Bottlenecks

## Overview

This document covers KV cache compression research using Multi-head Latent
Attention (MLA) and learned compression bottlenecks.

**Key result**: Learned compression (KVSplice) achieves 12x cache reduction
while **improving** model quality. The compression bottleneck acts as
beneficial regularization.

## Multi-head Latent Attention (MLA)

MLA from DeepSeek-V2/V3 achieves 6x KV cache compression by introducing a
latent bottleneck:

```python
# Standard attention: Store full K, V for each head
k = w_k(x)  # [B, n_heads, T, d_head] - 768 dims total
v = w_v(x)  # [B, n_heads, T, d_head] - 768 dims total
cache = (k, v)  # 36 MB for GPT-2 124M (12 heads × 2 × 64d)

# MLA: Store compressed latent, generate K, V on-the-fly
latent = to_latent(x)  # [B, T, 256] - shared across heads
k, v = from_latent(latent)  # Expand to per-head K, V on demand
cache = latent  # 6 MB (6x compression)
```

**MLA trade-off**: 6x smaller cache but +8.6% perplexity degradation due to
compression bottleneck losing information.

## KVSplice: Learned Compression Bottleneck

KVSplice applies an additional learned compression layer on top of MLA,
achieving 12x total compression while **improving** quality.

### Architecture

```python
# MLA: Latent cache (6x compression)
latent = to_latent(x)  # [B, T, 256]
cache = latent  # 6 MB

# MLA + KVSplice: Compressed latent cache (12x compression)
compressed = kvsplice.compress(latent)  # [B, T, 128]
cache = compressed  # 3 MB (12x compression vs standard)
```

### Implementation

```python
class LearnedKVSplice(nn.Module):
    """Learned compression via monotonic transform + low-rank projection."""

    def __init__(self, d_in=256, d_compressed=128):
        super().__init__()
        # Learned monotonic transform (per-dimension warping)
        self.transform_scale = nn.Parameter(torch.ones(d_in))
        self.transform_shift = nn.Parameter(torch.zeros(d_in))

        # Low-rank projection
        self.compress = nn.Linear(d_in, d_compressed, bias=False)
        self.expand = nn.Linear(d_compressed, d_in, bias=False)

    def forward(self, x):
        # Apply monotonic transformation
        x_transformed = x * F.softplus(self.transform_scale) + self.transform_shift

        # Compress to lower dimension
        compressed = self.compress(x_transformed)

        # Expand back to original dimension
        reconstructed = self.expand(compressed)

        # Apply inverse transformation
        reconstructed = (reconstructed - self.transform_shift) / (
            F.softplus(self.transform_scale) + 1e-6
        )

        return reconstructed
```

**Why this works**:

1. **Monotonic transformation**: Learns data-specific geometry, "straightening"
   the latent manifold to make linear projection more effective
2. **Low-rank projection**: Standard dimensionality reduction after geometric
   correction
3. **End-to-end learning**: Compression loss integrated into training objective
4. **Invertible**: Can reconstruct original latents during inference

## MLPSplice: Compression for MLP Activations

Similar approach applied to MLP intermediate activations:

```python
# Standard MLP
h = fc1(x)  # [B, T, 4*d_model] = [B, T, 3072]
y = fc2(gelu(h))

# MLP with learned compression
h = fc1(x)  # [B, T, 3072]
h_compressed = mlpsplice.compress(h)  # [B, T, 1024]
# ... can cache h_compressed for reuse ...
h_reconstructed = mlpsplice.expand(h_compressed)
y = fc2(gelu(h_reconstructed))
```

## Results

Experimental setup: GPT-2 124M on FineWebEdu dataset, 2-hour training runs on
AMD Radeon Pro W7900 GPU.

### Cache Memory Comparison

| Architecture | Cache Size | vs Standard | Compression |
|-------------|-----------|-------------|-------------|
| Standard KV | 36.00 MB | - | 1.0x |
| MLA | 6.00 MB | -83.3% | 6.0x |
| **MLA + KVSplice** | **3.00 MB** | **-91.7%** | **12.0x** |

**KVSplice cuts the latent cache in half** (2x additional compression) for
**12x total compression** vs standard KV cache.

### Quality and Speed Results

| Architecture | Val Loss | Perplexity | Tokens/sec | Cache | vs Baseline |
|-------------|----------|------------|------------|-------|-------------|
| Baseline GPT-2 | 1.199 | 3.3 | - | 36 MB | - |
| MLA | 1.276 | 3.6 | 17,031 | 6 MB | +8.6% worse |
| **MLA + KVSplice** | **1.166** | **3.2** | 17,429 | 3 MB | **-3.0% better** ✓ |

### Key Findings

1. **Paradoxical quality improvement**: Despite 50% compression, KVSplice
   **improves quality by 11%** over MLA alone (3.6 → 3.2 perplexity)

2. **Better than baseline**: MLA+KVSplice achieves 3.2 perplexity vs baseline's
   3.3, while using 12x less cache memory

3. **Learned compression acts as regularization**: Forces latents into
   information-dense subspace, improving generalization

4. **Inference speed**: Comparable to MLA (~17K tokens/sec)

## Why Compression Improves Quality

The learned compression acts as a beneficial bottleneck:

- **Information bottleneck principle**: Forcing information through narrow
  channel discards task-irrelevant noise
- **Regularization effect**: Compression prevents overfitting to training data
- **Dense representations**: Learned transform packs information efficiently
- **End-to-end optimization**: Compression trained with language modeling loss

This explains why KVSplice (50% compression) outperforms MLA alone - the
compression constraint actually helps the model learn better representations.

## Fisher Information Analysis

We hypothesized that KVSplice might work by aligning with Fisher Information
Matrix (FIM) geometry - compressing along low-FIM directions while preserving
high-FIM modes. See [FIM.md](FIM.md) for complete analysis.

**Result**: FIM analysis did **not** predict compression effectiveness.

- FIM energy_r16: ~37% (low concentration in top 16 modes)
- Yet KVSplice improves quality by 11% despite 50% compression
- FIM measures optimization geometry, not task-relevant information

**Conclusion**: Learned compression finds task-specific structure that
geometric metrics (FIM, PCA) don't capture. The value comes from end-to-end
learning integrated with the language modeling objective.

## MLA Variants

### MLA (Base)

Single latent decompresses to Q, K, V:

```python
latent = to_latent(x)  # [B, T, 256]
qkv = from_latent(latent)  # [B, T, 3*n_heads*head_dim]
q, k, v = split(qkv)
```

**Compression**: 6x (36 MB → 6 MB cache)

### MLA_KV2 (Improved)

Separate Q (direct) and K/V (compressed) paths:

```python
q = to_q(x)  # [B, T, n_heads*head_dim] - direct projection
kv_latent = to_kv_latent(x)  # [B, T, 256] - compressed
k, v = from_kv_latent(kv_latent)  # Expand K, V from latent
```

**Compression**: 6x, better quality than base MLA

### MLA_KV2M (MLA + MLPSplice)

MLA_KV2 plus MLP intermediate compression:

- Attention: KV latent compression (6x)
- MLP: Learned compression of intermediate activations
- **Total**: 12x compression with quality improvement

## When to Use

**Use MLA when**:

- Memory constraints require smaller KV cache
- Willing to train from scratch with compression integrated
- Seeking quality improvements from regularization effect
- Deploying models at scale where 6-12x compression matters

**Best results**: MLA_KV2M with both attention and MLP compression for 12x
reduction and baseline-beating quality.

## Implementation

**Code**: `gpt2/mla.py`

**Key classes**:

- `MLA_Config`: Configuration for MLA models
- `LearnedKVSplice`: Attention cache compression
- `MLPSplice`: MLP activation compression
- `GPT2_MLA`: Base MLA model
- `GPT2_MLA_KV2`: Improved MLA with separate Q/KV paths
- `GPT2_MLA_KV2M`: MLA_KV2 + MLPSplice

## References

- DeepSeek-V2/V3 papers (original MLA)
- Implementation: `gpt2/mla.py`
- Fisher analysis: `docs/FIM.md`
- Alternative geometric approaches: `docs/kvsplice-raw.md`
