# Key-Value (KV) Tying in Attention

**33% Parameter Reduction in Attention Layers**

This document explains KV tying - a weight tying technique that shares the same projection for both attention keys and values. Part of our broader [weight tying infrastructure](weight-tying.md).

---

## Table of Contents

1. [Overview](#overview)
2. [The Standard Attention Mechanism](#the-standard-attention-mechanism)
3. [KV Tying: Sharing K and V Projections](#kv-tying-sharing-k-and-v-projections)
4. [Implementation Details](#implementation-details)
5. [Ablation Study Results](#ablation-study-results)
6. [When to Use KV Tying](#when-to-use-kv-tying)
7. [References](#references)

---

## Overview

**Key-Value (KV) tying** is a parameter reduction technique where the key projection (`W_k`) and value projection (`W_v`) in self-attention share the same weights. Instead of learning separate transformations for keys and values, we compute only one projection and use it for both:

```python
# Standard attention:
K = x @ W_k  # Separate key projection
V = x @ W_v  # Separate value projection

# KV-tied attention:
V = x @ W_v  # Single projection
K = V        # Reuse for keys!
```

**Parameter reduction:**
- Standard QKV: 3 projections × `n_embd × n_embd` parameters
- KV-tied QKV: 2 projections × `n_embd × n_embd` parameters
- **Savings: 33% of attention projection parameters**

**For general weight tying concepts**, see **[Weight Tying Documentation](weight-tying.md)**.

**For reciprocal attention mechanisms**, see **[RA Documentation](ra.md)**.

---

## The Standard Attention Mechanism

### Multi-Head Self-Attention (Baseline)

Before understanding KV tying, let's review standard multi-head attention:

```python
class CausalSelfAttention(nn.Module):
    """Standard GPT-2 attention with separate Q, K, V projections"""

    def __init__(self, config):
        super().__init__()
        # Single linear layer projects to [Q | K | V] concatenated
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.n_head = config.n_head
        self.n_embd = config.n_embd

    def forward(self, x):
        B, T, C = x.size()  # Batch, sequence length, embedding dim

        # Project and split into Q, K, V
        qkv = self.c_attn(x)  # (B, T, 3*C)
        q, k, v = qkv.split(self.n_embd, dim=2)  # Each: (B, T, C)

        # Reshape for multi-head attention
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        # Now: (B, n_head, T, head_dim)

        # Scaled dot-product attention
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)

        # Recombine heads and project
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.c_proj(y)
        return y
```

### Attention Computation Breakdown

**Step 1: Project inputs to Q, K, V**
```python
# Input: x (B, T, 768)
# Weight: W_qkv (768, 2304)  # 3 × 768 = 2304
# Output: qkv (B, T, 2304)

qkv = x @ W_qkv  # Linear projection

# Split into three equal parts:
q = qkv[:, :, :768]       # Query:  (B, T, 768)
k = qkv[:, :, 768:1536]   # Key:    (B, T, 768)
v = qkv[:, :, 1536:]      # Value:  (B, T, 768)
```

**Step 2: Compute attention weights**
```python
# Q and K interact to determine "what to attend to"
scores = q @ k.transpose(-2, -1) / sqrt(head_dim)  # (B, n_head, T, T)
attn_weights = softmax(scores, dim=-1)             # Normalize per row
```

**Step 3: Aggregate values**
```python
# Attention weights aggregate values
output = attn_weights @ v  # (B, n_head, T, head_dim)
```

**Key observation**: K and V play different roles:
- **Keys**: Determine attention pattern (compared with queries)
- **Values**: Provide content to be aggregated (weighted by attention)

But do they **need** separate projections?

---

## KV Tying: Sharing K and V Projections

### The Core Idea

**Hypothesis**: Keys and values can share the same projection without significantly harming performance.

**Rationale**:
1. Both K and V transform the same input (x)
2. Both operate in the same embedding space
3. The attention mechanism may be robust to using identical K/V projections

**Implementation**: Compute only V projection, then set K = V.

### Modified Architecture

```python
class CausalSelfAttention(nn.Module):
    """Attention with optional KV tying"""

    def __init__(self, config):
        super().__init__()

        # When KV tying enabled: only Q and V (not K)
        if config.kv_tying:
            qkv_dim = 2 * config.n_embd  # Q + V only
        else:
            qkv_dim = 3 * config.n_embd  # Q + K + V

        self.c_attn = nn.Linear(config.n_embd, qkv_dim, bias=config.bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)

        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.kv_tying = config.kv_tying

    def forward(self, x):
        B, T, C = x.size()

        if self.kv_tying:
            # KV-tied path: only compute Q and V
            qkv = self.c_attn(x)  # (B, T, 2*C) instead of 3*C
            q, v = qkv.split(self.n_embd, dim=2)  # Split into two

            # Reshape for multi-head
            q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
            v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

            # Set K = V (the tying!)
            k = v

        else:
            # Standard path: compute Q, K, V separately
            qkv = self.c_attn(x)
            q, k, v = qkv.split(self.n_embd, dim=2)

            q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
            k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
            v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        # Attention computation is identical
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)

        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.c_proj(y)
        return y
```

### What Changes with KV Tying?

**Before (standard attention)**:
```python
# Three separate learned transformations:
Q = x @ W_q  # Query:  learns "what to look for"
K = x @ W_k  # Key:    learns "what can be found"
V = x @ W_v  # Value:  learns "what content to retrieve"

# K and V can diverge during training:
W_k ≠ W_v
```

**After (KV-tied attention)**:
```python
# Two learned transformations:
Q = x @ W_q  # Query:  learns "what to look for"
V = x @ W_v  # Value:  learns both "what can be found" AND "what to retrieve"
K = V        # Key:    uses the same projection as value

# K and V are always identical:
K = V  (same tensor!)
```

### Attention Pattern Implications

**Standard attention**:
```python
attn_scores = Q @ K^T  # Query compared against independent keys
output = attn_weights @ V  # Weighted aggregation of independent values
```

**KV-tied attention**:
```python
attn_scores = Q @ V^T  # Query compared against VALUES directly!
output = attn_weights @ V  # Weighted aggregation of the same V
```

**Key insight**: With KV tying, the attention pattern is computed by comparing queries against the **values themselves**, not against a separate key representation.

This means:
- ✅ **Simpler**: Fewer parameters to learn
- ❓ **Less flexible**: Can't learn separate "addressing" vs "content" representations
- ❓ **May hurt performance**: Depends on whether separate K/V is necessary

---

## Implementation Details

### Kconfig Integration

KV tying is controlled via Kconfig:

**File**: `gpt2/Kconfig`

```kconfig
config GPT2_KV_TYING
	bool "Use key-value tying in attention"
	default n
	help
	  Tie key and value projections in attention (K = V).
	  Instead of creating separate linear transformations for K and V,
	  only create V and use it for both K and V.
	  This reduces parameters in attention layers by ~33%.
	  Attention computation proceeds normally after tying.
```

**Usage**:
```bash
# Enable via menuconfig:
make menuconfig
# Navigate to: GPT-2 Configuration → Advanced options → KV tying

# Or use defconfig:
make defconfig-gpt2-kv-tying-w7900-ablation
```

### Parameter Count Comparison

For GPT-2 124M (12 layers, 12 heads, 768 embedding):

| Component | Standard | KV-Tied | Reduction |
|-----------|----------|---------|-----------|
| **Single layer QKV projection** | 768 × 2304 = 1.77M | 768 × 1536 = 1.18M | **33%** |
| **All 12 layers QKV** | 21.2M | 14.2M | **7M params** |
| **Total model** | 124M | 117M | **5.6%** |

**Note**: Since QKV is only part of the attention mechanism (output projection and MLPs remain unchanged), the overall model reduction is smaller than 33%.

### Memory and Compute Savings

**Training memory**:
```python
# Standard QKV:
params_qkv = n_layers × n_embd × (3 × n_embd) = 12 × 768 × 2304 = 21.2M params
memory_qkv = 21.2M × 4 bytes (FP32) = 84.8 MB

# KV-tied QKV:
params_qkv = n_layers × n_embd × (2 × n_embd) = 12 × 768 × 1536 = 14.2M params
memory_qkv = 14.2M × 4 bytes (FP32) = 56.6 MB

# Savings: 28.2 MB (~33%)
```

**Inference KV cache** (NOT affected!):
```python
# KV cache stores actual K and V tensors during inference
kv_cache_size = 2 × n_layers × batch × seq_len × n_embd

# With KV tying:
K_cached = V_cached  # Same values, but still stored separately for SDPA

# No inference cache savings! (K = V doesn't reduce cache size)
```

---

## Ablation Study Results

### Experimental Setup

**Defconfig**: `gpt2/defconfigs/gpt2-kv-tying-w7900-ablation`

**GPU**: AMD Radeon Pro W7900 (48GB VRAM)

**Ablation steps**:
- **V0**: Baseline GPT-2 (standard attention, K ≠ V)
- **V1**: GPT-2 with KV tying (K = V)

**Training**:
- Dataset: Shakespeare (tiny, ~1MB)
- Iterations: 10,000
- Batch size: 40 (optimized for W7900)
- Learning rate: 6e-4
- No torch.compile (ROCm compatibility)
- No flash attention (ROCm stability)

**Metrics tracked**:
- Final validation loss
- Final validation perplexity
- Training time
- Parameter count
- GPU memory usage

### Results

| Metric | V0 (Baseline) | V1 (KV Tying) | Change |
|--------|--------------|---------------|--------|
| **Val Loss** | TBD | TBD | TBD |
| **Val Perplexity** | TBD | TBD | TBD |
| **Parameters** | 124M | 117M | **-5.6%** |
| **Training Time** | TBD | TBD | TBD |
| **GPU Memory** | TBD | TBD | TBD |

**Status**: Ablation study in progress. Results will be updated here once complete.

### Running the Ablation

```bash
# Load KV tying ablation config
make defconfig-gpt2-kv-tying-w7900-ablation

# Run both V0 and V1 sequentially
make

# Results saved to:
# - ./output/final_model_stepV0.pt (baseline)
# - ./output/final_model_stepV1.pt (KV tying)
# - ./output/training_metrics_stepV0.json
# - ./output/training_metrics_stepV1.json

# W&B project: gpt2-kv-tying-w7900
```

### Expected Outcomes

**Hypothesis 1**: KV tying reduces parameters with minimal quality loss
- **Expected**: V1 perplexity ≤ 5% worse than V0
- **Rationale**: Attention mechanism may be overparameterized

**Hypothesis 2**: Training speed increases due to fewer parameters
- **Expected**: V1 trains 10-15% faster than V0
- **Rationale**: Smaller QKV projection means fewer FLOPs per layer

**Hypothesis 3**: GPU memory usage decreases
- **Expected**: V1 uses 5-10% less GPU memory
- **Rationale**: Fewer parameters = smaller optimizer state

### Mechanistic Interpretability Analysis

After training, analyze both checkpoints with mechint:

```bash
# Analyze KV circuits in both models
make defconfig-gpt2-kv-tying-w7900-ablation-mechint MODELS=./output
make mechint

# Compare circuit sparsity patterns:
# - V0: Standard attention may show different K/V importance patterns
# - V1: KV-tied attention must use identical patterns

# See docs/mechint.md for analysis workflow
```

---

## When to Use KV Tying

### ✅ Consider KV Tying When:

**1. Parameter budget is tight**
- Edge deployment (mobile, embedded)
- Memory-constrained GPUs (consumer GPUs with <16GB)
- Large models where every 1% reduction matters

**2. Attention mechanism is overparameterized**
- Small datasets (may overfit with full QKV)
- Simple tasks where standard attention is overkill
- Models where attention heads show redundancy

**3. Training from scratch**
- KV tying changes architecture fundamentally
- Easier to train from scratch than retrofit to existing checkpoints
- Allows optimizer to adapt to tied structure

### ❌ Avoid KV Tying When:

**1. Performance is critical**
- Research baselines (standard attention is the default)
- Production models where quality > size
- Competitive benchmarks where every 0.1 perplexity matters

**2. Keys and values need different representations**
- Some tasks may benefit from separate K/V (hypothesis!)
- Standard attention has been validated across billions of parameters
- KV tying is still experimental

**3. Fine-tuning pretrained models**
- Pretrained checkpoints use standard QKV
- Can't apply KV tying without full retraining
- Better to use LoRA or other parameter-efficient fine-tuning

### Alternatives to Consider

Instead of KV tying, consider:

**1. [Low-Rank Attention (LoRA)](https://arxiv.org/abs/2106.09685)**
```python
# Reduce QKV dimensions with low-rank decomposition
Q = x @ (W_q_down @ W_q_up)  # Factorized projection
# Reduces parameters while keeping Q, K, V separate
```

**2. [Multi-Query Attention (MQA)](https://arxiv.org/abs/1911.02150)**
```python
# Share K and V across heads, but keep per-head queries
Q = x @ W_q  # Per-head queries
K = x @ W_k_shared  # Shared key across heads
V = x @ W_v_shared  # Shared value across heads
```

**3. [Grouped-Query Attention (GQA)](https://arxiv.org/abs/2305.13245)**
```python
# Interpolate between MHA and MQA: group heads sharing K/V
# Reduces KV cache size while maintaining quality
```

---

## Relationship to Other Tying Methods

### Input-Output Embedding Tying

**Different purpose** - see [weight-tying.md](weight-tying.md):

```python
# Input-output tying: embedding ↔ language model head
wte.weight = lm_head.weight  # 50% reduction in embedding params

# KV tying: attention keys ↔ attention values
K = V  # 33% reduction in QKV params

# These are independent! Can enable both:
config.weight_tying = True   # Input-output tying
config.kv_tying = True       # KV tying
```

### Reciprocal Attention (RA) Tying

**Related but different** - see [ra.md](ra.md):

**RA** uses fused Q/K representations for reciprocal attention:
```python
# RA folds Q and K into single projection:
Qf = [Q_std | K_low]  # Folded query
Kf = [K_std | Q_low]  # Folded key

# Computes both Q@K^T and K@Q^T in single SDPA call
# This is NOT the same as K = V!
```

**R-MLP tying** (ablation V2) ties reciprocal MLP to attention projection:
```python
# Experimental: share R-MLP weights with attention output
c_fc_rec = attention.c_proj  # Reciprocal pathway reuses attention weights
```

See [RA weight tying section](ra.md#weight-tying-in-reciprocal-architectures) for details.

---

## References

### Related Documentation

- **[Weight Tying](weight-tying.md)**: General weight tying concepts and patterns
- **[Reciprocal Attention](ra.md)**: Advanced attention mechanisms with geometric initialization
- **[Mechanistic Interpretability](mechint.md)**: Analyzing learned KV patterns

### Research Papers

**Multi-Query/Grouped-Query Attention** (related approaches):
- [Fast Transformer Decoding](https://arxiv.org/abs/1911.02150) - Multi-Query Attention (MQA)
- [GQA: Training Generalized Multi-Query Attention](https://arxiv.org/abs/2305.13245) - Grouped-Query Attention
- [Multi-Head Latent Attention](https://arxiv.org/abs/2210.05144) - Latent attention bottlenecks

**Parameter-Efficient Transformers**:
- [LoRA](https://arxiv.org/abs/2106.09685) - Low-rank adaptation
- [ALBERT](https://arxiv.org/abs/1909.11942) - Cross-layer parameter sharing
- [DeBERTa](https://arxiv.org/abs/2006.03654) - Disentangled attention (no KV tying)

### Code References

**Our implementation**:
- `gpt2/model.py`: `CausalSelfAttention` class with KV tying support
- `gpt2/Kconfig`: `CONFIG_GPT2_KV_TYING` option
- `gpt2/defconfigs/gpt2-kv-tying-w7900-ablation`: Ablation study configuration

**Ablation studies**:
- V0 vs V1: KV tying comparison (in progress)
- Compare with: `make defconfig-gpt2-kv-tying-w7900-ablation-mechint MODELS=./output`

---

## Summary

**KV tying** is a simple parameter reduction technique that sets `K = V` in self-attention:

✅ **Benefits**:
- 33% reduction in QKV projection parameters
- ~5-6% overall model size reduction
- Faster training due to fewer FLOPs
- Lower GPU memory for optimizer states

❓ **Trade-offs**:
- Less flexibility: can't learn separate key/value representations
- Quality impact unknown: ablation study in progress
- Inference cache NOT reduced (KV cache still stores separate K and V)

**Recommendation**: Use KV tying when parameter budget is constrained and quality loss is acceptable. Always validate with ablation studies before production deployment.

**For broader context on weight tying**, see **[weight-tying.md](weight-tying.md)**.
