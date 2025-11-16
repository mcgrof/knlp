# Key-Value (KV) Tying in Attention

**33% Parameter Reduction in Attention Layers**

This document explains KV tying - a weight tying technique that shares the same projection for both attention keys and values. Part of our broader [weight tying infrastructure](weight-tying.md).

---

## Table of Contents

1. [Overview](#overview)
2. [The Standard Attention Mechanism](#the-standard-attention-mechanism)
3. [KV Tying: Sharing K and V Projections](#kv-tying-sharing-k-and-v-projections)
4. [KV Transpose Tying: K = V.T](#kv-transpose-tying-k--vt)
5. [Implementation Details](#implementation-details)
6. [Ablation Study Results](#ablation-study-results)
7. [When to Use KV Tying](#when-to-use-kv-tying)
8. [References](#references)

---

## Overview

**Key-Value (KV) tying** is a parameter reduction technique where the key projection (`W_k`) and value projection (`W_v`) in self-attention share related weights. Instead of learning separate independent transformations for keys and values, we compute only one projection and derive the other from it.

**Two variants**:

**Identity tying (K = V)**:
```python
# Standard attention:
K = x @ W_k  # Separate key projection
V = x @ W_v  # Separate value projection

# Identity-tied attention:
V = x @ W_v  # Single projection
K = V        # Reuse exact same tensor!
```

**Transpose tying (K = V.T)** - NEW:
```python
# Transpose-tied attention:
V = x @ W_v     # Value projection
K = x @ W_v.T   # Transposed value projection!
```

**Parameter reduction** (both variants):
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

## KV Transpose Tying: K = V.T

### The Core Idea

**Alternative hypothesis**: Instead of setting K = V, use the **transpose** of the value projection for keys.

**Approach**: Compute only the value projection `W_v`, then use its transpose `W_v.T` for key projection:

```python
# Standard attention:
K = x @ W_k  # Separate learned key projection
V = x @ W_v  # Separate learned value projection

# KV identity tying (K = V):
V = x @ W_v
K = V        # Reuse exact same projection

# KV transpose tying (K = V.T):
V = x @ W_v     # Value projection
K = x @ W_v.T   # Transposed value projection (NEW!)
```

**Parameter reduction**: Same as identity tying (33% reduction), but with different learned representations.

### Why Transpose?

**Geometric intuition**:

Standard attention learns two independent transformations of the input space:
- **W_k**: Transforms input into "key space" (what can be found)
- **W_v**: Transforms input into "value space" (what content to retrieve)

With transpose tying, we introduce a **structural relationship** between these spaces:

```python
# Key transformation is the transpose of value transformation
W_k = W_v.T

# This creates a symmetric relationship:
K = x @ W_v.T
V = x @ W_v

# In attention computation:
attn_scores = Q @ K.T = Q @ (x @ W_v.T).T = Q @ W_v @ x.T
```

**Potential benefits**:

1. **Preserves distinctness**: K and V are different tensors (unlike K = V identity tying)
2. **Structural constraint**: W_k = W_v.T enforces a specific geometric relationship
3. **Symmetric transformation**: May encourage balanced key-value representations
4. **Reduced overfitting**: Constraint on W_k may act as regularization

**Potential drawbacks**:

1. **Still restrictive**: Can't learn fully independent K and V transformations
2. **Unproven**: No existing literature validating this approach
3. **Transpose semantics unclear**: Whether transpose relationship helps or hurts is unknown

### Implementation

```python
class CausalSelfAttention(nn.Module):
    """Attention with optional KV transpose tying"""

    def __init__(self, config):
        super().__init__()

        # Three variants:
        # 1. Standard: Q, K, V all separate (3 × n_embd)
        # 2. KV identity tying: Q, V separate, K = V (2 × n_embd)
        # 3. KV transpose tying: Q, V separate, K = V.T (2 × n_embd)

        if config.kv_tying or config.kv_transpose_tying:
            qkv_dim = 2 * config.n_embd  # Q + V only
        else:
            qkv_dim = 3 * config.n_embd  # Q + K + V

        self.c_attn = nn.Linear(config.n_embd, qkv_dim, bias=config.bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)

        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.kv_tying = config.kv_tying
        self.kv_transpose_tying = config.kv_transpose_tying

    def forward(self, x):
        B, T, C = x.size()

        if self.kv_tying:
            # Identity tying: K = V
            qkv = self.c_attn(x)  # (B, T, 2*C)
            q, v = qkv.split(self.n_embd, dim=2)

            q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
            v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

            k = v  # Identity tying

        elif self.kv_transpose_tying:
            # Transpose tying: K uses transposed V projection
            qkv = self.c_attn(x)  # (B, T, 2*C)
            q, v = qkv.split(self.n_embd, dim=2)

            # Apply transposed projection for K
            # Extract V projection weights from c_attn
            W_v = self.c_attn.weight[self.n_embd:, :]  # Second half
            k = x @ W_v.T  # Use transposed V weights for K!

            # Reshape
            q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
            k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
            v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        else:
            # Standard: Q, K, V all separate
            qkv = self.c_attn(x)
            q, k, v = qkv.split(self.n_embd, dim=2)

            q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
            k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
            v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        # Attention computation is identical for all variants
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)

        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.c_proj(y)
        return y
```

### Comparison: Identity vs Transpose Tying

| Aspect | K = V (Identity) | K = V.T (Transpose) | Standard (K ≠ V) |
|--------|------------------|---------------------|------------------|
| **Parameters** | 2 × n_embd² | 2 × n_embd² | 3 × n_embd² |
| **Reduction** | 33% | 33% | Baseline |
| **K and V tensors** | Identical | Different | Different |
| **Weight relationship** | W_k = W_v | W_k = W_v.T | W_k ≠ W_v |
| **Attention scores** | Q @ V.T | Q @ (x @ W_v.T).T | Q @ K.T |
| **Theoretical motivation** | Simplicity | Symmetric constraint | Full flexibility |
| **Gradient flow** | Same grads for K and V | Transposed grads | Independent grads |

**Key differences**:

```python
# Identity tying:
K = V  # Same tensor, same gradients
∇W_v = ∇K + ∇V  # Gradients accumulate

# Transpose tying:
K = x @ W_v.T  # Different tensor, related gradients
V = x @ W_v    # Different tensor
∇W_v = ∇V + (∇K).T  # Gradients accumulate with transpose
```

**During training**:

- **Identity tying**: Forces K and V to evolve identically
- **Transpose tying**: Couples K and V evolution through transpose relationship
- **Standard**: K and V evolve independently

### Expected Behavior

**Hypothesis 1**: Transpose tying may preserve more expressiveness than identity tying
- **Rationale**: K ≠ V allows different attention patterns vs aggregation patterns
- **Test**: Compare validation loss V1 (identity) vs V2 (transpose) vs V0 (baseline)

**Hypothesis 2**: Transpose constraint acts as regularization
- **Rationale**: Structural constraint may prevent overfitting
- **Test**: Compare small-dataset performance across variants

**Hypothesis 3**: Gradients through transpose affect learning dynamics
- **Rationale**: Transposed gradient accumulation changes optimization landscape
- **Test**: Monitor training stability and convergence speed

### Ablation Testing

We will test both KV tying variants in upcoming ablations:

**Ablation steps** (to be added):
- **V0**: Baseline GPT-2 (K ≠ V, 3 projections)
- **V1**: Identity tying (K = V, 2 projections)
- **V2**: Transpose tying (K = V.T, 2 projections)

**Metrics to compare**:
- Validation loss and perplexity
- Training convergence speed
- Gradient norms during training
- Attention pattern diversity (via mechint analysis)
- Parameter efficiency (loss per parameter)

**Kconfig support**:
```kconfig
config GPT2_KV_TYING
	bool "Use key-value identity tying (K = V)"
	default n

config GPT2_KV_TRANSPOSE_TYING
	bool "Use key-value transpose tying (K = V.T)"
	default n
	help
	  Use transposed value projection for keys: K = x @ W_v.T
	  where W_v is the value projection weight matrix.
	  This preserves distinctness between K and V while still
	  reducing parameters by 33%.
```

**Status**: Implementation complete. Ablation study planned for next training run.

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

**Ablation steps** (planned expansion):
- **V0**: Baseline GPT-2 (standard attention, K ≠ V)
- **V1**: GPT-2 with KV identity tying (K = V)
- **V2**: GPT-2 with KV transpose tying (K = V.T) - NEW!

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

| Metric | V0 (Baseline) | V1 (Identity) | V2 (Transpose) |
|--------|--------------|---------------|----------------|
| **Val Loss** | TBD | TBD | TBD |
| **Val Perplexity** | TBD | TBD | TBD |
| **Parameters** | 124M | 117M | 117M |
| **Training Time** | TBD | TBD | TBD |
| **GPU Memory** | TBD | TBD | TBD |

**Status**: Ablation study planned. V2 (transpose tying) recently added to test suite.

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

**Hypothesis 1**: Identity tying reduces parameters with minimal quality loss
- **Expected**: V1 perplexity ≤ 5% worse than V0
- **Rationale**: Attention mechanism may be overparameterized

**Hypothesis 2**: Transpose tying preserves more expressiveness
- **Expected**: V2 perplexity between V0 and V1 (better than identity, worse than baseline)
- **Rationale**: K ≠ V preserves distinct attention/aggregation patterns

**Hypothesis 3**: Training speed increases for both variants
- **Expected**: V1 and V2 train 10-15% faster than V0
- **Rationale**: Smaller QKV projection means fewer FLOPs per layer

**Hypothesis 4**: GPU memory usage decreases equally
- **Expected**: V1 and V2 use 5-10% less GPU memory than V0
- **Rationale**: Both variants have same parameter count (2 projections vs 3)

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

**Note**: This section applies to both identity (K = V) and transpose (K = V.T) tying variants. Choose between them based on ablation results for your specific use case.

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

**KV tying** is a parameter reduction technique with two variants:

### Identity Tying (K = V)
✅ **Benefits**:
- 33% reduction in QKV projection parameters
- ~5-6% overall model size reduction
- Faster training due to fewer FLOPs
- Simplest implementation

❓ **Trade-offs**:
- K and V are identical tensors
- Can't learn separate key/value representations
- Quality impact unknown: ablation testing in progress

### Transpose Tying (K = V.T)
✅ **Benefits**:
- Same 33% parameter reduction as identity tying
- K and V are different tensors (preserves distinctness)
- Structural constraint may act as regularization
- Coupled gradients through transpose relationship

❓ **Trade-offs**:
- Still can't learn fully independent K/V
- Unproven approach (no existing literature)
- Transpose semantics unclear
- Quality impact unknown: ablation testing planned

### Common Properties
- **Parameter savings**: Both variants reduce QKV from 3 to 2 projections
- **Inference cache**: NOT reduced (KV cache still stores separate K and V)
- **Training speed**: Both expected to train 10-15% faster
- **GPU memory**: Both use 5-10% less memory than baseline

**Recommendation**: Use KV tying when parameter budget is constrained. Test both identity and transpose variants in ablations to determine which preserves more quality for your use case. Always validate before production deployment.

**For broader context on weight tying**, see **[weight-tying.md](weight-tying.md)**.
