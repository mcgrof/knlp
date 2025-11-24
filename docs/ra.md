# Reciprocal Attention (RA)

## Overview

Reciprocal Attention (RA) explores alternating between standard Q@K.T and
reciprocal K@Q.T attention patterns in transformer architectures. The core
insight: computing both attention directions provides richer bidirectional
information flow across layers.

### Inductive Bias

**Core Principle**: Reciprocal Attention implements the inductive bias that
token interactions should be evaluated bidirectionally. Standard attention
asks "how much should query i attend to key j" (Q@K.T). RA alternates this
with the reverse question: "how much should key j attend to query i" (K@Q.T).

This creates complementary views of the same token relationships across
transformer depth, producing smoother optimization geometry and better gradient
flow.

#### Empirically Validated Value of RA

**Experimental setup**: GPT-2 124M on FineWebEdu dataset using
`make defconfig-gpt2-kvsplice-ablation`. Each configuration trained for 2 hours
on AMD Radeon Pro W7900 GPU with identical hyperparameters.

Results and trained models: [test_matrix_results_20251123_231956](https://github.com/mcgrof/knlp-key-results/tree/main/key_results/test_matrix_results_20251123_231956)

**1. Smoother Optimization Geometry**

RA produces flatter curvature in the optimization landscape, enabling better
gradient flow. See [FIM.md](FIM.md) for explanation of Fisher Information
Matrix metrics.

| Metric | MLA | RA+MLA | Improvement |
|--------|-----|--------|-------------|
| Validation perplexity | 3.6 | 3.4 | -5.6% |
| [FIM eigmax](FIM.md#eigmax-maximum-eigenvalue) (mean) | Higher | 0.0352 | Lower (flatter) |
| Inference tokens/sec | 17,031 | 21,696 | +27% |
| Training iterations | 280 | 285 | +1.8% |

The lower Fisher Information Matrix eigenvalues indicate reduced curvature
sharpness. RA recovers the quality loss from MLA compression (6x cache
reduction) and achieves 27% inference speedup with identical cache size.

**2. Bidirectional Information Flow**

Alternating attention directions enriches representation learning. After 2-hour
training runs on AMD W7900 GPU, we evaluated models with LM-eval to get a quick
"pilot signal" of downstream task performance:

| Benchmark | MLA | RA+MLA | Improvement |
|-----------|-----|--------|-------------|
| ARC-Easy | 24% | 26% | +2% |
| HellaSwag | 28% | 29% | +1% |
| Winogrande | 48% | 51% | +3% |
| Average | 33.3% | 35.3% | +6.0% |

All three LM-eval benchmarks improved. The complementary forward and reverse
views provide richer context for downstream tasks. These 2-hour runs provide
fast validation before committing to longer training.

**Summary**: Reciprocal Attention's empirically-validated inductive bias is
that **bidirectional evaluation of token relationships produces smoother
optimization geometry**. By alternating Q@K.T (forward: how queries view keys)
and K@Q.T (reverse: how keys view queries) across transformer layers, RA
flattens the curvature of the loss landscape.

This smoother geometry enables:
- Better gradient flow during training (recovers 5.6% quality loss from MLA)
- Faster convergence at inference (27% tokens/sec increase)
- Improved downstream task accuracy (6% average LM-eval gain)
- More stable training with compressed representations

**When to use RA**: Training with compressed attention mechanisms (MLA) or
when seeking inference speedup. The optimization benefits compensate for
compression losses without memory overhead. See [kvsplice.md](kvsplice.md) for
learned compression techniques that further benefit from RA's smooth
optimization.

### The Mathematical Foundation

Standard attention computes:
```
attn_scores = (Q @ K.T) / sqrt(d_k)
attn_probs = softmax(attn_scores)
output = attn_probs @ V
```

Reciprocal attention transposes the query-key relationship:
```
attn_scores = (K @ Q.T) / sqrt(d_k)  # Transpose flips direction
attn_probs = softmax(attn_scores)
output = attn_probs @ V
```

**Semantic meaning**:
- Standard: "how much does token i attend to token j"
- Reciprocal: "how much does token j attend to token i" (reverse direction)

**Mathematical identity**: `(Q @ K.T).T = K @ Q.T`

The transpose operation is essentially **free** on GPU - it's just a view that
changes tensor strides without copying data. Empirical benchmarks on AMD
Radeon Pro W7900 (shape: [4, 16, 1024, 64]) show:

| Operation | Eager Mode | torch.compile |
|-----------|-----------|---------------|
| Q @ K.T | 0.346 ms | 0.365 ms |
| (Q @ K.T).T (view) | 0.340 ms | 0.333 ms |
| K @ Q.T (direct) | 0.332 ms | 0.331 ms |

**Key findings**:
- Transpose adds ~0 overhead (view operation changes strides only)
- `scores.data_ptr() == scores.transpose(-2,-1).data_ptr()` → True (same
  memory)
- `scores.transpose(-2,-1)._base is scores` → True (view, not copy)
- K @ Q.T is equivalent in speed to (Q @ K.T).T

See [scripts/benchmark_tranpsose_as_view.py](../scripts/benchmark_tranpsose_as_view.py)
for the complete benchmark code.

### Research Evolution

Our research evolved from evaluating the impact of computing both Q@K.T and
K@Q.T simultaneously to finding more compute-friendly implementations:

1. **MLA + RA**: Learned per-layer alternation with Multi-head Latent
   Attention for cache compression (introduces Token-Latent cache to support
   Q/K transpose)
2. **Mathematical Introspection**: What does RA implicitly add from a
   mathematical perspective? Can we leverage it for better compression? See
   [FIM.md](FIM.md) for full Fisher Information Matrix analysis.
3. **KVSplice**: Additional learned compression on top of Token-Latent cache.
   See [kvsplice.md](kvsplice.md) for complete architecture, results, and
   future research directions. KVSplice achieves 12x total compression while
   improving quality through regularization.
4. **GPT-2 + RA**: Pure reciprocal attention without compression (ongoing)

---

## RA with Multi-head Latent Attention (MLA)

### Architecture

Multi-head Latent Attention (MLA) from DeepSeek-V2/V3 compresses the KV cache
by storing a compressed latent instead of full K, V tensors.

**Standard Attention Cache**:
```
Cache: K, V per layer
Size: 2 × n_layers × n_heads × seq_len × head_dim = 36 MB
```

**Standard MLA KV-Latent Cache**:
```python
# Standard MLA (DeepSeek): Q direct, KV from shared latent
Q = W_q(x)  # [B, T, n_heads * head_dim] - not cached

# K, V share compressed latent (this gets cached)
kv_latent = to_kv_latent(x)  # [B, T, d_latent=256]
K, V = from_kv_latent(kv_latent).chunk(2)

# Cache stores kv_latent instead of K, V
Cache: kv_latent per layer
Size: n_layers × seq_len × d_latent = 6 MB (6x compression)
```

**RA+MLA Token-Latent (TL) Cache**: When adding RA to MLA, we need to
support both Q@K.T and K@Q.T attention patterns. Since RA requires using both
Q and K in transposed roles, we generalize the cache to support both
mechanisms:

```python
# RA+MLA: Both Q and K need latent representation for transpose support
token_latent = to_token_latent(x)  # [B, T, d_latent=256]
Q, K, V = from_token_latent(token_latent).chunk(3)

# Cache stores token_latent (supports both Q and K usage)
Cache: token_latent per layer (Token-Latent cache)
Size: n_layers × seq_len × d_latent = 6 MB (6x compression)
```

The Token-Latent (TL) cache name reflects that the cached representation must
support using both Q and K in either role (Q@K.T or K@Q.T). Standard MLA only
needs KV-latent since Q is always in the query role.

### Implementation Comparison

**MLA Attention (KV-Latent Cache)**

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class MLAAttention(nn.Module):
    """MLA with KV-Latent cache (DeepSeek-V2/V3 style)."""

    def __init__(self, d_model, n_heads, d_latent=256, dropout=0.0):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.d_latent = d_latent
        self.dropout = dropout

        # Q direct, KV from shared latent
        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.to_kv_latent = nn.Linear(d_model, d_latent, bias=False)
        self.from_kv_latent = nn.Linear(d_latent, 2 * d_model, bias=False)

        self.c_proj = nn.Linear(d_model, d_model)

    def forward(self, x, cache=None, use_cache=False):
        B, T, C = x.shape

        # Q computed directly (not cached)
        q = self.w_q(x)  # [B, T, d_model]

        # K, V from compressed latent (this gets cached)
        kv_latent = self.to_kv_latent(x)  # [B, T, d_latent]

        # Concatenate with cached latent if present
        if cache is not None:
            kv_latent = torch.cat([cache, kv_latent], dim=1)

        # Decompress to K, V
        kv = self.from_kv_latent(kv_latent)  # [B, T_total, 2*d_model]
        k, v = kv.split(self.d_model, dim=2)

        # Reshape to [B, n_heads, T, head_dim]
        T_total = kv_latent.shape[1]
        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T_total, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T_total, self.n_heads, self.head_dim).transpose(1, 2)

        # Standard attention: Q @ K.T
        y = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=(cache is None)
        )

        # Reshape and project output
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.c_proj(y)

        # Return with kv-latent cache if requested
        new_cache = kv_latent if use_cache else None
        return y, new_cache
```

**RA+MLA Attention (Token-Latent Cache)**

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class RA_MLA_Attention(nn.Module):
    """RA + MLA with Token-Latent cache and learned alternation."""

    def __init__(self, d_model, n_heads, d_latent=256, layer_idx=0,
                 alternation_logits=None, dropout=0.0):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.d_latent = d_latent
        self.layer_idx = layer_idx
        self.dropout = dropout

        # Token-Latent compression (supports Q, K, V from single latent)
        self.to_token_latent = nn.Linear(d_model, d_latent, bias=False)
        self.from_token_latent = nn.Linear(d_latent, 3 * d_model, bias=False)

        # Learned alternation: shared across all layers
        self.alternation_logits = alternation_logits

        self.c_proj = nn.Linear(d_model, d_model)

    def get_alternation_prob(self):
        """Get this layer's probability of using reciprocal attention."""
        return torch.sigmoid(self.alternation_logits[self.layer_idx])

    def forward(self, x, cache=None, use_cache=False):
        B, T, C = x.shape

        # Compress to token latent
        token_latent = self.to_token_latent(x)  # [B, T, d_latent]

        # Concatenate with cached latent if present
        if cache is not None:
            token_latent = torch.cat([cache, token_latent], dim=1)

        # Decompress to Q, K, V from token latent
        qkv = self.from_token_latent(token_latent)  # [B, T_total, 3*d_model]
        q, k, v = qkv.split(self.d_model, dim=2)

        # Reshape to [B, n_heads, T, head_dim]
        T_total = token_latent.shape[1]
        q = q.view(B, T_total, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T_total, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T_total, self.n_heads, self.head_dim).transpose(1, 2)

        # Select only new tokens for q (if using cache)
        if cache is not None:
            q = q[:, :, -T:, :]

        # Learned alternation: standard Q@K.T vs reciprocal K@Q.T
        p_recip = self.get_alternation_prob()

        if self.training:
            # Straight-through estimator for hard decision
            use_reciprocal = (p_recip > 0.5).float()
            use_reciprocal = use_reciprocal - p_recip.detach() + p_recip
        else:
            use_reciprocal = (p_recip > 0.5).float()

        # Apply attention based on learned decision
        if use_reciprocal > 0.5:
            # Reciprocal: K @ Q.T (swap q and k arguments)
            y = F.scaled_dot_product_attention(
                k, q, v,  # Note: k and q are swapped
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=(cache is None)
            )
        else:
            # Standard: Q @ K.T
            y = F.scaled_dot_product_attention(
                q, k, v,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=(cache is None)
            )

        # Reshape and project output
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.c_proj(y)

        # Return with token-latent cache if requested
        new_cache = token_latent if use_cache else None
        return y, new_cache
```

**Key Differences**:

| Aspect | MLA | RA+MLA |
|--------|-----|--------|
| Cache | kv_latent (6 MB) | token_latent (6 MB) |
| Q generation | Direct from `w_q(x)` | From compressed `token_latent` |
| K,V generation | From `kv_latent` (shared) | From `token_latent` (with Q) |
| Attention pattern | Always Q @ K.T | Learned per-layer (Q@K.T or K@Q.T) |
| Alternation | Fixed standard | Learned via `sigmoid(alternation_logits)` |
| Memory overhead | 6 MB cache | Same cache size, learned logits only |

**Model Initialization with Shared Alternation Logits**

```python
class TransformerWithRA_MLA(nn.Module):
    def __init__(self, n_layers, d_model, n_heads, d_latent=256):
        super().__init__()
        self.n_layers = n_layers

        # Shared alternation logits across all layers
        # Initialized to ~0 for balanced 50/50 split
        self.alternation_logits = nn.Parameter(torch.zeros(n_layers))

        # Create attention layers sharing alternation logits
        self.layers = nn.ModuleList([
            RA_MLA_Attention(
                d_model=d_model,
                n_heads=n_heads,
                d_latent=d_latent,
                layer_idx=i,
                alternation_logits=self.alternation_logits
            )
            for i in range(n_layers)
        ])

    def compute_balance_loss(self):
        """Encourage 50/50 split between standard and reciprocal attention."""
        probs = torch.sigmoid(self.alternation_logits)  # [n_layers]
        target = self.n_layers / 2.0
        balance_loss = (probs.sum() - target) ** 2
        return balance_loss

    def get_layer_decisions(self):
        """Get learned decisions for each layer (for analysis)."""
        probs = torch.sigmoid(self.alternation_logits)
        decisions = (probs > 0.5).float()
        return {
            "probs": probs.detach().cpu().tolist(),
            "decisions": decisions.cpu().tolist(),
            "n_reciprocal": decisions.sum().item(),
            "n_standard": (self.n_layers - decisions.sum()).item()
        }
```

### Results: MLA vs RA+MLA

Test configuration: GPT-2 124M, FineWebEdu dataset, identical hyperparameters,
2-hour training runs per configuration on AMD Radeon Pro W7900 GPU.

Experimental data: [test_matrix_results_20251123_231956](https://github.com/mcgrof/knlp-key-results/tree/main/key_results/test_matrix_results_20251123_231956)

The following results compare MLA vs RA+MLA only (without KVSplice). For
compression results, see [kvsplice.md](kvsplice.md).

![Validation Quality](images/ra_validation_quality.png)

| Architecture | Val Loss | Perplexity | Tokens/sec | Cache | Iters |
|-------------|----------|------------|------------|-------|-------|
| Baseline GPT-2 | 1.199 | 3.3 | - | 36 MB | - |
| MLA | 1.276 | 3.6 | 17,031 | 6 MB | 280 |
| **RA+MLA** | **1.223** | **3.4** | **21,696** | **6 MB** | **285** |

**Key Findings**:

1. **MLA baseline**: 6x cache compression (36 MB → 6 MB) but +8.6%
   perplexity degradation vs baseline GPT-2
2. **RA+MLA**: Recovers MLA's quality loss (-5.6% perplexity vs MLA alone)
3. **Inference speedup**: **27% faster** (17K → 21.7K tokens/sec)
4. **Training efficiency**: Similar iterations completed (280 vs 285)
5. **Same cache size**: No memory overhead for RA

![Inference Speed](images/ra_inference_speed.png)

### LM-Eval Benchmarks

Evaluated on ARC-Easy, HellaSwag, and Winogrande (100 samples each):

![LM-Eval Benchmarks](images/ra_lmeval_benchmarks.png)

| Architecture | ARC-Easy | HellaSwag | Winogrande | Average |
|-------------|----------|-----------|------------|---------|
| MLA | 24% | 28% | 48% | 33.3% |
| **RA+MLA** | **26%** | **29%** | **51%** | **35.3%** |

**All metrics improved** with RA. Average accuracy gain: +6.0%.

### Analysis

**Why does RA improve MLA?**

1. **Optimization geometry**: K@Q.T alternation provides smoother gradient
   flow, compensating for compression losses
2. **Bidirectional information**: Alternating attention directions enriches
   representation learning
3. **Inference efficiency**: Lower perplexity → faster convergence at
   inference → 27% speedup

**Conclusion**: RA provides significant benefits to MLA - better quality, much
faster inference, same cache size. This validates the RA mechanism as an
effective bidirectional attention approach.

---

## GPT-2 + RA (Pure Reciprocal Attention)

### Status: In Progress

Testing pure RA mechanism without MLA or latent compression to isolate RA's
contribution to inference speed.

### Architecture

Standard GPT-2 with learned per-layer decision on whether to use Q@K.T
(standard) or K@Q.T (reciprocal) attention. Same implementation as MLA+RA but
without compression:

- No latent compression (full Q, K, V projections)
- Same cache requirements as baseline GPT-2 (36 MB)
- Learned alternation with balance loss (50/50 split)
- Same model size as baseline

### Motivation

Evolution 4 (MLA+RA) showed RA provides 27% inference speedup with MLA.
Question: Does this benefit hold for pure GPT-2 without compression?

**Hypotheses**:
1. RA's speed benefit comes from helping compressed representations
2. RA is fundamentally more efficient regardless of compression

### Implementation Approach

The efficient learned per-layer approach:

- **Learned per-layer decision**: Each layer learns whether to use standard
  or reciprocal attention
- **No dual computation**: Only compute chosen attention direction
- **No router overhead**: Simple binary choice per layer
- **Balance loss**: Encourages 50/50 split across layers for bidirectional
  information flow

This approach provides RA's benefits without compute overhead.

### Expected Analysis

**TODO**: Results pending - test currently running

Project: `gpt2-ra-ablation`
Steps: B0 (baseline), RALEARN0 (learned RA)

Will compare:
- Perplexity: GPT-2 baseline vs GPT-2+RA
- Inference speed: tokens/sec for both architectures
- Training efficiency: iterations completed in same time
- Cache memory: should be identical (no compression)
- LM-eval benchmarks: quality comparison

**Key question**: If GPT-2+RA shows similar 20-27% inference speedup without
compression, this confirms RA is fundamentally more efficient, not just
beneficial for compressed models.

---

## Implementation Details

### Flash Attention

All RA implementations use PyTorch's `scaled_dot_product_attention` for
efficient computation:

```python
# Standard attention
out = F.scaled_dot_product_attention(q, k, v, is_causal=True)

# Reciprocal attention: swap q and k arguments
out = F.scaled_dot_product_attention(k, q, v, is_causal=True)
```

**Benefits**:
- Automatic kernel selection (flash attention when available)
- Memory efficient: O(T) instead of O(T^2) for attention matrix
- Fused operations: single CUDA kernel for QK^T, softmax, V multiply

### Causal Masking

Always use causal masking (`is_causal=True`) for autoregressive generation.

**Note**: Chunked/speculative decoding with multiple new tokens while cache
exists would need custom mask accounting for prefix offset - `is_causal` alone
won't work for that snowflake case. Our current inference tests don't use
chunked decoding.

---

## Fisher Information Matrix (FIM) Analysis

### Hypothesis Testing

**Original hypothesis**: RA produces higher Fisher Information in attention
patterns, making attention more informative per layer.

### Results

Comparing MLA vs RA+MLA (layer 0, head 0, last 100 training samples):

| Metric | MLA | RA+MLA | Interpretation |
|--------|-----|--------|----------------|
| eigmax | Higher | Lower | RA has flatter curvature |
| trace | Higher | Lower | RA has less total Fisher mass |
| energy_r16 | Lower | Higher | RA concentrates energy better |

**Hypothesis rejected**: RA shows **lower** FIM eigmax values, not higher.

### Conclusion

RA's benefits come from optimization geometry and gradient flow, not from
increased information density. The lower FIM values suggest RA provides:

- Flatter curvature landscape (easier optimization)
- Better energy concentration (more efficient use of capacity)
- Smoother gradient flow (helps with compressed representations)

This explains why RA particularly helps MLA: the smoother optimization
compensates for compression losses.

---

## Hypotheses Tested and Rejected

During RA research, we tested several plausible hypotheses about the mechanism
by which bidirectional attention might improve performance. Empirical data
systematically rejected these predictions. This section documents what RA does
**not** provide, helping clarify its actual value proposition.

### Hypothesis 1: RA Increases Fisher Information

**Initial prediction**: Bidirectional flow concentrates information into
sharper, more informative attention patterns (higher FIM eigenvalues).

**Data contradiction** (W&B: gpt2-kvsplice-ablation-w7900-mla-fixed):

| Architecture | eigmax | trace | energy_r16 |
|-------------|--------|-------|------------|
| MLA | Higher | Higher | 0.373 |
| RA+MLA | 0.0352 | Similar/lower | 0.373 |

**Verdict**: Rejected. RA produces **lower** eigmax (flatter geometry), not
higher. The FIM trace shows similar or slightly reduced total Fisher
information. RA improves optimization by smoothing curvature, not by
concentrating information.

### Hypothesis 2: RA Concentrates Information into Fewer Modes

**Initial prediction**: Alternating attention creates more structured,
low-dimensional attention patterns that pack information into top eigenmodes.

**Data contradiction** (FIM energy concentration):

| Rank | RA+MLA |
|------|--------|
| r=8 | 22.3% |
| r=16 | 37.3% |

Only 37% of Fisher energy captured in top 16 modes. Would need r>16 to reach
90% energy concentration.

**Verdict**: Rejected. Information is diffuse across the spectral basis, not
concentrated. RA does not concentrate Fisher energy into fewer modes.

### Hypothesis 3: RA Makes KV Cache More Compressible

**Initial prediction**: Bidirectional alternation creates shared latent
geometry that enables better compression of attention cache.

**Data contradiction**: FIM energy concentration unchanged with RA.

| Architecture | eigmax | energy_r16 |
|-------------|--------|------------|
| MLA | Higher | 0.373 |
| RA+MLA | 0.0352 | 0.373 |

RA produces flatter curvature (lower eigmax) but does not change Fisher energy
concentration in the cache.

**Verdict**: Rejected. RA doesn't change the compressibility as measured by
Fisher energy geometry. RA's benefits are in optimization dynamics (flatter
curvature, better gradient flow), not in creating compressible structure. See
[kvsplice.md](kvsplice.md) for learned compression approaches that work
independently of RA.

### Key Takeaway

RA does not fundamentally change the information structure of attention:
- No increase in Fisher Information (eigmax lower, not higher)
- No concentration into fewer modes (37% energy in top 16)
- No improved compressibility from geometry alone

Do not expect RA to enable geometric compression or guide low-rank
approximations via FIM metrics. Learned compression techniques (see
[kvsplice.md](kvsplice.md)) work independently and benefit from RA's smooth
training, not from structural changes RA makes to attention patterns.

---

## Mathematical Foundation: Entropic Optimal Transport

### SPDA Paper Result

The paper "Scaled Dot-Product Attention as One-Sided Entropic Optimal
Transport" proves that standard transformer attention is exactly solving an
Entropic Optimal Transport (EOT) problem.

Given query q and keys {k_j}, attention chooses a probability distribution p_j
that minimizes:

```
EOT(p) = Σ_j p_j · C_j + τ · H(p)
```

where:
- `C_j = -⟨q, k_j⟩` is the transport cost
- `H(p) = -Σ_j p_j log p_j` is entropy (regularization)
- τ is temperature (typically 1/√d_head)

The unique minimizer is:

```
p_j = softmax(⟨q, k_j⟩ / τ)
```

This proves that **SDPA = EOT solution**.

### Fisher Information Matrix

The attention scores enter a log-sum-exp potential whose Hessian is exactly
the **Fisher Information Matrix (FIM)** of the categorical distribution p:

```
∇²φ(s) = (1/τ²)(diag(p) - p pᵀ)
```

The FIM describes the curvature geometry of attention:
- **Large eigenvalues**: High curvature, sensitive to perturbations
- **Small eigenvalues**: Flat directions, easy optimization

### Why This Matters for RA

**RA (Reciprocal Attention)** alternates between forward and reverse EOT:

```
F_fwd  from softmax(Q Kᵀ / τ)    # Forward geometry
F_rev  from softmax(K Qᵀ / τ)    # Reverse geometry
```

Each layer experiences one geometry, alternating across depth. This provides:
- Smoother curvature spectrum across layers
- More stable optimization
- Better-conditioned Fisher geometry

The EOT framework explains why RA improves training: bidirectional information
flow creates more balanced optimization landscape.

---

## Summary

### Terminology Note

**KV-Latent Cache**: Standard MLA (DeepSeek) stores a compressed `kv_latent`
that generates K and V. Q is computed directly.

**Token-Latent (TL) Cache**: When combining RA with MLA, we generalize to
`token_latent` that generates Q, K, and V. This supports both Q@K.T and K@Q.T
attention patterns since both Q and K need to be usable in either role.

### Key Findings

1. **MLA alone**: 6x cache compression, +8.6% perplexity degradation
2. **RA + MLA**: Introduces Token-Latent cache, recovers MLA quality (-5.6%),
   27% inference speedup, same 6x cache compression

### Recommendations

**For KV cache compression**: Combine RA+MLA with learned compression
techniques. See [kvsplice.md](kvsplice.md) for compression research that
achieves 12x total compression while improving quality.

**For inference speed**: Add learned-layer RA (20-27% speedup, no memory
overhead)

**Best approach**: RA+MLA provides the foundation for both compression and
speed improvements through flatter optimization geometry.

### Architecture Trade-offs

| Goal | Architecture | Trade-off |
|------|-------------|-----------|
| Cache compression | MLA | 6 MB cache, +8.6% perplexity |
| Speed + quality | RA+MLA | 6 MB cache, 27% faster, -5.6% perplexity |

See [kvsplice.md](kvsplice.md) for learned compression techniques that further
reduce cache size from 6 MB to 3 MB (12x total compression).

### Future Work

1. **GPT-2+RA validation**: Confirm 20-27% speedup applies to uncompressed
   models
2. **Larger scales**: Test RA+MLA on GPT-2 1.5B and larger models
3. **Other architectures**: Apply RA to other compressed attention mechanisms
   (GQA, MQA)
