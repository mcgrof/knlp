# GPT2_RA Pseudo Code: SDDMM and Transpose Analysis

## Overview

GPT2_RA uses **learned alternation** between standard and reciprocal attention. Each layer learns a probability `p_recip` to decide which attention pattern to use.

## Key Operations

### SDDMM (Sampled Dense-Dense Matrix Multiplication)
An SDDMM computes `S = sparsity_pattern(A @ B)` where only certain outputs are computed. In attention, the causal mask creates sparsity, making this operation critical for efficiency.

### Transpose Operations
- Standard attention: `Q @ K.T` requires **K transpose**
- Reciprocal attention: `K @ Q.T` requires **Q transpose**

Different transpose operations have different memory access patterns and cache behavior.

---

## Full Forward Pass Pseudo Code

```python
# ============================================================================
# GPT2_RA Attention Layer
# ============================================================================

function GPT2_RA_Attention_Forward(x, layer_idx, alternation_logits):
    """
    Args:
        x: [B, T, C] input hidden states
        layer_idx: current layer index
        alternation_logits: [n_layers] learned logits for alternation

    Returns:
        y: [B, T, C] attention output
    """

    B, T, C = x.shape
    n_heads = 12
    head_dim = C / n_heads  # 768 / 12 = 64

    # ========================================================================
    # STEP 1: QKV Projection (dense matrix multiply)
    # ========================================================================
    # Single fused projection: x @ W_qkv -> [B, T, 3*C]
    # Cost: O(B*T*C * 3*C) = O(3 * B*T*C^2)

    qkv = Linear(x, weight=W_qkv, bias=b_qkv)  # [B, T, 3*C]

    # Split into Q, K, V
    q, k, v = split(qkv, dim=2, sizes=[C, C, C])
    # q: [B, T, C]
    # k: [B, T, C]
    # v: [B, T, C]

    # ========================================================================
    # STEP 2: Reshape to Multi-Head Format
    # ========================================================================
    # No computation, just memory layout change

    q = reshape(q, [B, T, n_heads, head_dim])  # [B, T, 12, 64]
    q = transpose(q, dims=[0, 2, 1, 3])        # [B, 12, T, 64]

    k = reshape(k, [B, T, n_heads, head_dim])  # [B, T, 12, 64]
    k = transpose(k, dims=[0, 2, 1, 3])        # [B, 12, T, 64]

    v = reshape(v, [B, T, n_heads, head_dim])  # [B, T, 12, 64]
    v = transpose(v, dims=[0, 2, 1, 3])        # [B, 12, T, 64]

    # ========================================================================
    # STEP 3: Learned Alternation Decision
    # ========================================================================
    # Per-layer probability of using reciprocal attention

    p_recip = sigmoid(alternation_logits[layer_idx])  # scalar in [0, 1]

    if training:
        # Straight-through estimator for backprop
        use_reciprocal = (p_recip > 0.5) ? 1.0 : 0.0
        # Gradient flows through p_recip during backward
        use_reciprocal = use_reciprocal - detach(p_recip) + p_recip
    else:
        # Hard decision at inference
        use_reciprocal = (p_recip > 0.5) ? 1.0 : 0.0

    # ========================================================================
    # STEP 4: Attention Computation (BRANCHING PATH)
    # ========================================================================

    if use_reciprocal > 0.5:
        # ====================================================================
        # PATH A: RECIPROCAL ATTENTION (K @ Q.T)
        # ====================================================================
        y = ScaledDotProductAttention_Reciprocal(k, q, v)
        # Detailed breakdown below

    else:
        # ====================================================================
        # PATH B: STANDARD ATTENTION (Q @ K.T)
        # ====================================================================
        y = ScaledDotProductAttention_Standard(q, k, v)
        # Detailed breakdown below

    # ========================================================================
    # STEP 5: Merge Heads and Output Projection
    # ========================================================================
    # y: [B, n_heads, T, head_dim] -> [B, T, C]

    y = transpose(y, dims=[0, 2, 1, 3])       # [B, T, n_heads, head_dim]
    y = reshape(y, [B, T, C])                 # [B, T, C]

    # Output projection
    y = Linear(y, weight=W_proj, bias=b_proj)  # [B, T, C]
    y = Dropout(y, p=dropout_rate)

    return y


# ============================================================================
# PATH A: RECIPROCAL ATTENTION (K @ Q.T)
# ============================================================================

function ScaledDotProductAttention_Reciprocal(k, q, v):
    """
    Reciprocal attention: K attends to Q (reversed role).

    Args:
        k: [B, H, T, D] key vectors (used as "queries" here)
        q: [B, H, T, D] query vectors (used as "keys" here)
        v: [B, H, T, D] value vectors

    Returns:
        out: [B, H, T, D] attention output
    """

    B, H, T, D = k.shape
    scale = 1.0 / sqrt(D)  # 1/sqrt(64) = 0.125

    # ========================================================================
    # OPERATION 1: Attention Scores (SDDMM on K and Q^T)
    # ========================================================================
    # Compute: scores = K @ Q^T
    # K:   [B, H, T, D]
    # Q^T: [B, H, D, T]  <- TRANSPOSE Q on last two dims
    # Output: [B, H, T, T]
    #
    # This is SDDMM because causal mask makes it sparse:
    # - Lower triangular matrix only (future tokens masked)
    # - Only T*(T+1)/2 elements actually computed
    # - FlashAttention fuses this with masking for efficiency

    scores = batched_matmul(k, transpose(q, dims=[-2, -1]))
    # scores: [B, H, T, T]
    # scores[b,h,i,j] = sum_d k[b,h,i,d] * q[b,h,j,d]
    #                 = k_i · q_j  (dot product between k_i and q_j)

    # Interpretation:
    # - Token i (using k_i) asks "how relevant is token j (via q_j)?"
    # - REVERSED from standard: normally q_i asks about k_j

    # Scale scores
    scores = scores * scale  # [B, H, T, T]

    # ========================================================================
    # OPERATION 2: Causal Masking
    # ========================================================================
    # Mask future positions (upper triangle)

    causal_mask = create_lower_triangular_mask(T)  # [T, T]
    scores = mask_fill(scores, mask=~causal_mask, value=-inf)

    # After masking:
    # scores[i,j] = k_i · q_j     if j <= i (past and current)
    #             = -inf          if j > i  (future, will be 0 after softmax)

    # ========================================================================
    # OPERATION 3: Softmax (row-wise normalization)
    # ========================================================================
    # Convert scores to probabilities

    attn_weights = softmax(scores, dim=-1)  # [B, H, T, T]
    # Each row sums to 1.0
    # attn_weights[i,j] = how much token i attends to token j

    attn_weights = Dropout(attn_weights, p=dropout_rate)

    # ========================================================================
    # OPERATION 4: Weighted Sum of Values (SDDMM on weights and V)
    # ========================================================================
    # Compute: out = attn_weights @ V
    # attn_weights: [B, H, T, T]
    # V:            [B, H, T, D]
    # Output:       [B, H, T, D]

    out = batched_matmul(attn_weights, v)  # [B, H, T, D]
    # out[b,h,i,:] = sum_j attn_weights[b,h,i,j] * v[b,h,j,:]

    return out
    # Each output token is a weighted average of value vectors


# ============================================================================
# PATH B: STANDARD ATTENTION (Q @ K.T)
# ============================================================================

function ScaledDotProductAttention_Standard(q, k, v):
    """
    Standard attention: Q attends to K (normal GPT-2).

    Args:
        q: [B, H, T, D] query vectors
        k: [B, H, T, D] key vectors
        v: [B, H, T, D] value vectors

    Returns:
        out: [B, H, T, D] attention output
    """

    B, H, T, D = q.shape
    scale = 1.0 / sqrt(D)

    # ========================================================================
    # OPERATION 1: Attention Scores (SDDMM on Q and K^T)
    # ========================================================================
    # Compute: scores = Q @ K^T
    # Q:   [B, H, T, D]
    # K^T: [B, H, D, T]  <- TRANSPOSE K on last two dims
    # Output: [B, H, T, T]

    scores = batched_matmul(q, transpose(k, dims=[-2, -1]))
    # scores: [B, H, T, T]
    # scores[b,h,i,j] = sum_d q[b,h,i,d] * k[b,h,j,d]
    #                 = q_i · k_j  (dot product between q_i and k_j)

    # Interpretation:
    # - Token i (using q_i) asks "how relevant is token j (via k_j)?"
    # - STANDARD: query asks about keys

    scores = scores * scale  # [B, H, T, T]

    # ========================================================================
    # OPERATION 2: Causal Masking
    # ========================================================================
    causal_mask = create_lower_triangular_mask(T)
    scores = mask_fill(scores, mask=~causal_mask, value=-inf)

    # ========================================================================
    # OPERATION 3: Softmax
    # ========================================================================
    attn_weights = softmax(scores, dim=-1)  # [B, H, T, T]
    attn_weights = Dropout(attn_weights, p=dropout_rate)

    # ========================================================================
    # OPERATION 4: Weighted Sum of Values
    # ========================================================================
    out = batched_matmul(attn_weights, v)  # [B, H, T, D]

    return out


# ============================================================================
# Complete Model Forward Pass
# ============================================================================

function GPT2_RA_Forward(input_ids, targets=None):
    """
    Full GPT-2 with RA forward pass.

    Args:
        input_ids: [B, T] token indices
        targets: [B, T] target tokens (optional, for loss)

    Returns:
        logits: [B, T, vocab_size] next-token predictions
        loss: scalar (if targets provided)
    """

    B, T = input_ids.shape

    # ========================================================================
    # Embeddings
    # ========================================================================
    tok_emb = TokenEmbedding(input_ids)         # [B, T, C]
    pos_emb = PositionEmbedding(range(T))       # [T, C]
    x = tok_emb + pos_emb                       # [B, T, C]
    x = Dropout(x, p=dropout_rate)

    # ========================================================================
    # Transformer Blocks
    # ========================================================================
    for layer_idx in range(n_layers):  # 12 layers
        # Attention (with learned RA alternation)
        attn_out = GPT2_RA_Attention_Forward(
            LayerNorm(x),
            layer_idx,
            alternation_logits
        )
        x = x + attn_out  # Residual connection

        # MLP (standard feed-forward)
        mlp_out = MLP_Forward(LayerNorm(x))
        x = x + mlp_out  # Residual connection

    # ========================================================================
    # Output
    # ========================================================================
    x = LayerNorm(x)                            # [B, T, C]
    logits = Linear(x, weight=W_lm_head)        # [B, T, vocab_size]

    # Loss (if targets provided)
    if targets is not None:
        loss = CrossEntropyLoss(
            logits.view(-1, vocab_size),
            targets.view(-1)
        )
        return logits, loss

    return logits, None


function MLP_Forward(x):
    """
    Standard GPT-2 MLP: expand 4x, GELU, project back.

    Args:
        x: [B, T, C]

    Returns:
        out: [B, T, C]
    """
    h = Linear(x, weight=W_fc, bias=b_fc)       # [B, T, 4*C]
    h = GELU(h)                                 # [B, T, 4*C]
    out = Linear(h, weight=W_proj, bias=b_proj) # [B, T, C]
    out = Dropout(out, p=dropout_rate)
    return out
```

---

## SDDMM Operations Summary

### Standard Attention (Q @ K.T)

```
Operation: scores = Q @ K.T
Shapes:    [B,H,T,D] @ [B,H,D,T] -> [B,H,T,T]
Transpose: K transposed on dims [-2,-1]
Cost:      O(B * H * T^2 * D) FLOPs
           BUT: Causal mask makes it O(B * H * T^2 * D / 2) effective
           FlashAttention fuses masking to avoid materializing full T×T

Compute:   scores[i,j] = q_i · k_j
Meaning:   "Token i queries token j"
```

### Reciprocal Attention (K @ Q.T)

```
Operation: scores = K @ Q.T
Shapes:    [B,H,T,D] @ [B,H,D,T] -> [B,H,T,T]
Transpose: Q transposed on dims [-2,-1]
Cost:      O(B * H * T^2 * D) FLOPs (same as standard)
           Same causal mask optimization applies

Compute:   scores[i,j] = k_i · q_j
Meaning:   "Token i (via K) queries token j (via Q)"
           REVERSED ROLES: K acts as query, Q acts as key
```

---

## Key Differences

### Memory Access Patterns

**Standard Attention (Q @ K.T):**
- Q: sequential read per row
- K: needs transpose → potential cache misses
- K.T memory layout may not be contiguous

**Reciprocal Attention (K @ Q.T):**
- K: sequential read per row
- Q: needs transpose → potential cache misses
- Q.T memory layout may not be contiguous

### Computational Cost

Both paths have **identical FLOP count**:
- SDDMM: `2 * B * H * T^2 * D` (with causal mask: ~T^2/2)
- Softmax: `B * H * T^2` (with causal mask: ~T^2/2)
- Value aggregation: `2 * B * H * T^2 * D` (with causal mask: ~T^2/2)

**Total per attention layer: ~4 * B * H * T^2 * D FLOPs**

### Transpose Operations

**Standard:**
1. Reshape Q,K,V: `[B,T,C] -> [B,T,H,D] -> [B,H,T,D]` (memory layout change)
2. Transpose K for scores: `K.T` on dims [-2,-1]
3. Transpose output heads: `[B,H,T,D] -> [B,T,H,D]` (memory layout change)

**Reciprocal:**
1. Reshape Q,K,V: Same as standard
2. **Transpose Q** for scores: `Q.T` on dims [-2,-1] ← DIFFERENT
3. Transpose output heads: Same as standard

---

## Performance Implications

### Theoretical
- **Same FLOP count** for both paths
- **Same memory footprint** (T×T scores materialized in both)
- **Same causal masking** optimization applies

### Practical (GPU)
- Transpose overhead negligible (handled by cuBLAS efficiently)
- FlashAttention fuses operations → transposes mostly disappear
- Runtime difference: **~12% slower** (measured)
  - Due to branching overhead, not the transpose itself
  - GPU prefers uniform computation across layers
  - Learned alternation creates divergence

### Measured Performance
```
GPT2:     248.0 tok/s (baseline)
GPT2_RA:  217.3 tok/s (12.4% slower)
```

Slowdown comes from:
1. **Per-layer branching** (if/else on `use_reciprocal`)
2. **Sigmoid evaluation** for alternation probability
3. **Potential kernel launch overhead** from divergent paths

**NOT from transpose operations** (cuBLAS handles these efficiently).

---

## Optimization Opportunities

### Current Implementation
```python
if use_reciprocal > 0.5:
    y = SDPA(k, q, v)  # K @ Q.T
else:
    y = SDPA(q, k, v)  # Q @ K.T
```

### Potential Optimizations
1. **Remove branching**: Pre-commit to standard or reciprocal per layer
2. **Fused kernels**: Custom CUDA kernel that conditionally swaps Q/K
3. **Block-level alternation**: Alternate entire blocks, not layers (reduces branches)

The SDDMM operations themselves are optimal (FlashAttention-2 is state-of-the-art).
