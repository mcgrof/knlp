# Benchmark Plan: Does RA Beat Standard Attention?

## Goal
Determine if Reciprocal Attention provides sufficient quality improvement to justify its computational overhead.

## Baseline Comparisons

### Test 1: Standard Attention (SDPA, fused)
- **Implementation**: F.scaled_dot_product_attention()
- **Expected speed**: ~1.98 ms/iter (measured)
- **Config**: GPT-2 124M, batch=8, T=1024, standard 4:1 MLP ratio
- **Measure**: val_loss curve over wall-clock time

### Test 2: Standard Attention (open-coded, unfused)
- **Implementation**: Q @ K^T → softmax → @ V (manual)
- **Expected speed**: ~9.13 ms/iter (measured, 4.6x slower)
- **Config**: Same as Test 1
- **Purpose**: Fair comparison baseline for open-coded RA

### Test 3: RA Attention (open-coded)
- **Implementation**: w_std * S + w_rec * S^T + w_disc * d
- **Expected speed**: ~16.76 ms/iter (measured, 8.5x slower than SDPA)
- **Config**: Same model size, with RA gates
- **Measure**: val_loss curve, check if quality improvement justifies 8.5x slowdown

### Test 4: RA Attention (Triton fused)
- **Implementation**: Our new Triton kernel
- **Expected speed**: ~4.85 ms/iter (measured, 2.45x slower than SDPA)
- **Config**: Same as Test 3
- **Measure**: val_loss curve, check if quality improvement justifies 2.45x slowdown

## Success Criteria

RA is "worth it" if:
1. **RA (Triton) reaches val_loss < SDPA baseline after same wall-clock time**
   - Example: After 1 hour, SDPA = 3.57, RA = 3.50
2. **RA (Triton) reaches SDPA's val_loss in < 1/2.45 of the time**
   - Example: SDPA reaches 3.57 in 1 hour, RA reaches 3.57 in < 25 minutes
3. **RA (Triton) reaches significantly better final loss**
   - Example: SDPA converges to 3.57, RA converges to 3.45

## Measurement Protocol

### Short Test (1 hour each)
- Run each config for 1 hour wall-clock time
- Log: iteration count, val_loss every 100 iters
- Plot: val_loss vs wall-clock time (not iterations!)
- Compare: which reaches lowest val_loss in 1 hour?

### Full Test (to convergence)
- Run each config until validation loss plateaus
- Measure: total wall-clock time, final val_loss
- Compare: quality vs efficiency tradeoff

## Optimization Directions (if RA shows promise)

If RA proves valuable but still too slow, try:

### 1. Sparse Reciprocity
**Idea**: Only compute S^T for high-attention positions
```python
# First pass: standard attention
S = Q @ K^T / sqrt(D)
attn_weights = softmax(S)

# Identify sparse mask (e.g., top-10% per row)
mask = attn_weights > percentile(attn_weights, 90, dim=-1)

# Second pass: reciprocal only for masked positions
S_rec = sparse_S^T[mask]  # Much cheaper than full S^T
```
**Expected speedup**: 5-10x for reciprocity component

### 2. Local Reciprocity Window
**Idea**: Only compute S^T within local window (e.g., ±128 tokens)
```python
# Standard attention: full [T, T]
S = Q @ K^T

# Reciprocal: only local [T, 256] band
S_rec_local = Q[i] @ K[i-128:i+128]^T for each i
```
**Expected speedup**: T/256 (4x for T=1024)

### 3. Learned Reciprocity Gating
**Idea**: Learn which positions need reciprocity
```python
# Lightweight predictor
need_reciprocity = sigmoid(mlp(query_stats))  # [T] binary mask

# Selective reciprocity
S_rec = need_reciprocity[:, None] * S^T
```
**Expected speedup**: Depends on learned sparsity (2-5x)

### 4. Progressive Reciprocity Annealing
**Idea**: Use full RA early, reduce over training
```python
# Training schedule
reciprocity_weight = max(0.1, 1.0 - step / total_steps)

# Dynamically adjust
logits = w_std * S + reciprocity_weight * w_rec * S^T
```
**Benefit**: Fast convergence early, fast iterations late

### 5. Layer-Selective Reciprocity
**Idea**: Only use RA in layers where it helps
```python
# Hypothesis: middle layers benefit most from bidirectional info
use_ra = layer_idx in [4, 5, 6, 7, 8]  # Middle 5 of 12 layers

if use_ra:
    logits = w_std * S + w_rec * S^T + w_disc * d
else:
    logits = S  # Fast SDPA path
```
**Expected speedup**: 2.4x (use SDPA in 7/12 layers)

## Questions to Answer

1. **Does RA improve convergence speed?** (validation loss drops faster per iteration)
2. **Does RA improve sample efficiency?** (reaches target loss in fewer iterations)
3. **Does RA improve final quality?** (better final validation loss)
4. **Where does RA help most?** (which layers, which training phase)
5. **What's the minimum effective reciprocity?** (sparse? local? gated?)

## Clever Optimization Ideas

### Idea A: Reciprocity Budget
**Concept**: Fixed compute budget for reciprocity, learn how to spend it
```python
# Budget: 10% of tokens can receive reciprocal attention
budget = int(0.1 * T)

# Learn which tokens to select
importance = learned_scorer(Q, K)
top_k_indices = torch.topk(importance, budget).indices

# Compute reciprocity only for selected tokens
S_rec_sparse[top_k_indices, :] = S^T[top_k_indices, :]
```

### Idea B: Amortized Reciprocity
**Concept**: Compute S^T every N steps, cache and reuse
```python
# Cache reciprocal attention patterns
if step % N == 0:
    cached_S_T = compute_reciprocity(Q, K)

# Reuse cached pattern for N steps
logits = w_std * S + w_rec * cached_S_T
```
**Risk**: Stale patterns may hurt quality

### Idea C: Factorized Reciprocity
**Concept**: Low-rank approximation of S^T
```python
# Standard: S^T = (Q @ K^T)^T = K @ Q^T
# Full cost: [T, D] @ [D, T] = O(T^2 * D)

# Low-rank: S^T ≈ (Q @ W1) @ (K @ W2)^T
# W1, W2 are [D, r] with r << D
# Cost: O(T * D * r + T^2 * r) << O(T^2 * D) for small r
```

### Idea D: Adaptive Reciprocity
**Concept**: Model learns when to use reciprocity
```python
# Per-head reciprocity gates (already have)
# But also: per-token reciprocity gates
token_needs_rec = sigmoid(gate_net(hidden_states))  # [T]

# Selective application
S_rec_masked = token_needs_rec[:, None] * S^T
logits = w_std * S + w_rec * S_rec_masked
```

## Next Steps

1. **Run 4 baseline tests** (1 hour each, ~4 hours total)
2. **Analyze val_loss curves** - does RA show promise?
3. **If yes**: Implement most promising optimization
4. **If no**: Consider if RA architectural concept needs rethinking
