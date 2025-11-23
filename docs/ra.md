# Reciprocal Attention (RA) with Compute Routing

## The Core Insight

Not all tokens need the same computational effort. The distance between a
token's contextual hidden state and its static embedding measures "contextual
hardness":

```
shift = |x - E(x)|
```

- **Small shift**: Token stayed near its embedding. Context didn't reshape its
  meaning. Use cheap compute.
- **Large shift**: Token moved far from embedding. Highly context-dependent.
  Deserves full attention.

This gives us a FLOP-cheap routing signal without T^2 overhead.

## Architecture

### 1. Shared QKV with Head Groups

Single QKV projection, heads partitioned into FULL and RA groups:

```python
# Example: 12 heads total, ra_head_frac=0.25
# FULL heads: 0-8 (9 heads)
# RA heads: 9-11 (3 heads)

qkv = c_attn(x)                    # [B, T, 3D]
out = sdpa(q, k, v)                # Single SDPA for all heads
out_full = proj_full(heads[:9])    # [B, T, D]
out_ra = proj_ra(heads[9:])        # [B, T, D]
```

RA's cheapness: fewer heads = smaller output projection.

### 2. SDPA with Flash Attention

Uses PyTorch's `scaled_dot_product_attention` for efficient attention computation:

```python
out = F.scaled_dot_product_attention(
    q, k, v,
    attn_mask=None,
    dropout_p=dropout if training else 0.0,
    is_causal=True,  # Enables flash attention kernel
)
```

Benefits:
- Automatic kernel selection (flash attention when available)
- Memory efficient: O(T) instead of O(T^2) for attention matrix
- Fused operations: single CUDA kernel for QK^T, softmax, V multiply
- ~60% faster than manual matmul + softmax implementation

### 3. Context Router

2-way routing based on shift and other cheap features:

```python
shift = |x - E(x)|                 # [B, T]
features = [shift, ||x||, ||E||, <x,E>]
probs = softmax(mlp(features))     # [B, T, 2]
# probs: p_ra, p_full
```

Output mixed according to router:

```python
out = p_ra * out_ra + p_full * out_full
```

Compute penalty discourages expensive path:

```python
L_compute = lambda_comp * p_full.mean()
L_total = L_lm + L_compute
```

### 4. Warmup Phase

Phase 1: Full attention only (router OFF). Wait until shift becomes meaningful.
Phase 2: Enable routing after ~15% loss drop from initial.

```python
scheduler = WarmupScheduler(threshold=0.15)

# At each eval
if scheduler.should_transition(eval_loss):
    for block in model.blocks:
        block.set_phase(phase1=False)
```

### 5. Weight Tying

RA output projection initialized from FULL projection tail. Creates relationship
where RA learns compressed view. Since output_embeddings = E.T in most models,
RA and E span related subspaces.

## Design Rationale

**Why fewer heads (not smaller dimensions)?**
- Uniform head_dim = cleaner GPU kernels
- torch.compile sees static shapes
- RA cheapness from projection, not weird dimensions

**Why |x - E(x)|?**
- E(x) = context-free "default meaning"
- x = contextual representation
- Gap = how much context bent the token
- Cheap: just subtraction and norm

**Why 2-way routing?**
- RA: Cheap attention with fewer heads (e.g., 3 heads)
- FULL: Full attention with more heads (e.g., 9 heads)
- Simpler than 4-way: NONE rarely useful, BOTH loses information by averaging

## KVSplice Compatibility

Architecture preserves symmetry for KV cache compression:
- Same QK^T geometry for all heads
- Clean head group partitioning
- Router decisions don't affect attention structure
- Future: router confidence informs pruning aggressiveness

## Usage

```python
from ra import RAConfig, RABlock, WarmupScheduler

cfg = RAConfig(
    d_model=768,
    n_heads=12,
    ra_head_frac=0.25,
    warmup_loss_drop=0.15,
)

# Replace attention in GPT-2 block
block = RABlock(cfg, layer_idx=0)

# In training loop
scheduler = WarmupScheduler(cfg.warmup_loss_drop)

# Forward
e_tok = embedding(input_ids)
out = block(x, e_tok=e_tok)

# Check phase transition at eval
if scheduler.should_transition(eval_loss):
    for block in model.blocks:
        block.set_phase(phase1=False)
    print("Phase 2: routing enabled")

# Compute penalty for loss (phase 2 only)
penalty = block.compute_penalty(x, e_tok)
loss = lm_loss + lambda_comp * penalty
```

## Configuration

```python
@dataclass
class RAConfig:
    d_model: int = 768           # Model dimension
    n_heads: int = 12            # Total heads
    block_size: int = 1024       # Max sequence length
    ra_head_frac: float = 0.25   # Fraction for RA group
    router_hidden: int = 16      # Router MLP hidden dim
    router_bias_full: float = -1.0  # Discourage expensive paths
    warmup_loss_drop: float = 0.15  # Trigger for phase 2
    tie_ra_proj: bool = True     # Init RA from FULL proj
    dropout: float = 0.0
```

### Quick Testing

To skip the RA warmup phase and enable routing from the start:

```bash
# CLI flag
python gpt2/train.py --architecture unified-ra --ra-step 1 --skip-ra-warmup

# Environment variable (for test matrix)
SKIP_RA_WARMUP=1 make
```

This is useful for quick validation that routing works correctly.

## Training Schedule

1. **Phase 1** (steps 0 to trigger): Full attention only
   - Router and RA routing disabled
   - Embeddings learn meaningful geometry
   - shift = |x - E(x)| becomes informative

2. **Trigger**: After ~15% eval loss drop
   - At least 2 evals completed
   - shift now correlates with contextual hardness

3. **Phase 2** (after trigger): Routing enabled
   - Router decides compute tier per token
   - Compute penalty encourages cheap paths
   - Model learns when full attention is worth it

## Experiments

### Baseline vs RA (1-hour each)

Compare GPT-2 124M:
- **Baseline**: All heads as FULL, no routing
- **RA**: Router decides, compute penalty active

Both use same code path through RABlock for fair comparison.
Baseline keeps phase1=True throughout; RA transitions to phase2.

Key metrics:
- Final eval loss/perplexity
- Average router distribution (p_ra, p_full)
- Compute penalty over training
- Loss curve smoothness at phase transition

## TODO: Future Enhancements

### Smooth Phase Transition
Instead of hard switch at 15% loss drop, gradually blend in routing strength
over N steps. Use linear interpolation: `alpha = min(1.0, steps_since_trigger / N)`.
This prevents training instability at transition.

### Router Feature Normalization
Apply LayerNorm to router features before MLP. Currently raw norms and dots
have different scales which may cause training issues.

### Per-Layer Routing Thresholds
Early layers may need more FULL attention (building representations), later
layers can use more RA (compressing/retrieving). Learn per-layer bias terms
or use fixed schedule based on layer depth.

### Learned Temperature
Add temperature parameter to router softmax that sharpens during training.
Start with high temperature (soft routing), anneal to low (hard routing).
`probs = softmax(logits / temperature)`

### KV Cache Pruning Integration
Use router confidence to inform KV pruning aggressiveness. High p_ra tokens
are "easy" and can have more aggressive KV pruning. Connect to KVSpliceAttention.

### Skip Computation Optimization
In production, actually skip FULL head computation when p_ra is high (not just
mix outputs). Requires dynamic computation graph or head-level masking.

## Mathematical Identity: Transpose Optimization

### The Identity

For standard and reciprocal attention:

```
(Q @ K.T).T = K @ Q.T
```

This is a fundamental property: (AB)^T = B^T @ A^T, so (Q @ K.T).T = K.T.T @ Q.T = K @ Q.T

### GPU Kernel Implications

**Naive approach** (two separate matmuls):
```python
attn_std = Q @ K.T      # Standard: [B,H,T,T]
attn_ra  = K @ Q.T      # Reciprocal: [B,H,T,T]
```

**Optimized approach** (one matmul + transpose):
```python
attn_std = Q @ K.T                    # [B,H,T,T]
attn_ra  = attn_std.transpose(-2, -1) # Near-free view operation
```

The transpose is essentially free on GPU (just a stride change, no data copy),
eliminating the second matmul entirely for ~2× speedup in attention compute.

### Implementation in Flash Attention

When using `F.scaled_dot_product_attention` (Flash Attention), we can't access
the internal matmul. Instead, the optimization is achieved via **argument swapping**:

```python
# Standard: Q @ K.T @ V
attn_out = F.scaled_dot_product_attention(q, k, v, is_causal=True)

# Reciprocal: K @ Q.T @ V (swap q and k arguments)
attn_out = F.scaled_dot_product_attention(k, q, v, is_causal=True)
```

This achieves the same effect - PyTorch's fused kernel handles the swapped
arguments efficiently. The RA_MLA_Flash implementation uses this approach.

### Semantic Meaning

The transpose flips the attention direction:
- `attn_std[b,h,i,j]` = how much token i attends to token j
- `attn_ra[b,h,i,j] = attn_std[b,h,j,i]` = how much token j attends to token i

This is the **reciprocal** relationship that enables Markov chain balance -
information flows bidirectionally across layers when alternating between
standard and reciprocal attention.

## Mathematical Foundation: Entropic Optimal Transport

### SPDA Paper Result

The paper "Scaled Dot-Product Attention as One-Sided Entropic Optimal Transport"
(August 2025, https://arxiv.org/pdf/2508.08369) proves that standard transformer
attention is exactly solving an Entropic Optimal Transport (EOT) problem.

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

The attention scores enter a log-sum-exp potential:

```
φ(s) = τ log Σ_j exp(s_j / τ)
```

Its gradient gives the attention distribution:

```
∇φ(s) = softmax(s / τ)
```

And its **Hessian** is:

```
∇²φ(s) = (1/τ²)(diag(p) - p pᵀ)
```

This Hessian is exactly the **Fisher Information Matrix (FIM)** of the categorical
distribution p. Therefore:

```
Hessian of log-sum-exp = Fisher Information Matrix
```

The FIM describes the curvature geometry of attention:
- **Eigenvalues**: Information density in each direction
- **Large eigenvalues**: High curvature, sensitive to perturbations
- **Small eigenvalues**: Flat directions, easy optimization

### Why This Matters for RA/SBA

**RA (Reciprocal Attention)** alternates between forward and reverse EOT:

```
F_fwd  from softmax(Q Kᵀ / τ)    # Forward geometry
F_rev  from softmax(K Qᵀ / τ)    # Reverse geometry
```

Each layer experiences one geometry, alternating across depth.

**SBA (Symmetric Bidirectional Attention)** mixes both within each layer:

```
p_SBA = α · p_fwd + (1-α) · p_rev
```

Since FIM is positive semi-definite, the effective curvature is:

```
F_SBA ≈ α · F_fwd + (1-α) · F_rev
```

This **convex combination cannot increase the maximal eigenvalue** beyond
the worse of the two. SBA produces:
- Smoother curvature spectrum
- More stable optimization
- Better-conditioned Fisher geometry

### Backward Pass = Advantage Gradients

The SPDA paper also shows that attention gradients have the exact structure
of REINFORCE policy gradients:

```
∂L/∂s_j = -(p_j / τ) · (u_j - E_p[u])
```

This is an **advantage update** (reward minus baseline), revealing that
attention learns via advantage-style policy optimization.

### Implications for KV Compression

The FIM defines which temporal directions are information-critical:

- **High FIM eigenvalues** = important temporal modes (keep these)
- **Low FIM eigenvalues** = flat directions safe to discard

KVSplice can use FIM-guided compression:

```python
# Calibration: compute average FIM over samples
F = mean_i(diag(p_i) - p_i @ p_i.T)
U, Λ, _ = svd(F)  # Eigenvectors sorted by information

# Compression: project K/V onto top-r FIM eigenvectors
C = U_r.T @ K     # Compressed representation
K_hat = U_r @ C   # Reconstruction
```

This is more principled than PCA because it preserves **information structure**
rather than just variance.

### Fisher Metrics in Training

We log Fisher spectrum metrics to W&B:

- `fisher/layer{i}/head{h}/eigmax`: Maximum eigenvalue (curvature sharpness)
- `fisher/layer{i}/head{h}/trace`: Sum of eigenvalues (total information)
- `fisher/layer{i}/head{h}/cond`: Condition number (curvature ratio)

Lower eigmax and better conditioning indicate smoother optimization.
SBA typically shows lower eigmax than standard attention due to the
geometry averaging effect.

### Summary: The Induction Chain

1. SPDA shows attention = solution to an EOT problem
2. Its curvature is given by the Hessian of log-sum-exp
3. This Hessian is exactly the Fisher Information Matrix
4. RA and SBA manipulate two EOT geometries (forward/reverse)
5. SBA mixes these geometries → smoother curvature/FIM
6. FIM eigenvalues reveal information structure of attention
7. KV compression is best done by discarding low-Fisher directions
8. Fisher metrics quantify optimization stability during training

This mathematical foundation provides principled justification for:
- SBA's smoother training compared to standard attention
- Fisher-guided KV compression outperforming PCA
- The inductive bias of reversible Markov chains in RA
