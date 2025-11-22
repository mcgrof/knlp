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

### 2. Context Router

4-way routing based on shift and other cheap features:

```python
shift = |x - E(x)|                 # [B, T]
features = [shift, ||x||, ||E||, <x,E>]
probs = softmax(mlp(features))     # [B, T, 4]
# probs: p_none, p_ra, p_full, p_both
```

Output mixed according to router:

```python
out = p_none * x + p_ra * out_ra + p_full * out_full + p_both * (ra+full)/2
```

Compute penalty discourages expensive paths:

```python
L_compute = lambda_comp * (p_full + p_both).mean()
L_total = L_lm + L_compute
```

### 3. Warmup Phase

Phase 1: Full attention only (router OFF). Wait until shift becomes meaningful.
Phase 2: Enable routing after ~15% loss drop from initial.

```python
scheduler = WarmupScheduler(threshold=0.15)

# At each eval
if scheduler.should_transition(eval_loss):
    for block in model.blocks:
        block.set_phase(phase1=False)
```

### 4. Weight Tying

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

**Why 4-way routing?**
- NONE: Skip token-mixing (just residual)
- RA: Cheap attention with fewer heads
- FULL: Standard attention for hard tokens
- BOTH: "Clutch mode" when both help

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
- Average router distribution (p_none, p_ra, p_full, p_both)
- Compute penalty over training
- Loss curve smoothness at phase transition
