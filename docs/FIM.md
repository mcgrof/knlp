# Fisher Information Matrix (FIM) in Attention

## What is the Fisher Information Matrix?

The Fisher Information Matrix (FIM) measures how much information an observation
carries about an unknown parameter. In the context of attention mechanisms, it
quantifies the sensitivity of attention distributions to changes in the input.

Think of it as a "curvature map" of the optimization landscape. High curvature
means small changes in parameters cause large changes in outputs (sensitive,
hard to optimize). Low curvature means the landscape is flatter (stable, easier
to optimize).

## Key Discovery: FIM Diagonal ≈ Adam exp_avg_sq

A fundamental result connects FIM to Adam optimizer state. The diagonal FIM
approximation we use for quantization and compression is mathematically
equivalent to Adam's second moment:

```
FIM_diag(θ) = E[(∂L/∂θ)²] = E[g²]
Adam exp_avg_sq = β₂ · exp_avg_sq + (1-β₂) · g² ≈ E[g²]
```

This equivalence, validated by [Squisher (2025)](https://arxiv.org/abs/2507.18807),
has profound implications for our R&D:

| Application | Explicit FIM Cost | Adam State Cost | Savings |
|-------------|-------------------|-----------------|---------|
| Layer sensitivity | 100s of batches | Zero (free) | 100% |
| Quantization guidance | Calibration pass | Zero (free) | 100% |
| Pruning importance | Separate computation | Zero (free) | 100% |

### Why This Matters

**bitter7 pruning works because it directly uses FIM diagonal**:
```python
importance = |w| × (exp_avg_sq + ε)^0.25  # exp_avg_sq ≈ FIM diagonal
```

This explains the 15.6% improvement over magnitude pruning: bitter7 leverages
the accumulated Fisher Information that Adam has already computed, identifying
parameters with high gradient variance (sensitive to perturbation).

**Mobile weight packing uses the same signal**:
```python
fim_score[tensor] = Σ (param.grad ** 2) / num_batches  # Explicit FIM diagonal
```

Both methods identify the same sensitive weights because they compute the
same underlying quantity: E[g²].

### Future Integration (Hypothesis)

Currently, KVSplice and Reciprocal Attention use **post-training FIM trace
analysis** on calibration data with frozen weights. The FIM-Adam equivalence
*suggests* we might extract importance from Adam state:

```python
# Hypothetical (needs validation):
def get_layer_importance_from_adam(optimizer, layer_name):
    """Extract FIM approximation from Adam state (zero extra cost)."""
    for param in get_layer_params(layer_name):
        state = optimizer.state[param]
        if 'exp_avg_sq' in state:
            return state['exp_avg_sq'].mean().item()  # ≈ FIM diagonal
    return 0.0
```

**Important caveat**: This is **unvalidated** for KVSplice/RA. Key differences:
- Adam exp_avg_sq accumulates **during training** as the model changes
- Post-training FIM is computed on **frozen weights** with calibration data
- Layer rankings may differ due to timing and distribution differences

Before using exp_avg_sq for KVSplice/RA layer selection, we must empirically
verify that training-time accumulation correlates with post-training analysis.

See [docs/hierarchical-tiering.md](hierarchical-tiering.md) for how this
unifies our compression, pruning, and tiering research.

## Why We Care About FIM in Attention

The SPDA paper ("Scaled Dot-Product Attention as One-Sided Entropic Optimal
Transport") proves that attention solves an Entropic Optimal Transport (EOT)
problem. The attention scores enter a log-sum-exp potential, and the Hessian
(second derivative) of this potential is exactly the Fisher Information Matrix.

For attention with probabilities `p = softmax(scores / τ)`:

```
FIM = (1/τ²) * (diag(p) - p * p^T)
```

This matrix tells us:
- **How sensitive** the attention distribution is to perturbations
- **How curved** the optimization landscape is at this point
- **Which directions** in token space carry the most information

## FIM Metrics We Track

We log several FIM metrics during training to understand attention geometry:

### eigmax (Maximum Eigenvalue)

**What it is**: The largest eigenvalue of the Fisher Information Matrix.

**What it means**: The sharpest curvature direction in the optimization
landscape. High eigmax = sharp peaks/valleys = sensitive to small changes.
Low eigmax = flatter landscape = more stable optimization.

**Why it matters**: Learning rate needs to be smaller than 2/eigmax for
stability. Lower eigmax allows larger learning rates and more stable training.

**FIM eigmax (mean)**: Average of eigmax across all heads in a layer, computed
over the last 100 training samples. This gives a per-layer summary of
optimization difficulty.

### trace (Total Fisher Information)

**What it is**: Sum of all eigenvalues of the FIM.

**What it means**: Total amount of information in the attention distribution
across all directions. Think of it as the "total curvature mass."

**Why it matters**: Higher trace means more total information, but also more
directions that need careful optimization.

### energy_r8, energy_r16 (Energy Concentration)

**What it is**: Fraction of total Fisher energy (trace) captured in the top 8
or 16 eigenmodes.

**What it means**: How concentrated the information is. High energy
concentration (e.g., 90% in top 8 modes) means most information is in a few
directions. Low concentration (e.g., 37% in top 16 modes) means information is
diffuse across many directions.

**Why it matters**: High concentration suggests we could compress effectively
using low-rank approximations (keep top-k modes, discard the rest). Low
concentration means compression is harder—information is spread out.

### decay (Spectral Concentration)

**What it is**: Ratio of maximum eigenvalue to the 5th eigenvalue (eigmax / λ_5).

**What it means**: How quickly eigenvalues drop off. High decay = eigenvalues
drop fast = information concentrated in top modes. Low decay = eigenvalues
decrease slowly = information spread across many modes.

**Why it matters**: Another indicator of whether low-rank compression will work
effectively.

## Why We Added FIM Metrics

We initially hypothesized that Reciprocal Attention (RA) might concentrate
Fisher Information into fewer modes, making attention more compressible. The
SPDA theoretical framework suggested that alternating Q@K.T and K@Q.T could
change the information geometry.

**Hypothesis**: RA produces higher energy concentration (energy_r16 → 1.0),
enabling better low-rank compression of the KV cache.

**Result**: Hypothesis rejected. RA shows:
- **Lower eigmax** (0.0352 vs higher in MLA) = flatter optimization, not sharper
- **Similar trace** = no increase in total Fisher information
- **Low energy concentration** (37% in top 16 modes) = information diffuse, not concentrated

The FIM metrics revealed that RA's value comes from **smoother optimization
geometry** (lower eigmax, easier training), not from concentrating information
into compressible modes.

## Mathematical Introspection: What Does RA Add?

**Motivation**: The SPDA result provides a principled way to analyze attention
geometry. If alternating Q@K.T and K@Q.T changes the optimization landscape,
FIM metrics should reveal how.

**Research Question**: Does RA change the Fisher Information geometry in ways
that enable better cache compression?

### FIM Data from Experiments

Source: test_matrix_results_20251123_231956 (W&B project:
gpt2-kvsplice-ablation-w7900-mla-fixed)

| Architecture | eigmax | energy_r8 | energy_r16 | Interpretation |
|-------------|--------|-----------|------------|----------------|
| RA+MLA | 0.0352 | 0.223 | 0.373 | Low concentration, flat curvature |
| RA+MLA+KVSplice | 0.0341 | 0.220 | 0.370 | Similar geometry, compression orthogonal |

**Key Findings**:

1. **Low energy concentration**: Only ~37% of Fisher energy in top 16 modes
   across all architectures. Would need r>16 to capture 90% energy.

2. **No FIM improvement from compression**: KVSplice shows nearly identical
   energy concentration (0.370 vs 0.373) despite 50% cache reduction.

3. **Slight eigmax reduction**: KVSplice has marginally lower eigmax (0.0341 vs
   0.0352), suggesting flatter curvature, but effect is small.

### Conclusion

**FIM analysis did not provide clear guidance for compression decisions**.
Despite low energy_r16 (~0.37), KVSplice with d=128 (50% compression)
empirically improves quality by 11%. The learned compression appears to find
task-specific structure that FIM-based metrics don't capture.

**Interpretation**: Fisher Information measures optimization geometry, not
necessarily task-relevant information. Learned compression (KVSplice) acts as
beneficial regularization that forces representations into information-dense
subspaces, but this structure isn't visible in variance-based or FIM-based
metrics. The value comes from end-to-end learning, not from following
prescribed compression directions indicated by FIM.

## Relationship to RA's Inductive Bias

RA (Reciprocal Attention) alternates between forward and reverse EOT problems:

```
F_fwd  from softmax(Q * K^T / τ)    # Forward geometry
F_rev  from softmax(K * Q^T / τ)    # Reverse geometry
```

Each layer experiences one geometry, alternating across depth. The FIM metrics
revealed that this alternation produces:

- **Flatter curvature** (lower eigmax) = easier optimization
- **Better gradient flow** = compensates for compression losses in MLA
- **More stable training** = particularly helpful for compressed representations

But RA does **not** produce:
- Higher total Fisher Information (trace similar/lower)
- Concentrated information modes (energy_r16 remains ~37%)
- Improved compressibility from geometric changes alone

This explains why RA helps MLA (optimization benefits) but doesn't predict or
enable further compression (no structural changes to information geometry).

## Practical Implications

**For RA usage**:
- Expect optimization benefits (flatter curvature, better gradient flow)
- Do not expect FIM-guided compression opportunities
- Learned compression (KVSplice) works independently of RA's geometric properties

**For compression research**:
- FIM metrics don't predict learned compression effectiveness
- Task-specific information structure differs from geometric information (FIM)
- End-to-end learning finds compressible structure that variance/FIM miss

## Actionable Interpretations for Compression

While early experiments showed FIM didn't predict learned compression
effectiveness, FIM metrics provide valuable guidance for identifying
compression targets, critical heads to protect, and tiering strategies.

### High Trace Hotspots

**What it means mathematically**: Sum of eigenvalues → total curvature /
total sensitivity of that parameter block.

**What it means for the model**:
- Head/layer doing disproportionate representational work
- Encodes features heavily used across the dataset
- Critical feature extractor or key-value transformer

**Implications for KV compression**:
- **Cannot be compressed aggressively** - stores information that
  frequently affects next-token distribution
- Structure is still learnable (PCA, low-rank) even at high trace
- Often remain low-rank despite high trace

**Implications for pruning/tiering**:
- **DO NOT prune these heads** - critical for model performance
- **Best candidates for**:
  - Quantization-aware retraining
  - Adaptive precision (FP16/BF16 for critical, FP8 for others)
  - Dual-tier KV cache (keep these in fast tier)

**Implications for routing/scheduling**:
- High-trace heads are where dynamic attention routing is most valuable
- Easy tokens can skip these heads
- Hard tokens should go through these heads

**Example**: `trace > 0.95` in layer11/head5 → critical head, protect it

### High Eigmax Hotspots

**What it means mathematically**: Largest eigenvalue = maximum curvature
direction. High eigmax → extremely sensitive to perturbations along one
specific direction.

**What it means for the model**:
- Specialized, sharp, learning highly non-smooth features
- Handles rare / high-stakes token interactions
- Often corresponds to:
  - Rare token handling
  - Long-tail disambiguation
  - Discrete structural cues (syntax repair)
  - Retrieval-style behaviors (pre-GQA)

**Implications for compression**:
- **Top-eigenvector projection is extremely powerful**
- Dominant direction captures disproportionate information
- Usually rank-1 or rank-2 reconstruction recovers most behavior

**Implications for pruning/tiering**:
- **DO NOT prune the whole head**
- **Can prune orthogonal low-signal directions**
- Can compress aggressively in orthogonal space

**Best candidates for**:
- Residual-branch gating
- Attention reuse / caching
- Reversible layers
- **KVSplice-style latent KV compression**
- FIM-guided routing

**Example**: `eigmax > 0.2` in layer6/head8 → specialized head, use
top-eigenvector projection

### High Condition Number Hotspots

**What it means mathematically**: `cond(FIM) = eigmax / eigmin` →
anisotropy. High cond means curvature is very stretched → gradient flows
overwhelmingly along 1-few directions, while others are flat.

**What it means for the model**:
- Highly anisotropic
- Has a small set of "key directions" that matter a LOT
- Other directions are nearly irrelevant
- Indicates natural low-rank structure

**Implications for compression** (BEST COMPRESSION TARGETS):
- If `cond >> 1e4` can safely compress:
  - KV cache (50-90% reduction via low-rank projection)
  - Projections (W_q, W_k, W_v quantization to FP8/FP4)
  - MLP intermediate activations
- Effective rank is small regardless of total dimensionality

**Implications for pruning**:
- **Prune low-energy directions → nearly zero loss**
- Head pruning sometimes possible, but direction pruning is safer

**Implications for tiering**:
- High-cond heads are:
  - Expensive in dimension
  - BUT cheap to approximate (most info in top few eigenvectors)
- Great for:
  - Mixed precision KV caches (FP8/FP4)
  - Rank-limited KVSplice
  - Gated heads
  - Adaptive attention at inference

**Condition number scale**:
- `cond < 1e3`: Well-conditioned / isotropic (hard to compress)
- `cond ~ 1e5`: Moderately anisotropic (moderate compression)
- `cond ~ 1e7`: Strong anisotropy (good compression target)
- `cond > 1e7`: Extremely ill-conditioned (excellent compression target)

**Example**: `cond = 2.7e7` in layer6/head8 → compress KV by 80-90%

### Energy Concentration (energy_r8, energy_r16)

**What it means**: Fraction of total energy captured by top k eigenvalues.

**Interpretation scale**:
- `energy_r8 ≥ 0.95`: Almost all energy in top 8 modes → very low-rank
- `energy_r8 ≥ 0.90`: Most energy in top 8 modes → low effective rank
- `energy_r8 ≥ 0.80`: Substantial low-rank structure
- `energy_r8 < 0.80`: Energy spread across spectrum (harder to compress)

**Effective rank**: Smallest k where `energy_rk ≥ 0.9`

**Implications**:
- High energy concentration → can use rank-k approximation
- `effective_rank < 16` → excellent candidate for low-rank compression
- Combined with high condition number → best compression targets

### Compression Strategy Decision Tree

```
For each head/layer:

1. Check trace:
   - If trace > 0.95: CRITICAL HEAD
     → Protect from compression
     → Use adaptive precision (keep FP16/BF16)
     → Place in fast-tier cache

2. Check condition number:
   - If cond > 1e7 AND trace < 0.95: EXCELLENT COMPRESSION TARGET
     → KV cache: 70-90% compression via low-rank
     → Quantize to FP8/FP4
     → Prune low-energy directions

   - If cond > 1e5 AND trace < 0.95: GOOD COMPRESSION TARGET
     → KV cache: 50-70% compression
     → Quantize to FP8
     → Moderate pruning

   - If cond < 1e5: HARD TO COMPRESS
     → Minimal compression (< 30%)
     → Keep higher precision

3. Check eigmax:
   - If eigmax > 0.15: SPECIALIZED HEAD
     → Use top-eigenvector projection
     → KVSplice-style latent compression
     → Keep in routing-aware tier

4. Check energy_r8:
   - If energy_r8 > 0.9 AND cond > 1e6: VERY COMPRESSIBLE
     → Rank-8 approximation captures 90% of information
     → Can use aggressive low-rank compression

   - If energy_r8 < 0.8: SPREAD SPECTRUM
     → Need higher rank for compression
     → Use learned compression (KVSplice) instead of PCA
```

### Example: Tiered Compression Strategy

From `gpt2-ra-v2-h100` analysis:

**Layer 6, Head 8** (Best compression target):
```
trace = 0.85, cond = 2.7e7, eigmax = 0.28, energy_r8 = 0.35
```
**Strategy**:
- KV cache: 80% compression (256 → 51 dims via low-rank)
- Projections: Quantize to FP8
- Top-eigenvector projection: Keep top-1 direction at full precision
- Place in slow-tier cache (can reconstruct on demand)

**Layer 11, Head 5** (Critical head):
```
trace = 0.975, cond = 3.5e6, eigmax = 0.08, energy_r8 = 0.31
```
**Strategy**:
- KV cache: Minimal compression (30% max)
- Projections: Keep FP16/BF16
- NO pruning
- Place in fast-tier cache (always available)

**Layer 0, Head 3** (Moderate):
```
trace = 0.88, cond = 5.2e6, eigmax = 0.06, energy_r8 = 0.28
```
**Strategy**:
- KV cache: 50% compression
- Projections: Quantize to FP8/INT8
- Moderate pruning of low-energy directions
- Place in medium-tier cache

## Tool Integration

### Automated FIM Analysis

Use the unified FIM analysis tool to generate actionable compression
recommendations:

```bash
# Extract and analyze FIM metrics from W&B
source ~/envs/w7900-ml/bin/activate
python scripts/analyze_fim_metrics.py \
  --entity mcgrof-citizen \
  --project gpt2-ra-v2-h100 \
  --output-dir test_matrix_results \
  --output-summary fim.txt

# View human-readable summary with compression recommendations
cat test_matrix_results/fim.txt
```

The summary includes:
- Per-run FIM statistics
- Cross-run comparisons
- Hotspot detection (extreme metric values)
- Compression potential scores
- Interpretation guide

### W&B Visualization

Log FIM insights to W&B for interactive exploration:

```python
from scripts.fim_wandb_viz import log_fim_to_wandb

log_fim_to_wandb(
    entity="mcgrof-citizen",
    project="gpt2-ra-v2-h100",
    run_id=wandb_run.id,
    df=fim_dataframe,
    global_stats=global_stats,
    low_rank_stats=low_rank_stats,
    summary_text=summary_text,
)
```

This creates:
- Compression potential heatmap (layer×head grid)
- Scatter plot (condition number vs trace)
- Single-number insights (compression_potential, efficiency_score)
- HTML panel with full interpretation

See `docs/fim_analysis.md` for detailed documentation.

## References

- SPDA Paper: "Scaled Dot-Product Attention as One-Sided Entropic Optimal Transport"
- Code for FIM metrics: `gpt2/trainers/ra.py` (FisherMetricsCallback)
- Experimental results: [test_matrix_results_20251123_231956](https://github.com/mcgrof/knlp-key-results/tree/main/key_results/test_matrix_results_20251123_231956)
- FIM Analysis Tools: `scripts/analyze_fim_metrics.py`, `scripts/fim_wandb_viz.py`
- Full FIM Analysis Guide: `docs/fim_analysis.md`
