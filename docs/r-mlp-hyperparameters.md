# R-MLP Hyperparameters for Tuning

This document describes all hyperparameters available for R-MLP (Reciprocal MLP) tuning, based on the r-mlp-prune defconfig and cowboy experiment findings.

## Executive Summary

**Key Finding from Cowboy Experiment:**
- RA alone: 12% worse than baseline (harmful)
- R-MLP alone: 0.26% worse than baseline (nearly equivalent)
- R-MLP + KV pruning: Memory benefits with minimal quality loss

**Recommendation:** Focus on R-MLP without RA, tuning R_ff and KV pruning for optimal memory/quality tradeoff.

---

## Core R-MLP Parameters

### 1. `rmlp_R_ff` (Reciprocal Rank)

**What it controls:** Size of the reciprocal pathway in the MLP hidden dimension.

**Formula:**
```
D_ff_total = 3072 (for GPT-2 124M, expansion=4)
D_ff_std = D_ff_total - R_ff
```

**Tested values:**
- `R_ff=64`: Basic (2.08% of hidden dim)
  - Gates: w_std=0.9792, w_rec=0.0208
  - Result: 55.60 perplexity (+1.52% vs baseline)

- `R_ff=256`: Medium (8.33% of hidden dim)
  - Gates: w_std=0.9167, w_rec=0.0833
  - Result: Not yet tested

- `R_ff=512`: Large (16.67% of hidden dim)
  - Gates: w_std=0.8333, w_rec=0.1667
  - Result: Not yet tested

- `R_ff=1152`: Golden ratio (37.5% of hidden dim)
  - Gates: w_std=0.625, w_rec=0.375
  - Result: 54.91 perplexity (+0.26% vs baseline)

**Tuning guidance:**
- Larger R_ff gives more reciprocal capacity but reduces standard capacity
- Golden ratio (1152) performed best in cowboy experiment
- Consider testing intermediate values: 384, 768

**Memory impact:** R_ff doesn't significantly affect memory (same total parameters)

---

### 2. `rmlp_use_mixer` (1x1 Convolution on Reciprocal Path)

**What it controls:** Adds a learnable 1x1 linear transformation on the reciprocal hidden state.

**Values:**
- `False`: No mixer (default)
- `True`: Add `nn.Linear(R_ff, R_ff)` to enhance expressivity

**Formula:**
```python
if use_mixer:
    h_low = mixer(h_low)  # h_low: [B, T, R_ff]
```

**Parameters added:** `R_ff * R_ff` (e.g., 64*64 = 4K for R_ff=64)

**Tuning guidance:**
- Enables the reciprocal pathway to learn non-linear transformations
- Adds minimal parameters for small R_ff
- May improve quality at cost of slightly more computation
- Test in V5 ablation step

---

### 3. `rmlp_use_gates` (Per-Token Gating)

**What it controls:** Adds learnable per-token gating on the reciprocal contribution.

**Values:**
- `False`: No per-token gates (default)
- `True`: Add learnable `gate_alpha` parameter

**Formula:**
```python
if use_gates:
    # gate_alpha initialized to 0.1
    out = h_std + gate_alpha * h_low
else:
    out = w_std * h_std + w_rec * h_low  # Geometric mixing
```

**Parameters added:** 1 scalar parameter per layer (12 params for GPT-2 124M)

**Tuning guidance:**
- Allows model to learn how much reciprocal contribution to use
- Very lightweight (only 1 param per layer)
- May help with training stability
- Test in V6 ablation step

---

## KV Cache Pruning Parameters

### 4. `kv_cache_prune` (Enable KV Pruning)

**What it controls:** Whether to enable KV cache pruning.

**Values:**
- `False`: Full KV cache (1024 tokens)
- `True`: Enable pruning (controlled by k or learned ratio)

**Memory impact:** Significant! See memory savings calculation below.

---

### 5. `kv_prune_k` (Fixed Pruning Ratio)

**What it controls:** Fixed number of tokens to keep in KV cache.

**Values:**
- `k=391`: Golden ratio (391/1024 ≈ 0.382)
  - Keeps 38.2% of tokens
  - Prunes 61.8% of KV cache
  - Memory savings: 178 MB per batch of 8 (61.8% reduction)

**Formula:**
```
memory_saved = (context_len - k) / context_len * full_kv_size
             = (1024 - 391) / 1024 * 288 MB
             = 178 MB per batch
```

**Tuning guidance:**
- Golden ratio (0.382) is a good starting point
- Can test: k=256 (25%), k=512 (50%), k=768 (75%)
- Trade-off: Lower k = more memory savings but may hurt quality

---

### 6. `kv_prune_learned` (Learned Pruning Ratio)

**What it controls:** Whether the pruning ratio is learned or fixed.

**Values:**
- `False`: Use fixed `kv_prune_k`
- `True`: Learn `keep_ratio` parameter

**Formula:**
```python
if kv_prune_learned:
    # keep_ratio initialized to kv_prune_init_ratio (e.g., 0.382)
    k = int(keep_ratio * context_len)  # Differentiable
    # Gradients flow to keep_ratio
```

**Parameters added:** 1 scalar parameter per attention layer (12 params for GPT-2 124M)

**Tuning guidance:**
- Allows model to learn optimal pruning ratio
- V18 (learned) slightly outperformed V17 (fixed): 54.91 vs 55.60 perplexity
- Requires `kv_prune_init_ratio` to set starting point

---

### 7. `kv_prune_init_ratio` (Initial Learned Ratio)

**What it controls:** Initial value for learned keep_ratio.

**Values:**
- `0.382`: Golden ratio (default)
- `0.25`: Aggressive pruning start
- `0.50`: Conservative pruning start

**Tuning guidance:**
- Start at golden ratio (0.382) based on empirical results
- Model will adjust during training if `kv_prune_learned=True`

---

### 8. `kv_prune_recency` (Recency Bias)

**What it controls:** How many of the most recent tokens to always keep.

**Values:**
- `64`: Keep last 64 tokens always (default)
- `32`: Smaller recency window
- `128`: Larger recency window

**Rationale:** Recent tokens are critical for autoregressive generation, so always keep them regardless of pruning.

**Tuning guidance:**
- 64 is a good default (1/16 of context)
- Adjust based on task: longer dependencies → larger recency

---

## Training Parameters

### 9. Training Time (`GPT2_MAX_TIME`)

**What it controls:** Maximum training time in seconds.

**Values:**
- `7200`: 2 hours (default, used in cowboy experiment)
- `28800`: 8 hours (for more thorough training)
- `600`: 10 minutes (quick test)

**Formula:**
```bash
GPT2_MAX_TIME=7200 make  # Override at runtime
```

**Tuning guidance:**
- Cowboy experiment used 2 hours
- Longer training may help R-MLP learn better reciprocal patterns
- 8-hour runs recommended for production evaluation

---

### 10. Variance-Guided Activation

**Status:** DISABLED in r-mlp-prune defconfig

**Rationale:** Geometric initialization eliminates need for delayed activation.

**Historical context:**
- Old approach: Delay reciprocal mechanisms until loss stabilizes
- New approach: Geometric init allows training from step 0 ("cowboy mode")

---

## Memory Savings Calculation

### Full KV Cache (GPT-2 124M)

```
KV size per token = 2 (K+V) * 12 layers * 768 dims * 2 bytes (bfloat16)
                  = 36,864 bytes = 36 KB

Full KV cache (context=1024, batch=8):
  = 36 KB * 1024 tokens * 8 batch
  = 288 MB
```

### With Pruning (k=391)

```
Pruned KV cache:
  = 36 KB * 391 tokens * 8 batch
  = 110 MB

Memory saved: 288 - 110 = 178 MB (61.8% reduction)
```

### Scaling to Larger Batch Sizes

```
Batch size 16:  576 MB → 220 MB (356 MB saved)
Batch size 32: 1152 MB → 440 MB (712 MB saved)
```

**Key insight:** KV pruning savings scale linearly with batch size. Essential for large-scale deployment.

---

## Recommended Tuning Strategy

### Phase 1: R_ff Sweep (V1-V4)

Test different reciprocal ranks to find optimal capacity split:
- V1: R_ff=64 (baseline)
- V2: R_ff=256 (medium)
- V3: R_ff=512 (large)
- V4: R_ff=1152 (golden, best from cowboy)

**Metrics to watch:**
- Validation perplexity (primary)
- Training loss convergence
- w_std/w_rec gate evolution (if logging works)

### Phase 2: Mechanism Ablation (V5-V6)

Once optimal R_ff is found, test additional mechanisms:
- V5: + mixer (enhanced expressivity)
- V6: + per-token gates (adaptive contribution)

**Hypothesis:** Mixer may help, per-token gates likely minimal impact.

### Phase 3: KV Pruning Tuning

Test different pruning ratios:
- k=256 (25% keep, aggressive)
- k=391 (38% keep, golden ratio)
- k=512 (50% keep, conservative)
- Learned ratio (adaptive)

**Goal:** Find maximum memory savings with <1% perplexity degradation.

---

## Configuration Files

### r-mlp-prune Defconfig

Location: `gpt2/defconfigs/gpt2-r-mlp-prune`

**Usage:**
```bash
make defconfig-gpt2-r-mlp-prune
make BASELINE=mcgrof-citizen/gpt2-kv-pruning/6dvbpfuh
```

**Steps configured:** V0-V6 (baseline + 6 R-MLP variations)

---

## Implementation Details

### Geometric Initialization

R-MLP uses geometric initialization based on dimensional capacity:

```python
D_ff = D_ff_std + R_ff
w_std_init = D_ff_std / D_ff  # Standard pathway weight
w_rec_init = R_ff / D_ff       # Reciprocal pathway weight
# Always sums to 1.0
```

**Examples:**
- R_ff=64:   w_std=0.9792, w_rec=0.0208 (48:1 ratio)
- R_ff=1152: w_std=0.6250, w_rec=0.3750 (5:3 ratio)

**Benefit:** Stable training from step 0, no warmup needed.

---

## Future Hyperparameters to Explore

### 1. Variable R_ff per Layer

Allow different reciprocal ranks for different layers:
- Early layers: Smaller R_ff (more standard capacity)
- Late layers: Larger R_ff (more reciprocal flow)

### 2. Attention-MLP Coupling

Coordinate KV pruning with R-MLP contribution:
- When KV cache is pruned, increase R_ff to compensate
- Learnable coupling strength

### 3. Dynamic Pruning

Adjust k based on input complexity:
- Easy sequences: Aggressive pruning
- Hard sequences: Conservative pruning

---

## References

- Cowboy experiment: `test_matrix_results_20251114_174752/`
- RA documentation: `docs/ra.md`
- Baseline W&B run: `mcgrof-citizen/gpt2-kv-pruning/6dvbpfuh`
- Implementation: `ra.py` (ReciprocalMLP class)
- Trainer: `gpt2/trainers/ra.py` (ablation steps)
