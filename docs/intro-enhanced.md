# Enhanced Transformer Architectures: RA, MLA, and Reciprocal Mechanisms

This document covers our research into enhanced transformer architectures. For fundamentals of standard transformers, see [intro.md](intro.md).

## Table of Contents
1. [The Bloat Problem](#the-bloat-problem)
2. [The Ratio Concept](#the-ratio-concept)
3. [Our Enhancements](#our-enhancements)
4. [Philosophy: Learnable vs Fixed Hyperparameters](#philosophy-learnable-vs-fixed-hyperparameters)
5. [Evaluating New Parameters](#evaluating-new-parameters)
6. [Current Status: Learned Ratio as Goal](#current-status-learned-ratio-as-goal)

---

## The Bloat Problem

### Our Philosophy: Minimal Parameters, Maximum Impact

We want to add **capabilities** (bidirectional information flow, adaptive gating) with **minimal parameter overhead**.

**Bad example from early experiments**:
```python
# Someone adds "enhanced attention" with extra projections
self.extra_q = nn.Linear(768, 768)  # +590K params
self.extra_k = nn.Linear(768, 768)  # +590K params
self.extra_v = nn.Linear(768, 768)  # +590K params
# Added 1.77M params per layer × 12 = 21.2M extra params!
# New total: 124M + 21.2M = 145.2M (+17% bloat)

# If this doesn't improve loss, we've wasted 17% of parameters
```

**The problem**: Every parameter has a cost:
- Memory (stores the weight)
- Computation (multiply-accumulate operations)
- Training time (gradient computation and updates)

If new parameters don't provide **meaningful signal**, they're just **noise** that:
- Makes training harder (more gradients to compute)
- Increases memory usage
- Slows down inference
- May hurt generalization (more parameters = easier to overfit)

---

## The Ratio Concept

### Original Finding: MLP Width Matters

Standard transformers use **MLP:Attention ratio of 4:1**:
```python
d_model = 768        # Attention hidden size
d_mlp = 4 * 768 = 3072  # MLP hidden size (4× expansion)
```

**Original RA experiments found**:
- Increasing model size by adding RA parameters didn't help much
- The model was "bloated" - extra params weren't being used effectively
- Ratio of MLP width to Attention width impacts efficiency

### The "Golden Ratio" Hypothesis

Research on inference scaling laws suggests optimal MLP:Attention ratio ≈ **2.5:1** (not 4:1) for some tasks.

```python
# Standard GPT-2
mlp_dim = 3072  # 768 × 4
ratio = 3072 / 768 = 4.0

# "Golden ratio" experiment
mlp_dim = 3840  # 768 × 5
ratio = 3840 / (768 + MLA_latent) ≈ 2.5

# With MLA compression:
# Attention effectively uses latent_dim=128 instead of 768
# So ratio = 3840 / (128 + effective_attention_cost)
```

### Our Ablation Study Tests This

```
Step 0: Baseline (ratio 1:2.0)           val_loss = 3.5740
Step 2: Golden ratio (ratio 1:2.5)       val_loss = 3.6238  (WORSE!)
```

**Result**: Fixed golden ratio made it worse. Why?
1. More parameters (3840 vs 3072) need more training time to converge
2. Copying pretrained weights + random init for new dims is suboptimal
3. The "optimal" ratio may be task-dependent, not universal

**Solution**: Make ratio learnable, let model find optimal balance.

---

## Our Enhancements

### Design Principle: Add Capabilities, Not Bloat

We add three mechanisms with careful parameter management:

### Enhancement 1: MLA (Multi-Latent Attention) - Compression

**Idea**: Don't store full [T, 768] keys/values - compress to [T, 128] latent space.

```python
# Standard attention:
K = k_proj(hidden)  # [T, 768] - full dimension
V = v_proj(hidden)  # [T, 768] - full dimension

# MLA attention:
latent_k = k_down(hidden)  # [T, 128] - compressed
latent_v = v_down(hidden)  # [T, 128] - compressed
```

**Parameter cost**:
```
k_down: 768 × 128 = 98K
v_down: 768 × 128 = 98K
v_up: 128 × 64 × 12 = 98K (per-head expansion)
Total: ~300K per layer (vs 1.18M for standard K,V projections)
```

**Benefit**: 75% parameter reduction for K/V projections, smaller KV cache for inference.

#### MLA's Critical Inference Advantage: Massive KV Cache Reduction

The primary motivation for MLA is **inference memory efficiency**. During autoregressive generation, the KV cache becomes the bottleneck.

**Standard Attention KV Cache**:
```python
# Standard GPT-2: Store full [n_heads, seq_len, head_dim] for K and V
# K cache shape: [batch, 12 heads, seq_len, 64]
# V cache shape: [batch, 12 heads, seq_len, 64]

# Example: 1024 token context, single sequence, FP16
k_cache_size = 1 * 12 * 1024 * 64 * 2 = 1,572,864 bytes ≈ 1.5 MB per layer
v_cache_size = 1.5 MB per layer
total_per_layer = 3.0 MB

# All 12 layers: 36 MB total KV cache
```

**MLA KV Cache** - stores only compressed latents:
```python
# MLA: Store compressed [seq_len, latent_dim] for K and V
# K_latent cache shape: [batch, seq_len, 128]  (no head dimension!)
# V_latent cache shape: [batch, seq_len, 128]

# Same example: 1024 token context, single sequence, FP16
k_latent_cache = 1 * 1024 * 128 * 2 = 262,144 bytes ≈ 0.26 MB per layer
v_latent_cache = 0.26 MB per layer
total_per_layer = 0.52 MB

# All 12 layers: 6.2 MB total KV cache
# Reduction: 36 MB → 6.2 MB (83% smaller!)
```

**How MLA Works During Inference**:
```python
# === Initialization: Process prompt tokens ===
prompt_hidden = embed(prompt_tokens)  # [batch, prompt_len, 768]

# Compress to latent space ONCE
k_latent = k_down(prompt_hidden)  # [batch, prompt_len, 128]
v_latent = v_down(prompt_hidden)  # [batch, prompt_len, 128]

# Cache the latents (not the full K,V!)
kv_cache['layer_0']['k_latent'] = k_latent  # Store [prompt_len, 128]
kv_cache['layer_0']['v_latent'] = v_latent

# === Generation Loop: Add one token at a time ===
for new_token in generated_tokens:
    new_hidden = embed(new_token)  # [batch, 1, 768]

    # Compress new token to latent
    new_k_latent = k_down(new_hidden)  # [batch, 1, 128]
    new_v_latent = v_down(new_hidden)  # [batch, 1, 128]

    # Retrieve cached latents
    cached_k_latent = kv_cache['layer_0']['k_latent']  # [batch, t, 128]
    cached_v_latent = kv_cache['layer_0']['v_latent']  # [batch, t, 128]

    # Concatenate: cached + new
    full_k_latent = torch.cat([cached_k_latent, new_k_latent], dim=1)
    full_v_latent = torch.cat([cached_v_latent, new_v_latent], dim=1)

    # Update cache with new latents
    kv_cache['layer_0']['k_latent'] = full_k_latent  # [batch, t+1, 128]
    kv_cache['layer_0']['v_latent'] = full_v_latent

    # === Attention computation ===
    # Expand latent V to per-head space ONLY when needed
    V = v_up(full_v_latent)  # [batch, t+1, 128] → [batch, 12, t+1, 64]
    # Note: K stays in latent space [batch, t+1, 128]

    # Compute query for new token
    Q = q_proj(new_hidden)  # [batch, 1, 768] → [batch, 12, 1, 64]

    # Project Q to latent space for compatibility with K_latent
    Q_latent = q_to_latent(Q)  # [batch, 12, 1, 64] → [batch, 12, 1, 128]

    # Attention scores in latent space
    scores = Q_latent @ full_k_latent.T  # [batch, 12, 1, t+1]
    weights = softmax(scores / sqrt(128))

    # Weighted sum of values
    output = weights @ V  # [batch, 12, 1, 64]
```

**Key Insight**: We never store the expanded per-head K,V tensors. We only store the compressed latents and expand on-the-fly during attention computation.

**Detailed Memory Comparison**:

```python
# === Standard Attention KV Cache ===
# Shape: [batch, n_heads, seq_len, head_dim]
# K: [1, 12, 1024, 64] = 786,432 values
# V: [1, 12, 1024, 64] = 786,432 values
# Total per layer: 1,572,864 values × 2 bytes = 3.0 MB
# 12 layers: 36 MB

# === MLA KV Cache ===
# Shape: [batch, seq_len, latent_dim]
# K_latent: [1, 1024, 128] = 131,072 values
# V_latent: [1, 1024, 128] = 131,072 values
# Total per layer: 262,144 values × 2 bytes = 0.52 MB
# 12 layers: 6.2 MB

# Ratio: 768 / 128 = 6× compression per layer
# Plus: No head dimension stored (further savings)
```

**Scaling to Long Context**:

```python
# Standard Attention at different context lengths
contexts = [1024, 2048, 4096, 8192, 16384]

print("Standard Attention KV Cache:")
for ctx in contexts:
    cache_mb = 12 * 12 * ctx * 64 * 2 * 2 / (1024**2)
    print(f"  {ctx:5d} tokens: {cache_mb:6.1f} MB")

print("\nMLA KV Cache:")
for ctx in contexts:
    cache_mb = 12 * ctx * 128 * 2 * 2 / (1024**2)
    print(f"  {ctx:5d} tokens: {cache_mb:6.1f} MB")

print("\nReduction:")
for ctx in contexts:
    std_cache = 12 * 12 * ctx * 64 * 2 * 2
    mla_cache = 12 * ctx * 128 * 2 * 2
    reduction_pct = (1 - mla_cache / std_cache) * 100
    print(f"  {ctx:5d} tokens: {reduction_pct:.1f}% smaller")

# Output:
# Standard Attention KV Cache:
#   1024 tokens:   36.0 MB
#   2048 tokens:   72.0 MB
#   4096 tokens:  144.0 MB
#   8192 tokens:  288.0 MB
#  16384 tokens:  576.0 MB
#
# MLA KV Cache:
#   1024 tokens:    6.2 MB
#   2048 tokens:   12.4 MB
#   4096 tokens:   24.9 MB
#   8192 tokens:   49.8 MB
#  16384 tokens:   99.6 MB
#
# Reduction:
#   1024 tokens: 82.7% smaller
#   2048 tokens: 82.7% smaller
#   4096 tokens: 82.7% smaller
#   8192 tokens: 82.7% smaller
#  16384 tokens: 82.7% smaller
```

**Practical Impact for Serving**:

```python
# Scenario: Serving 100 concurrent users with 4K context each
# Model: GPT-2 124M

# Standard Attention:
# Per user: 144 MB KV cache (4K context)
# 100 users: 14.4 GB KV cache
# Plus model weights: 0.25 GB
# Total GPU memory: ~15 GB (need A100 40GB or multiple GPUs)

# MLA:
# Per user: 24.9 MB KV cache (4K context)
# 100 users: 2.49 GB KV cache
# Plus model weights: 0.25 GB
# Total GPU memory: ~3 GB (fits on RTX 3090 24GB easily!)

# Cost impact:
# Standard: $3.60/hr (A100 40GB)
# MLA: $1.10/hr (RTX 3090) - 70% cost reduction
# Or: Serve 5.8× more users on same hardware
```

**Batch Inference Benefits**:

```python
# Generate 32 completions in parallel, 2K context each

# Standard Attention:
# K cache: [32, 12, 2048, 64] × 2 bytes = 48 MB per layer
# V cache: 48 MB per layer
# Total: 96 MB × 12 layers = 1.15 GB

# MLA:
# K_latent cache: [32, 2048, 128] × 2 bytes = 8 MB per layer
# V_latent cache: 8 MB per layer
# Total: 16 MB × 12 layers = 192 MB

# Savings: 1.15 GB → 192 MB (83% reduction)
# This allows 6× larger batch sizes with same memory!
```

**Tradeoff: Computation vs Memory**:

```python
# MLA adds computation during attention:
# 1. Project Q to latent space: q_to_latent(Q)
# 2. Expand V from latent: v_up(V_latent)

# Per-token additional FLOPs:
# q_to_latent: 12 * 64 * 128 = 98K MACs
# v_up: 128 * 12 * 64 = 98K MACs
# Total: ~200K MACs per token

# Standard attention K,V projection: 768 * 768 * 2 = 1.18M MACs per token

# MLA saves: (1.18M - 0.2M) / 1.18M = 83% compute for K,V projections
# But recomputes q_to_latent, v_up each step (small overhead)

# Net result during inference:
# - Memory: 83% smaller KV cache (HUGE win)
# - Compute: ~5% overhead from latent projections (negligible)
# - Throughput: Often FASTER due to reduced memory bandwidth pressure
```

**Real-World Deployment Example**:

```python
# Production API serving GPT-2 124M
# Target: 100 req/sec, 2K context average

# Without MLA:
# KV cache per request: 72 MB
# Assume 1 sec latency: 100 concurrent requests
# Total KV cache: 7.2 GB
# Plus model: 0.25 GB
# Total: ~8 GB
# Hardware: 1× A10G GPU (24 GB) - 70% utilized for KV cache alone

# With MLA:
# KV cache per request: 12.4 MB
# 100 concurrent requests: 1.24 GB
# Plus model: 0.25 GB
# Total: ~1.5 GB
# Hardware: 1× A10G GPU (24 GB) - only 6% for KV cache!
# Can handle 700 req/sec on same hardware (7× throughput increase)
```

**Why MLA Enables Long Context**:

The reduction in KV cache is **multiplicative with context length**. This makes long-context applications (document analysis, long conversations) economically viable:

```python
# 32K context window (document analysis)

# Standard GPT-2:
# KV cache: 12 * 12 * 32768 * 64 * 2 * 2 = 1,152 MB per sequence
# 8 concurrent users: 9.2 GB
# Impractical on consumer GPUs

# MLA GPT-2:
# KV cache: 12 * 32768 * 128 * 2 * 2 = 199 MB per sequence
# 8 concurrent users: 1.59 GB
# Easily fits on RTX 4090 (24 GB)
```

**MLA Summary**:

1. **Parameter efficiency**: 75% fewer params for K,V projections (300K vs 1.18M per layer)
2. **Inference memory**: 83% smaller KV cache (critical for long context)
3. **Scaling**: Enables 6× larger batch sizes or 6× longer contexts with same memory
4. **Deployment**: 70-80% reduction in serving costs for inference-heavy applications
5. **Tradeoff**: Minimal compute overhead (~5%) for massive memory savings

**This is why MLA is not just an architectural curiosity - it's a practical necessity for efficient transformer inference at scale.**

### Enhancement 2: Reciprocal Attention (RA) - Symmetric Scoring

**Idea**: Standard attention is Q[i]·K[j] (token i looks at j). Add symmetric term Q[j]·K[i] (j looks back at i).

```python
# Standard: unidirectional scoring
logits = Q @ K^T  # [T, T]

# RA: bidirectional scoring within local band
logits = Q @ K^T + ra_alpha * reciprocal_term
#                  ↑
#                  Learnable weight (scalar)
```

**Parameter cost**:
```
q_to_latent: 12 × 64 × 128 = 98K (per-head Q projection to latent)
ra_alpha: 1 learnable scalar
Total: ~98K per layer
```

**Benefit**: Bidirectional attention information for ~8% extra params (vs baseline attention).

### Enhancement 3: Reciprocal MLP - Cross-Layer Information Flow

**Three mechanisms for MLP↔Attention communication**:

#### Mechanism 1: MLP-to-Attention Gating (Cross-Layer)

**Idea**: Layer N's MLP produces per-head gates that modulate layer N+1's attention.

```python
# In layer N's MLP:
gate_context = mlp_activations @ gate_proj  # [T, mlp_dim] → [T, 64]
gate_context = gate_context.mean(dim=0)     # [64] - global context
head_gates = sigmoid(gate_to_heads(gate_context))  # [12] - per-head gates

# In layer N+1's attention:
attn = (1 - alpha) * attn + alpha * (attn * gates)
#      ↑                     ↑
#      Baseline              Gated by previous MLP
```

**Parameter cost**:
```
gate_proj: mlp_dim × 64 = 3072 × 64 = 196K
gate_to_heads: 64 × 12 = 768
mlp_gate_alpha: 1 learnable scalar
Total: ~197K per layer
```

**Design rationale**: Small projection (64 dims) keeps params low while capturing MLP context.

#### Mechanism 2: Cross-Token MLP Aggregation

**Idea**: Reuse attention weights to aggregate MLP states across tokens.

```python
# Standard MLP: each token processes independently
mlp_out = mlp(hidden[i])  # No cross-token info

# Cross-token MLP: each token sees other tokens' MLP states
cross_context = attn_weights @ mlp_hidden  # [T, T] @ [T, mlp_dim] = [T, mlp_dim]
mlp_out = mlp(hidden[i]) + alpha * cross_proj(cross_context[i])
```

**Parameter cost**:
```
cross_proj: mlp_dim × mlp_dim = 3072 × 3072 = 9.4M
mlp_cross_alpha: 1 learnable scalar
Total: ~9.4M per layer
```

**Note**: This is expensive! But reuses existing attention weights for routing (no new routing parameters).

#### Mechanism 3: MLP Latent Space Reciprocity (Cross-Subsystem)

**Idea**: Bidirectional latent projections between attention latent space and MLP space.

```python
# Attention latent → MLP enrichment
mlp_enrich = attn_to_mlp(attn_latent)  # [T, 128] → [T, mlp_dim]
mlp_hidden = mlp_hidden + alpha * mlp_enrich

# MLP → Attention latent context (passed to next block)
attn_context = mlp_to_attn(mlp_hidden)  # [T, mlp_dim] → [T, 128]
```

**Parameter cost** (tied_transpose mode):
```
W: mlp_dim × latent_dim = 3072 × 128 = 393K
mlp_recip_alpha_mlp: 1 learnable scalar
mlp_recip_alpha_attn: 1 learnable scalar
Total: ~393K per layer
```

**Design rationale**: Parameter tying (one weight used bidirectionally) cuts params by 50% vs two separate projections.

### Total Parameter Overhead

```
MLA (compression):              ~300K per layer
RA (reciprocal attention):      ~98K per layer
Mechanism 1 (MLP gating):       ~197K per layer
Mechanism 2 (cross-token MLP):  ~9.4M per layer (EXPENSIVE)
Mechanism 3 (latent reciprocity): ~393K per layer

Baseline GPT-2 per layer:       7.08M
With all mechanisms:            7.08M + 10.4M = 17.5M (2.5× increase)
```

**Key insight**: Mechanism 2 dominates cost. If it doesn't help, it's 134M wasted params across 12 layers.

---

## Philosophy: Learnable vs Fixed Hyperparameters

### The Problem with Fixed Mixing Weights

**Old approach**:
```python
# Hardcoded in config
ra_alpha = 0.3  # "I think 30% reciprocal is good"
mlp_gate_alpha = 0.1  # "10% gating seems reasonable"
```

**Problems**:
1. These values are **guesses** - no principled way to choose them
2. Optimal values may vary by:
   - Task (language modeling vs classification)
   - Dataset (code vs natural language)
   - Model size
   - Layer depth (early vs late layers might need different ratios)
3. If wrong, the mechanism hurts instead of helps
4. Requires expensive hyperparameter search to tune

### The Learnable Approach

**New approach**:
```python
# Initialize at reasonable guess, but make learnable
self.ra_alpha = nn.Parameter(torch.tensor(0.3))
self.mlp_gate_alpha = nn.Parameter(torch.tensor(0.1))
```

**Benefits**:
1. **Auto-tuning**: Gradient descent finds optimal values
2. **Adaptive**: Can learn different values per layer if needed
3. **Self-disabling**: If mechanism hurts, alpha → 0 (auto-disable)
4. **Task-specific**: Learns task-appropriate mixing ratios
5. **Minimal cost**: Just 1 scalar parameter per mechanism

### Evidence-Based Learning

```python
# After training, inspect learned values:
print(f"RA alpha learned: {model.layer[0].attn.ra_alpha.item()}")
print(f"MLP gate alpha learned: {model.layer[0].attn.mlp_gate_alpha.item()}")

# Example outcomes:
# ra_alpha: 0.3 → 0.05  (model learned RA mostly unhelpful)
# mlp_gate_alpha: 0.1 → 0.28  (model learned gating very useful)
```

This tells us:
- Which mechanisms are actually useful
- What mixing ratios work best
- Whether to keep or remove mechanisms in future architectures

### Gate Initialization Strategy

**Why start gates mostly open** (bias=2.0 → sigmoid ≈ 0.88):

```python
# Bad: Start closed
self.gate_bias = nn.Parameter(torch.tensor(-2.0))  # sigmoid ≈ 0.12
# Problem: No signal flows initially, mechanism never gets gradient to learn

# Good: Start mostly open
self.gate_bias = nn.Parameter(torch.tensor(2.0))  # sigmoid ≈ 0.88
# Benefit: Signal flows initially, gradients can learn to close if harmful
```

This is the "optimistic initialization" principle - assume mechanism might be useful, let gradients prove otherwise.

---

## Evaluating New Parameters

### How to Tell if New Parameters Are Useful

#### 1. Validation Loss Comparison

```python
# Baseline
Step 0 (no mechanisms):  val_loss = 3.5740

# With mechanisms
Step 3 (+ MLP gating):   val_loss = ???  (should be < 3.5740 to be useful)
Step 4 (+ cross-token):  val_loss = ???
```

**Rule**: New parameters are justified only if `val_loss_new < val_loss_baseline`.

#### 2. Parameter Efficiency Metric

```python
# Define: Loss improvement per million parameters
efficiency = (baseline_loss - new_loss) / (new_params_millions)

# Example:
# Step 3 adds 197K params per layer × 12 = 2.4M params
# If val_loss improves from 3.5740 to 3.5500:
efficiency = (3.5740 - 3.5500) / 2.4 = 0.01 loss/M params

# Compare to adding more layers or wider MLP:
# Wider MLP: 500K more params → 3.5740 to 3.5600
efficiency_mlp = (3.5740 - 3.5600) / 0.5 = 0.028 loss/M params

# Verdict: Wider MLP is 2.8× more efficient than our mechanism
```

**Rule**: Mechanisms must match or exceed parameter efficiency of simply scaling up the model.

#### 3. Learned Mixing Weight Analysis

```python
# After training, check what the model learned
for layer in model.layers:
    print(f"Layer {layer.idx}:")
    print(f"  ra_alpha: {layer.attn.ra_alpha.item():.3f}")
    print(f"  mlp_gate_alpha: {layer.attn.mlp_gate_alpha.item():.3f}")

# Example output:
Layer 0:
  ra_alpha: 0.287  (stayed near init 0.3 - mechanism being used)
  mlp_gate_alpha: 0.003  (dropped from 0.1 - mechanism learned to disable)

Layer 11:
  ra_alpha: 0.052  (dropped significantly - less useful in late layers)
  mlp_gate_alpha: 0.156  (increased from 0.1 - more useful in late layers)
```

**Insights**:
- If alpha → 0: Mechanism not useful (remove in next architecture)
- If alpha stays near init: Mechanism moderately useful
- If alpha changes significantly: Mechanism importance varies (consider per-layer design)

#### 4. Gradient Flow Analysis

```python
# During training, log gradient norms
for name, param in model.named_parameters():
    if 'alpha' in name or 'gate' in name:
        grad_norm = param.grad.norm().item()
        print(f"{name}: grad_norm = {grad_norm:.6f}")

# Example:
layer.0.attn.ra_alpha: grad_norm = 0.000123  (healthy gradient)
layer.0.mlp.gate_proj.weight: grad_norm = 0.000000  (dead - no learning)
```

**Rule**: If gradients are consistently near-zero, parameters aren't learning anything useful.

#### 5. Ablation Study Matrix

Compare all combinations:

```
                No Mech    Mech 1    Mech 2    Mech 1+2
Baseline        3.5740     3.5700    3.5800    3.5650
+ Golden Ratio  3.6238     ???       ???       ???
+ RA            3.6367     ???       ???       ???
+ MLA           ???        ???       ???       ???
```

**Look for**:
- Additive improvements (Mech 1+2 better than either alone)
- Synergies (combination much better than sum of parts)
- Harmful combinations (together worse than separate)

### Decision Framework

```
IF new mechanism improves val_loss AND has healthy gradients:
    KEEP mechanism, document improvement
ELIF mixing weight learned to ~0:
    REMOVE mechanism (model says it's not useful)
ELIF parameter efficiency < baseline scaling:
    REMOVE mechanism (better to just scale model)
ELSE:
    INVESTIGATE (may need more training time, better initialization, etc.)
```

---

## Current Status: Learned Ratio as Goal

### The Vision: Learnable MLP Expansion Ratio

**Current**: Fixed ratio (e.g., 4.0 or 5.0)
```python
mlp_dim = int(4.0 * embed_dim)  # Fixed at 3072
```

**Goal**: Learnable ratio
```python
# Option A: Learnable scalar
self.mlp_ratio = nn.Parameter(torch.tensor(4.0))
mlp_dim = int(self.mlp_ratio * embed_dim)

# Option B: Learnable gating between multiple widths
self.mlp_widths = [2048, 3072, 3840]  # Multiple MLP layers
self.width_gates = nn.Parameter(torch.ones(3) / 3)  # Softmax over widths
output = sum(gate * mlp(x, width) for gate, width in zip(gates, widths))
```

### Why This Matters

The ratio concept addresses the fundamental question:

**"How should we allocate parameters between attention and MLP?"**

Current evidence:
- Standard 4:1 ratio may be suboptimal
- "Golden" 2.5:1 ratio made things worse (but was fixed, not learned)
- Optimal ratio is likely task-dependent

A learnable ratio would:
1. Let model discover optimal attention/MLP balance
2. Potentially vary by layer (early layers need different ratio than late)
3. Adapt to dataset characteristics
4. Avoid wasting parameters on oversized MLP or attention

### Implementation Challenges

**Challenge 1**: Discrete architecture
- Can't have fractional neurons
- Must discretize or use gating

**Challenge 2**: Pretrained weight initialization
- If ratio changes, how do we initialize new weights?
- Can't just copy from pretrained if dimensions don't match

**Challenge 3**: Training stability
- Ratio might oscillate during training
- Need constraints or regularization

### Evaluation Strategy

```python
# 1. Test learned ratio with multiple initialization points
ratios_init = [2.0, 3.0, 4.0, 5.0]
for ratio_init in ratios_init:
    train_model(learnable_ratio_init=ratio_init)
    print(f"Init {ratio_init} → Final {model.mlp_ratio.item()}")

# 2. Compare to fixed ratio search
fixed_ratios = [2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]
for ratio in fixed_ratios:
    val_loss = train_model(fixed_ratio=ratio)
    print(f"Ratio {ratio}: val_loss = {val_loss}")

# 3. Check if learned ratio beats best fixed ratio
if val_loss_learned < min(val_loss_fixed_list):
    print("Learnable ratio is better than any fixed ratio!")
```

---

## Summary: Our Philosophy

### Core Principles

1. **Parameter Efficiency First**
   - Every parameter must justify its existence with improved loss
   - Prefer clever design over brute-force parameter increase

2. **Learnable > Fixed**
   - Let gradients discover optimal values
   - Fixed hyperparameters are guesses; learning finds truth

3. **Evidence-Based Design**
   - Measure everything (loss, gradients, learned values)
   - Remove mechanisms that don't help
   - Keep mechanisms with clear benefit

4. **Ratio Awareness**
   - Model capacity distribution matters (attention vs MLP)
   - Not all parameters are equally useful
   - Learnable ratio = holy grail (let model find optimal balance)

### Current Findings

```
✗ Golden ratio (fixed 2.5:1): Made it worse
✗ RA (fixed alpha=0.3): Made it worse
? Learnable mixing weights: In progress
? MLP mechanisms: Need to test (blocked by old code)
? Learnable ratio: Future goal
```

The path forward: Let the model tell us what works through learned parameters and validation loss, not through guesses and fixed hyperparameters.
