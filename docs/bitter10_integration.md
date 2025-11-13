# Bitter10 (Coherence-Based Pruning) Integration Plan

## Status: NOT YET INTEGRATED - In Test Queue

## ⚠️ CRITICAL DESIGN VIOLATION: Memory Overhead

**Bitter10 fundamentally violates AdamWPrune's core design principle** of leveraging existing Adam states without introducing new variables.

### Memory Comparison (GPT-2 124M example)

**Bitter0-9** (follows design goals):
```
Parameter memory:        496MB (1x)
Adam exp_avg (m):        496MB (1x)
Adam exp_avg_sq (v):     496MB (1x)
──────────────────────────────────
Total:                   1.49GB (3x)
```
✅ Zero new memory - scores computed on-the-fly at pruning time

**Bitter10** (violates design goals):
```
Parameter memory:        496MB (1x)
Adam exp_avg (m):        496MB (1x)
Adam exp_avg_sq (v):     496MB (1x)

CoherenceTracker NEW overhead:
  - prev_w:              496MB (1x)
  - ema_dw:              496MB (1x)
  - ema_dw2:             496MB (1x)
  - scalars:             negligible
──────────────────────────────────
Total:                   2.98GB (6x)
```
❌ **Doubles optimizer memory** - adds 3x parameter memory on top of Adam's 2x

### Why This Violates Design Goals

- **Bitter0-9**: Stateless scoring using only existing Adam states (exp_avg, exp_avg_sq)
- **Bitter10**: Stateful tracking requiring history storage across ALL training steps

**Impact**: For large models (GPT-2 1.5B, GPT-3), bitter10's memory overhead may exceed available GPU memory.

**Recommendation**: Use bitter10 only for research/experimentation on smaller models where memory is not constrained. For production pruning, use bitter0-9 variants which uphold the memory-efficient design.

## Overview

Bitter10 extends bitter3 (|w| * sqrt(|m|)) with temporal coherence tracking:

```
Score = alpha * |w| * sqrt(|m|)
        * (1 - flip_rate_ema)      # fewer sign flips = better
        * cos_dir_ema               # update direction consistency
        * cos_mw                    # momentum-weight alignment
        / (1 + jitter_ema)          # lower variance = better
```

## Key Innovation

Unlike bitter0-9 which score parameters at pruning time only, bitter10 **tracks temporal coherence** across training steps:

1. **Sign Stability**: EMA of parameter update sign flips
2. **Directional Consistency**: Cosine similarity of Δw_t and Δw_{t-1}
3. **Momentum Alignment**: Cosine similarity of Adam m and current w
4. **Jitter Penalty**: Variance of update magnitudes over time

## Architecture Difference from Bitter0-9

**Bitter0-9**: Stateless scoring at pruning time
```python
# Just use current optimizer state
importance = compute_importance(weight, optimizer.state)
```

**Bitter10**: Stateful tracking throughout training
```python
# Must track every step
for epoch in epochs:
    optimizer.step()
    for p in model.parameters():
        tracker.update(p)  # ← NEW REQUIREMENT

    if should_prune():
        scores = bitter10_scores(params, optimizer, tracker)
```

## Integration Requirements

### 1. Add CoherenceTracker to adamprune_state

In `create_optimizer()` when initializing AdamWPrune:

```python
if variant == "bitter10":
    from lib.bitter10_coherence import CoherenceTracker, Bitter10Config
    adamprune_state["coherence_tracker"] = CoherenceTracker(Bitter10Config())
    adamprune_state["bitter10_config"] = Bitter10Config()
```

### 2. Update Tracker After Every Step

This is the **critical challenge** - requires modifying all training loops:

```python
# In gpt2/train.py, lenet5/train.py, resnet/train.py
optimizer.step()

# NEW: Update coherence tracker for bitter10
if hasattr(optimizer, 'adamprune_state') and \
   optimizer.adamprune_state.get("variant") == "bitter10":
    tracker = optimizer.adamprune_state["coherence_tracker"]
    for p in model.parameters():
        tracker.update(p)
```

### 3. Use Bitter10 Scorer at Pruning Time

In `update_adamprune_masks()`:

```python
elif variant == "bitter10":
    from lib.bitter10_coherence import bitter10_coherence_scores
    tracker = adamprune_state["coherence_tracker"]
    cfg = adamprune_state["bitter10_config"]

    # Collect all prunable parameters
    params = [module.weight for name, module in model.named_modules()
              if isinstance(module, (nn.Linear, nn.Conv2d))]

    # Get global scores
    global_scores = bitter10_coherence_scores(params, optimizer, tracker, cfg)

    # Compute threshold for target sparsity
    k = int(global_scores.numel() * current_sparsity)
    threshold = torch.topk(global_scores, k, largest=False).values.max()

    # Apply per-layer using same scorer
    for name, module in model.named_modules():
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            layer_scores = bitter10_coherence_scores([module.weight],
                                                     optimizer, tracker, cfg)
            mask = (layer_scores > threshold).reshape_as(module.weight)
            module.adamprune_mask.copy_(mask)
```

### 4. Add to Kconfig

```kconfig
config GPT2_ADAMWPRUNE_VARIANT_BITTER10
    bool "Bitter10: Coherence-based (EXPERIMENTAL - requires tracker)"
    depends on OPTIMIZER_ADAMWPRUNE
    help
      Temporal coherence tracking: tracks sign stability, directional
      consistency, momentum alignment, and jitter across training.

      WARNING: Requires CoherenceTracker.update() after every step.
      Not yet integrated into standard training loops.
```

### 5. Update Comments in optimizers.py

```python
# Line 346-348
),  # bitter0=original, bitter1=magnitude, bitter2=scale-aware,
   # bitter3=gradient-magnitude, bitter4=layer-adaptive,
   # bitter5=movement-to-zero, bitter6=coherence-weighted,
   # bitter7=second-moment, bitter8=bias-corrected, bitter9=hybrid,
   # bitter10=temporal-coherence (requires tracker integration)
```

```python
# Line 584
if variant in ["bitter3", "bitter4", "bitter5", "bitter6",
               "bitter7", "bitter8", "bitter9", "bitter10"]:
    # Cubic schedule
    progress = progress**3
```

## Testing Plan (When Ready)

1. **Unit Test**: `lib/bitter10_coherence.py` has built-in smoke test
   ```bash
   python lib/bitter10_coherence.py
   ```

2. **Integration Test**: Add to LeNet-5 first (simplest training loop)
   ```bash
   make defconfig-lenet5-bitter10-test
   make
   ```

3. **Full Evaluation**: Compare against bitter0-9 on GPT-2 FineWebEdu
   - Baseline: bitter3 = 43.11 perplexity @ 13k iters
   - Target: bitter10 ≤ 45 perplexity with better pruning dynamics

## Expected Benefits

1. **Stability**: Penalizes parameters with erratic updates
2. **Coherence**: Rewards parameters moving consistently toward minima
3. **Alignment**: Keeps parameters that match optimizer's learned direction
4. **Robustness**: Combines multiple signals vs single-signal bitter variants

## Risks

1. **Overhead**: Tracker.update() adds ~5-10% per-step cost
2. **Memory**: Stores prev_w and EMAs for all parameters (~2x state overhead)
3. **Complexity**: Harder to debug than stateless bitter0-9

## Memory-Efficient Alternatives (RECOMMENDED)

Given the memory violation, **three alternative implementations** are provided in `lib/bitter10_variants.py` that capture temporal coherence aspects while respecting the zero-new-memory design goal:

### bitter10z (Zero-New-Memory) - PRODUCTION RECOMMENDED ✅

**Score**: `(|w| * sqrt(|m|)) * (|m| / sqrt(v))`

**Memory overhead**: **ZERO** - Pure stateless scoring using only existing Adam states

**Memory comparison (GPT-2 124M)**:
```
Parameters:          496MB (1x)
Adam states:         992MB (2x)
bitter10z overhead:    0MB (0x)
─────────────────────────────
Total:              1.49GB (3x)
```

**Integration**: Drop-in replacement for bitter0-9 (no training loop modifications)

**Signal**: Stability via `|m|/sqrt(v)` ratio (low-noise, consistent gradient directions)

**Pros**:
- 100% compliant with design goals
- No tracking, no history, no overhead
- Same integration complexity as bitter0-9
- Better than bitter3 by adding stability signal

**Cons**:
- No true temporal coherence (no sign flip tracking)
- Just a smarter combination of existing Adam states

**Verdict**: **Best for production** - clean, memory-efficient, stateless.

---

### bitter10s (Sketch Tracker) - RESEARCH RECOMMENDED ✅

**Memory overhead**: **~156KB total** for GPT-2 124M (negligible)

**Memory comparison (GPT-2 124M)**:
```
Per tensor tracking (k=64 samples):
  - idx (sample indices):    512 bytes
  - prev_m (prev momentum):  256 bytes
  - ema_cos/flip/jitter:      12 bytes
Total per tensor:           ~780 bytes

GPT-2 124M (~200 weight tensors):
  Total overhead:           156KB
```

**Memory breakdown**:
```
Parameters:          496MB (1x)
Adam states:         992MB (2x)
bitter10s sketch:   0.15MB (0.0003x)
─────────────────────────────
Total:              1.49GB (3.0003x)
```

**Integration**: Requires `tracker.update(p, m, v)` after every `optimizer.step()`

**Signal**: Layer-level temporal coherence (sign flips, directional consistency, jitter)

**Pros**:
- Full temporal coherence tracking with negligible memory (0.01% overhead)
- Statistically sound sampling with k=64
- Layer-level modifiers (not per-parameter overhead)
- Cheap updates (only processes k samples per tensor)

**Cons**:
- Requires training loop modifications (same as bitter10)
- Sampling approximation (may miss outliers)

**Verdict**: **Best compromise** - temporal coherence without memory explosion.

---

### bitter10c (Checkpoint Δw) - LARGE MODEL OPTION ⚠️

**Memory overhead**: **0 GPU, 496MB CPU/disk** (1× params)

**Memory comparison (GPT-2 124M)**:
```
GPU memory:
  Parameters:          496MB (1x)
  Adam states:         992MB (2x)
  bitter10c overhead:    0MB (0x)
  ─────────────────────────────
  Total GPU:          1.49GB (3x)

CPU/disk storage:
  Snapshot:            496MB (taken every N steps)
```

**Integration**: Requires periodic `ckpt.take_snapshot(params)` every N steps (e.g., N=1000)

**Signal**: Long-horizon Δw = w_now - w_snapshot, combined with cos(m, Δw) and jitter

**Pros**:
- Zero GPU memory overhead
- No per-step tracking (infrequent snapshots only)
- Can offload to disk with torch.save() for huge models
- Long-horizon temporal signal

**Cons**:
- Coarser temporal resolution (depends on snapshot frequency)
- CPU-GPU copy overhead when computing scores
- Requires snapshot management logic

**Verdict**: **Use only for very large models** (>10B params) where even 156KB matters.

---

## Memory-Efficient Variant Comparison

| Variant | GPU Memory | Temporal Signal | Training Loop Mod | Design Compliance | Recommendation |
|---------|------------|-----------------|-------------------|-------------------|----------------|
| **bitter10z** | 0 bytes | ❌ None (stability from m/v) | ❌ None | ✅ 100% | **Production** |
| **bitter10s** | ~156KB | ✅ Full (sampled) | ✅ Every step | ✅ 99.99% | **Research** |
| **bitter10c** | 0 bytes | ⚠️ Coarse (snapshot) | ⚠️ Every N steps | ✅ GPU 100% | Large models only |
| **bitter10 (original)** | 1.49GB | ✅ Full | ✅ Every step | ❌ 0% | ❌ **DO NOT USE** |

## Final Recommendation

**Integrate bitter10z as the primary variant** (rename to bitter10) and optionally add bitter10s as an experimental variant (bitter11).

- **bitter10z** perfectly aligns with AdamWPrune's design philosophy: leverage Adam states, zero new memory
- **bitter10s** provides advanced temporal coherence for research at negligible cost (0.01% overhead)
- **bitter10 (original)** should be deprecated or marked research-only due to memory violation

## References

- Memory-efficient implementations: `lib/bitter10_variants.py`
- Original (memory-heavy) implementation: `lib/bitter10_coherence.py`
- Original idea: Extends bitter3 with temporal features
- Related: bitter6 (coherence-weighted) is simpler, stateless version

---

**Status**: Memory-efficient variants ready (bitter10z/bitter10s), original bitter10 deprecated.
**Priority**: Low (in queue after RA+MLA experiments)
**Effort**: Low (bitter10z is drop-in), Medium (bitter10s requires training loop mods)
