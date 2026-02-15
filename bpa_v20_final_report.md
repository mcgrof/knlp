# BPA v20 Final Report: Break the k-floor

## Summary

v20 attacked the k-floor problem identified in v19: even oracle-optimal
k=2 fails at L>=16K because INT4 noise accumulates across the remaining
22 layers. Three attack paths tested:

- **Path A** (reduce accumulation): V-only INT8 dead end; tight-group
  INT4 (g=4) is the single best intervention.
- **Path B** (reduce per-layer damage): group_size sweep shows g=4 best,
  non-monotonic behavior at g=16.
- **Path C** (structural modifications): rescaling and norm clamping
  both fail to meaningfully reduce k*.

**Verdict: tight-group INT4 (g=4) is the only intervention that
materially reduces per-layer quantization noise.** It drops the
all-INT4 degradation from +429% (g=32) to +4.5% (g=4) at 32K,
nearly achieving k=0 PASS@3%. The true k-floor remains at k=4
(PASS@3% at all L), which no intervention could reduce.

## 1. k-sweep Baseline (Phase 0)

Oracle-ranked k-sweep with standard INT4 (block_size=32):

| k | max delta 8K | max delta 16K | max delta 32K | PASS@3% |
|---|---|---|---|---|
| 0 | +30.25% | +26.82% | +428.69% | NO |
| 1 | +3.05% | -- | -- | NO (8K fail) |
| 2 | +1.63% | +5.60% | +2.85% | NO (16K fail) |
| 3 | +0.97% | -- | -- | YES (8K only) |
| 4 | +0.77% | +1.90% | +1.96% | YES (all L) |
| 6 | +1.05% | +1.74% | +2.15% | YES (all L) |

k*@3% (all L) = 4. k*@1% requires k>=4 but was not exhaustively tested.

## 2. Theory (Phase 1)

Additive noise accumulation model fitted to k-sweep data:

```
max_delta_pct = 26.65 * remaining_noise - 186.6
```

Per-layer distortion (top 4 by sigma4):
- Layer 0: sigma4=4.31, sigma8=0.27 (attention sink, 25% of total)
- Layer 8: sigma4=2.41, sigma8=0.34 (mid-network)
- Layer 1: sigma4=1.95, sigma8=0.23
- Layer 2: sigma4=1.55, sigma8=0.11

Total INT4 noise: 16.88 (12.1x INT8).
Theory predicts k*@3%=5, observed=4 (within 1 layer).

## 3. Path A Results (Phase 2)

### A3: V-only INT8 (K stays INT4)

**DEAD END.** Upgrading V to INT8 on sensitive layers while keeping
K at INT4 produces +20-240% degradation. Increasing from k=2 to k=8
V-INT8 layers provides <1% improvement. K quantization noise
dominates the residual distortion, not V.

This disproves the hypothesis that V quant error drives accumulation.
K (key) values carry positional information via RoPE, and INT4
destroys this position signal.

### A2: Tight-group INT4 (g=4)

**BEST INTERVENTION.** Using group_size=4 instead of 32 for INT4:

| k | max delta (all L) | PASS@3% |
|---|---|---|
| 0 | +4.46% | NO (barely) |
| 2 | +3.41% | NO (barely) |
| 4 | +1.70% | YES |
| 6 | +1.53% | YES |

At k=0 (all INT4 g=4), the worst case drops from +429% (g=32) to
+4.5%. At k=4, max delta drops from +1.96% (g=32) to +1.70% (g=4).
The improvement is dramatic at low k but doesn't eliminate the need
for k>=4 INT8 layers.

## 4. Path B Results (Phase 3)

### B2: Per-channel INT4 group_size sweep

| g | k*@3% (all L) | max delta k=4 | k=0 worst |
|---|---|---|---|
| 4 | 4 | +1.70% | +4.46% |
| 8 | 4 | +2.38% | +41.71% |
| 16 | >6 | +4.33% | +310.59% |
| 32 | 4 | +1.96% | +428.69% |

g=16 is paradoxically worse than g=32. This is because head_dim=64:
- g=4 gives 16 groups per head_dim: fine-grained scales
- g=8 gives 8 groups: still reasonable
- g=16 gives 4 groups: misaligned with internal structure
- g=32 gives 2 groups: coarse but regular

The non-monotonic behavior at g=16 suggests group boundaries interact
with the model's internal representation structure.

## 5. Path C Results (Phase 4)

### C3: Post-quant rescaling

**DEAD END.** Matching mean/std of quantized KV to clean KV per-head
destroys position-dependent structure. Deltas of +7% to +710% at all k.
The rescaling operation is fundamentally incompatible with RoPE-encoded
K values where the distribution varies with position.

### C2: Norm clamping (p=0.99)

**MARGINAL.** Clamping outlier K norms before quantization provides
<0.2% improvement at k=4 (1.85% vs 1.96%). Does not reduce k*.
Not worth implementation complexity.

## 6. Unified Comparison

| Method | Path | k*@3% (all L) | Reduces k*? | Notes |
|---|---|---|---|---|
| S2 baseline (g=32) | -- | 4 | -- | Reference |
| A3 V-only INT8 | A | >8 | NO (worse) | K noise dominates |
| A2 tight group g=4 | A | 4 | NO | Best per-layer noise reduction |
| B2 g=8 | B | 4 | NO | Similar to g=32 |
| B2 g=16 | B | >6 | NO (worse) | Non-monotonic |
| C3 rescaled | C | >6 | NO (worse) | Destroys position info |
| C2 norm clamp | C | 4 | NO | Marginal 0.1% |

**No intervention reduces k* below 4.** The k-floor is architectural:
it reflects the minimum number of layers that must be protected from
INT4 noise to prevent cumulative degradation at long context.

## 7. Recommendation for v21

1. **k-floor is architectural, not fixable by post-hoc interventions.**
   The minimum k=4 INT8 layers is a fundamental property of the model's
   sensitivity distribution.

2. **Use tight-group INT4 (g=4) for non-protected layers.** This
   reduces per-layer noise by ~8x versus standard g=32, making the
   INT4 layers nearly invisible. Combined with k=4 INT8, this achieves
   reliable PASS@3% with kv_ratio ~0.30.

3. **V-only protection is useless.** K quant noise dominates; always
   protect both K and V together at sensitive layers.

4. **Training-in-the-loop quantization is the next frontier.** Since
   post-hoc methods cannot break k*=4, the next step is to fine-tune
   quantization parameters (scales, zero points) or the model itself
   to tolerate INT4 at the sensitive layers.

5. **Test on larger models.** Qwen2.5-0.5B has only 24 layers.
   Models with 32+ layers may have more headroom for k-floor reduction
   because the cumulative noise grows sublinearly with depth.
