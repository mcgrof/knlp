# BPA v20 Branch Tree: Outcome-Based Decision

## Observed Outcome: k-floor is architectural (k*=4 unbreakable)

```
v20 Break the k-floor
├── Path A: Reduce accumulation
│   ├── A3 V-only INT8? → DEAD END (K noise dominates)
│   └── A2 Tight group g=4? → BEST INTERVENTION but k*=4 unchanged
├── Path B: Reduce per-layer damage
│   ├── B2 g=8? → Similar to g=32
│   ├── B2 g=16? → Paradoxically worse (non-monotonic)
│   └── B2 g=32? → Baseline k*=4
├── Path C: Structural modifications
│   ├── C3 Rescaling? → DEAD END (destroys position info)
│   └── C2 Norm clamp? → Marginal (0.1% improvement)
└── Conclusion: k*=4 is architectural floor
```

## v21 Action Plan

### If pursuing tighter compression (kv_ratio < 0.30):
- **Combine g=4 + k=2-4 INT8**: Use tight-group INT4 (g=4) for
  non-protected layers instead of standard g=32. This reduces
  the INT4 noise floor, potentially allowing k=2-3 to pass.
- **Measure kv_ratio accurately**: g=4 requires 2x more scale
  factors than g=32. Compute actual byte savings including
  scale overhead.

### If pursuing training-in-the-loop:
- **Quantization-aware fine-tuning**: Freeze model, learn only
  INT4 scales and zero-points per layer on calibration data.
  Target: reduce sigma4 at the 4 most sensitive layers (0,8,1,2)
  enough that k=2 passes at all L.
- **Metric**: reduce layer 0 sigma4 from 4.31 to <1.0.

### If transferring to larger models:
- **Test Qwen2.5-1.5B/7B**: More layers means each layer
  contributes proportionally less to total noise. k* may scale
  sublinearly with depth.
- **Use C3_residual_ratio** from v19 as first-pass signal for
  identifying sensitive layers in new models.

## Why k*=4 Is Architectural

1. **Layer 0 dominance**: Layer 0 (attention sink) accounts for 25%
   of total INT4 noise. It MUST be protected (k>=1).

2. **Top-3 concentration**: Layers 0, 8, 1 account for 52% of total
   noise. Protecting them (k=3) handles the majority of distortion.

3. **Long tail**: The remaining 21 layers each contribute ~0.27-0.35
   noise units. Their cumulative effect requires k>=4 at long context
   (>16K) where errors compound over more attention positions.

4. **Position sensitivity**: K values encode position via RoPE. INT4
   quantization corrupts the phase of the rotation, which cannot be
   corrected by post-hoc rescaling or norm clamping.
