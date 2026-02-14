# BPA v19 Branch Tree: Outcome-Based Decision

## Observed Outcome: Branch D — None Match S2

```
v19 Signal Bakeoff
├── A) Activation-noise wins?
│   └── NO. Best A6 rho=0.093, needs k=24. No compression.
├── B) Cheap proxy wins?
│   └── PARTIAL. C3_residual_ratio rho=0.394, top6=4/6.
│       But k=2 insufficient at L>=16K (even oracle fails).
│       At k=6, any reasonable signal converges to S2.
├── C) Adam transform wins?
│   └── NO. E1 rho=0.17, same as v18. Transforms monotone-invariant.
└── D) None match S2? ← OBSERVED
    └── Budget is the bottleneck, not signal quality.
```

## v20 Action Plan (from Branch D)

### Immediate (low effort)
- **Keep S2 manual** for Qwen2.5-0.5B production deployment.
- **Use C3_residual_ratio** as a cheap screening signal when
  applying mixed-precision to new models. It correctly flags
  layer 0 and has the best oracle correlation (rho=0.394) at
  near-zero compute cost.

### Medium-term (v20 scope)
- **Hybrid signal regressor**: Train a small linear model over
  the top-3 signals (C3_residual_ratio, D3_K_condnum,
  C1_norm_QK) to predict oracle sensitivity. All three have
  4/6 or 3/6 top-6 overlap with complementary layer emphasis.
- **Transfer validation**: Test whether C3-derived schedules
  transfer to Qwen2.5-1.5B and Llama-3.1-8B. If the same
  signal ranks well across models, it can serve as a universal
  first-pass allocator.
- **INT6 exploration**: If INT6 quantization is feasible, it
  could reduce the budget gap between INT4 (too lossy) and
  INT8 (too conservative), potentially allowing k=2 schedules
  to pass at long context.

### Long-term
- **Per-head mixed precision**: v19 Phase 4 (skipped) explored
  protecting only sensitive heads within a layer. This could
  reduce effective k while maintaining quality, but requires
  KV storage to support per-head bitwidth.
- **Learned compression operators**: v13/v14 showed random
  projections fail. A learned compressor (trained offline)
  could achieve better ratios than binary INT4/INT8 allocation.
- **Combine with rope_complex**: v15 showed rope_complex
  achieves FULL PASS at ratio=0.55 by compressing K magnitude.
  Stacking rope_complex + INT4 on remaining dimensions could
  break the 0.33 wall without requiring k=6 INT8 layers.

## Why Branch D Is Informative

The v19 bakeoff is a negative result, but a useful one:

1. **Layer 0 dominance**: All signals detect layer 0 as most
   sensitive. The real challenge is ranking layers 1-23, where
   sensitivity varies by only 0.03-0.77% and signals disagree.

2. **Cumulative degradation**: INT4 error accumulates across
   layers. Protecting 2 layers leaves 22 contributing small
   errors that sum to >3% at 32K. This is a fundamental
   property of per-layer quantization, not a signal failure.

3. **Transform invariance**: Monotone transforms (sqrt, root4,
   log1p) have zero effect on rankings for all 19 signals.
   This means heavy-tail compression (which helped in some
   contexts) is irrelevant here because the 24-layer budget
   is too small for outlier effects.

4. **Activation norms beat curvature**: Forward-only C3
   (rho=0.394) outperforms backward-requiring E1 (rho=0.17)
   and E3 (rho=0.088). This is good news for deployment:
   the best signal is the cheapest to compute.
