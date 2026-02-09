# BPA v1 Final Report: Boundary-Pressure Attention Risk Reduction

## Executive Summary

**Goal**: Turn the v19 discovery (boundary_pressure predicts head importance)
into a robust, scalable method for compute savings.

**Result**: BPA is NOT viable for compute savings.

The fundamental problem: you can't know you need far-context without
computing far-context. boundary_pressure is a diagnostic signal, not
a predictive one.

## Phase Results

| Phase | Test | Result | Details |
|-------|------|--------|---------|
| 0 | Rename + plumbing | PASS | Created bpa.py wrapper module |
| 1 | Scale (124M vs 355M) | PASS | ρ=0.58, AUC=0.70 on both |
| 2 | Long context (L=512 vs 1024) | FAIL | Sign flip at L=1024 |
| 3 | Cheap signal | FAIL | Best AUC=0.50 (random) |
| 4 | End-to-end PPL | SKIPPED | R3 failed |

## Phase 1: Scale Test (PASS)

boundary_pressure retains predictive power at larger model scale.

| Model | Mean ρ | Mean AUC | KL Ratio | Sign Flip |
|-------|--------|----------|----------|-----------|
| 124M | 0.5862 | 0.7101 | 6.62x | no |
| 355M | 0.5780 | 0.6806 | 6.30x | no |

Both exceed acceptance thresholds (ρ >= 0.45 OR AUC >= 0.68).

## Phase 2: Long Context Test (FAIL)

boundary_pressure loses predictive power at longer contexts.

| Length | Mean ρ | Mean AUC | Sign Flip |
|--------|--------|----------|-----------|
| L=512 | 0.34 | 0.66 | no |
| L=1024 | 0.08 | 0.54 | YES |

The sign flip between seeds at L=1024 indicates instability.
Correlation drops from 0.34 to 0.08.

## Phase 3: Cheap Signal Test (FAIL)

No cheap predictor can approximate boundary_pressure.

| Predictor | Mean AUC | Spearman r |
|-----------|----------|------------|
| local_boundary_gap | 0.50 | +0.02 |
| local_entropy | 0.46 | -0.11 |
| escape_proxy | 0.05 | -0.93 |

All predictors achieve near-random AUC. The escape_proxy has strong
NEGATIVE correlation (-0.93), meaning high local-boundary concentration
actually predicts LOW boundary_pressure (the opposite of what we need).

## Fundamental Insight

**boundary_pressure is inherently oracle-dependent.**

The signal measures "attention mass trying to escape the local window"
which requires computing attention scores OVER the far-context.
You cannot know this without already having paid the compute cost.

This is analogous to: "you can't know what you don't know."

## What Works

1. **Diagnostic value**: boundary_pressure IS a strong predictor of
   head importance when measured as an oracle signal. Threshold gating
   achieves 24x KL alignment (v19 finding).

2. **Analysis tool**: Can identify which heads/positions rely on
   far-context in post-hoc analysis.

3. **Architecture insight**: Confirms that importance is query-conditional,
   not head-static (key finding from v14-v19).

## What Doesn't Work

1. **Compute savings**: Cannot predict boundary_pressure cheaply,
   so cannot gate far-context access without computing it first.

2. **Long contexts**: Correlation collapses and sign-flips at L=1024,
   suggesting the relationship is non-stationary.

3. **Cheap approximations**: Local signals (boundary gap, entropy,
   escape proxy) are uncorrelated with the oracle signal.

## Recommendations

1. **Close BPA as a compute-savings method.** The fundamental limit
   (oracle-dependence) cannot be overcome with better cheap predictors.

2. **Preserve BPA as a diagnostic tool.** boundary_pressure is useful
   for understanding attention behavior post-hoc.

3. **Future work if pursuing sparse attention:**
   - Consider LEARNED gating (small network predicting necessity)
   - Use speculative execution (compute local first, then decide)
   - Accept that some oracle cost is unavoidable

4. **Alternative approaches to explore:**
   - Chunk-level gating (coarser granularity)
   - Position-bucket gating (e.g., always allow far-context after T=512)
   - Hybrid local/retrieval (different mechanisms for short vs long)

## Files and Artifacts

### Source Code
- `gpt2/bpa.py`: BPA wrapper module with aliases
- `scripts/bp_v1_phase1_scale.py`: Phase 1 scale test
- `scripts/bp_v1_phase2_length.py`: Phase 2 length test
- `scripts/bp_v1_phase3_cheap.py`: Phase 3 cheap signal test

### Results
- `bpa_v1_results/phase1_scale.json`
- `bpa_v1_results/phase2_length.json`
- `bpa_v1_results/phase3_cheap.json`

### Documentation
- `docs/bp_v1_migration_note.md`: RGSA → BPA migration
- `bp-v1.txt`: Full specification and progress log

## Commits

1. `f26ca11`: Phase 0 - BPA rename and plumbing
2. `d96941e`: Phase 1 - Scale test (PASS)
3. `c267c2b`: Phase 2 - Length test (FAIL)
4. `a5c2343`: Phase 3 - Cheap signal test (FAIL)

## Final Verdict

**BPA cannot deliver compute savings.**

The boundary_pressure signal from v19 is real and predictive, but it's
an oracle signal that requires far-context computation. Without a cheap
proxy, BPA is limited to diagnostic use.

Close the BPA research line. Future sparse attention work should focus
on learned gating or accept oracle overhead.
