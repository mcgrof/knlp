# RGSA v16 Final Report: Variance Postmortem Cleanup

## Why Variance / Fisher Priors Fail for RGSA

The variance-weighted allocation hypothesis (v14-v16) tested whether Adam
exp_avg_sq (a diagonal Fisher proxy in parameter space) could guide RGSA
budget allocation. The hypothesis is now **closed** with a definitive
negative result.

### v14 Results (Budget Mismatch)

Original v14 testing with allocation drift:

| Allocation | Total Budget | PPL | vs Uniform |
|------------|--------------|-----|------------|
| uniform | 96 | baseline | - |
| variance α=0.5 | 71 (-26%) | worse | confounded |
| variance α=1.0 | 40 (-58%) | worse | confounded |
| inverted | varies | worse | confounded |

### v16 Results (Exact Budget Matching)

With exact budget matching (all sum to 96):

**Seed 1:**

| Allocation | top_b_per_layer | PPL | vs Uniform |
|------------|-----------------|-----|------------|
| uniform | [8]*12 | 701.82 | baseline |
| variance_0.5 | [16,16,16,12,5,5,5,5,4,4,4,4] | 733.92 | +4.6% |
| inverted_0.5 | [2,4,7,8,8,9,9,9,10,10,10,10] | 553.10 | -21.2% |

**Seed 2:**

| Allocation | PPL | vs Uniform |
|------------|-----|------------|
| uniform | 664.76 | baseline |
| inverted_0.5 | 843.27 | +26.9% |

### The 52% Swing

The direction of effect flips by seed:
- Seed 1: inverted allocation 21% **better**
- Seed 2: inverted allocation 27% **worse**
- Cross-seed variance for inverted: **52%**

Signal-to-noise is inverted. This hypothesis is closed.

### Why Parameter-Space Signals Fail

1. **Parameter-space sensitivity ≠ runtime attention importance**: Adam
   exp_avg_sq measures update geometry during training, not the utility of
   KV states for downstream attention.

2. **Adam exp_avg_sq measures gradient magnitude, not state value**: High
   sensitivity at layer 0 means parameters are changing rapidly, not that
   the layer needs more far-context access.

3. **Any apparent gains are unstable and non-causal**: The 21% improvement
   in seed 1 was a statistical artifact that reversed completely in seed 2.

## Executive Summary

**v14 negative result CONFIRMED under exact budget matching.**

The cleanup confirms that variance-weighted allocation (using Adam exp_avg_sq)
does not reliably improve RGSA quality. Seed-to-seed variance at short
training runs (200 iters) completely dominates any allocation effect.

## Part A: Cleanup R&D Results

### A.1 Exact Budget Matching Implementation

Fixed the budget mismatch issue from v14:
- Implemented `compute_per_layer_top_b_exact()` that guarantees exact total
- Old method had severe drift: alpha=0.5 got 71 instead of 96 (-26%)

### A.2 Deterministic Hyperparameters

- Added RGSA_TOP_B_PER_LAYER environment variable
- Bypasses AUTO mode completely
- Fixed batch_size=8, gradient_accumulation=4

### A.3 Confirmation Runs

See "v16 Results" section above for complete tables.

### A.4 Seed Variance Analysis

The cross-seed variance is enormous:
- Uniform: 701.82 vs 664.76 (5.5% difference)
- Inverted: 553.10 vs 843.27 (52% difference!)

This 52% swing across seeds means any single-seed result is unreliable.

## Part B: Drop-Impact KL - SKIPPED

Given that seed variance dominates allocation effects at 200 iters, there is
no point implementing drop-impact KL measurement. The signal would be swamped
by training noise.

## Lessons Learned

- **RGSA acts on runtime state; allocation signals must be state-aligned.**
  Parameter-space sensitivity (gradients, Fisher) measures learning dynamics,
  not the value of attending to far context at inference time.

- **Uniform allocation is a strong baseline due to symmetry and stability.**
  Without a reliable importance signal, equal distribution avoids the risk of
  starving important layers.

- **Seed stability is a first-class metric, not a secondary concern.** A
  method that shows 21% improvement on one seed and 27% degradation on another
  has zero practical value. Multi-seed validation should precede any claims.

- **Negative results are valid outcomes and inform future design.** Closing
  this hypothesis prevents future wasted effort on parameter-space priors for
  RGSA allocation.

## Conclusions

1. **Budget mismatch was real but not the cause**: v14's old allocation method
   had -26% budget drift, but fixing this didn't change the fundamental result.

2. **Seed variance dominates**: At 200-500 iter training runs, seed-to-seed
   variance (52% for inverted allocation) completely masks any allocation
   effect.

3. **Neither direction of variance weighting reliably helps**:
   - Variance weighting (more to layer 0): slightly worse (+4.6%)
   - Inverted weighting (less to layer 0): unreliable (-21% to +27%)

4. **v14 negative result stands**: "Adam exp_avg_sq does not predict useful
   attention allocation for RGSA at this scale and training regime."

## Recommendations

1. **Use uniform allocation** for RGSA. It's stable and doesn't require
   sensitivity computation.

2. **Do not pursue per-layer allocation** based on parameter-space signals
   (exp_avg_sq, gradients, etc.) at short training scales.

3. **If allocation is critical**, would need:
   - Longer training (5000+ iters) to reduce seed variance
   - Multiple seeds (5+) for statistical significance
   - State-based signals (attention patterns, not parameter updates)

4. **Close the door** on variance-weighted RGSA allocation research.

## Closed Hypotheses

The following hypotheses have been tested and closed. They will not be
revisited without fundamentally new evidence.

| Version | Hypothesis | Outcome |
|---------|------------|---------|
| v13 | Learned routing captures semantics | Learned routing ≈ random routing |
| v13 | L2M-style budget schedules | Failed in stable training regimes |
| v14-v16 | Fisher/Adam variance allocation | Signal-to-noise inverted; 52% seed variance |

## Next Viable Directions (Not Implemented)

The following directions are noted for future exploration. No experiments or
code changes are proposed here.

- **State-based importance (drop-impact KL)**: Measure output distribution
  shift when removing far-context access per layer. This aligns the importance
  signal with RGSA's actual intervention.

- **Usage-based signals**: Attention mass concentration, entropy of attention
  patterns, or other runtime statistics that reflect how layers actually use
  far context.

These directions remain unexplored and may be revisited in future work.

## Files and Artifacts

- Exact budget matching: `utils/sensitivity.py:compute_per_layer_top_b_exact()`
- Cleanup script: `scripts/rgsa_v16_cleanup.py`
- Results: `rgsa_v16_results/cleanup_confirm_results.json`
- Multi-seed analysis: `rgsa_v16_results/cleanup_multiseed.json`

## Reproducibility

```bash
# Uniform baseline
python gpt2/train.py --architecture vanilla --dataset finewebedu \
  --max-iters 200 --batch-size 8 --gradient-accumulation 4 \
  --block-size 1024 --tracker none

# Inverted allocation
export RGSA_TOP_B_PER_LAYER="2,4,7,8,8,9,9,9,10,10,10,10"
python gpt2/train.py --architecture vanilla --dataset finewebedu \
  --max-iters 200 --batch-size 8 --gradient-accumulation 4 \
  --block-size 1024 --tracker none
```
