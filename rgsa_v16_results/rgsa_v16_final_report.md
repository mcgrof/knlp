# RGSA v16 Final Report: Variance Postmortem Cleanup

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

#### Seed 1 Results (200 iters)

| Allocation | top_b_per_layer | PPL | vs Uniform |
|------------|-----------------|-----|------------|
| uniform | [8]*12 | 701.82 | baseline |
| variance_0.5 | [16,16,16,12,5,5,5,5,4,4,4,4] | 733.92 | +4.6% |
| inverted_0.5 | [2,4,7,8,8,9,9,9,10,10,10,10] | 553.10 | -21.2% |

Initial interpretation: Inverted weighting is dramatically better!

#### Seed 2 Results (200 iters)

| Allocation | PPL | vs Uniform |
|------------|-----|------------|
| uniform | 664.76 | baseline |
| inverted_0.5 | 843.27 | +26.9% |

**CRITICAL**: Seed 2 shows inverted is WORSE, not better!

### A.4 Seed Variance Analysis

The cross-seed variance is enormous:
- Uniform: 701.82 vs 664.76 (5.5% difference)
- Inverted: 553.10 vs 843.27 (52% difference!)

This 52% swing across seeds means any single-seed result is unreliable.

## Part B: Drop-Impact KL - SKIPPED

Given that seed variance dominates allocation effects at 200 iters, there is
no point implementing drop-impact KL measurement. The signal would be swamped
by training noise.

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
