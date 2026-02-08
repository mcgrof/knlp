# RGSA v14 Final Report: Variance-Weighted L2M + Sensitivity-Aware Budgeting

## Executive Summary

**Conclusion: Variance prior is NOT predictive for RGSA budget allocation at
GPT-2 124M scale with 500 training iterations.**

The hypothesis that Adam optimizer's exp_avg_sq (diagonal Fisher proxy) could
guide per-layer attention budget allocation was tested and falsified. While
the sensitivity signal shows clear layer-wise structure (layer 0 is 90x more
sensitive than deep layers), using this signal to allocate budgets does not
improve and actively hurts model quality.

## v13 Recap (A0/A1 Baseline)

From v13 experiments:
- A0 (Fixed RGSA, uniform budgets): Baseline stable configuration
- A1 (Macro-only schedules, uniform reduction): Failed - uniform capacity
  reduction degraded quality once training stabilized

Key insight from v13: "Uniform capacity reduction fails once training is
stable. The failure mode is likely misallocation (budget removed from critical
subspaces), not just insufficient total budget."

This motivated v14's hypothesis: use Adam variance as a static sensitivity
prior to allocate attention capacity where it matters.

## v14 Implementation

### Sensitivity Extraction

Created `utils/sensitivity.py` with:
- `extract_sensitivity()`: Extracts per-layer S from Adam exp_avg_sq
- `compute_variance_weights()`: Computes w_l = S_l^alpha / sum(S^alpha)
- `compute_per_layer_top_b()`: Allocates per-layer budgets

Sensitivity saved to JSON alongside checkpoints with:
- S_layer: per-layer sensitivity values
- summary stats (min, max, p50, p90)
- mapping assumptions

### Variance-Weighted Allocation

Modified RGSA to accept per-layer top_b:
- `RGSAConfig.top_b_per_layer: Optional[List[int]]`
- `RGSAConfig.variance_alpha: float` (0=uniform, 0.5=sqrt, 1=linear)
- Layer-aware attention via `layer_idx` parameter

Budget allocation:
```
w_l = S_l^alpha / sum(S^alpha)
top_b_l = clamp(round(top_b_base * w_l * n_layer), min=2, max=16)
```

Total budget matched by searching for top_b_base that gives target total.

## A2 Results: Variance-Only Allocation at Fixed Budget

### Quick Sweep (200 iters, seed 1)

| Alpha | Val PPL | vs Uniform |
|-------|---------|------------|
| 0.0   | 552.32  | baseline   |
| 0.5   | 532.34  | -3.6%      |
| 1.0   | 893.47  | +62%       |

Initial interpretation: alpha=0.5 helps, alpha=1.0 hurts.

### Full Sweep (500 iters, seeds 1,2,3)

| Alpha | Seed | Val PPL |
|-------|------|---------|
| 0.0   | 1    | 241.54  |
| 0.0   | 2    | 364.85  |
| 0.0   | 3    | 200.19  |
| 0.5   | 1    | 353.84  |
| 0.5   | 2    | 198.14  |
| 0.5   | 3    | 431.85  |

Summary:
| Alpha | Mean PPL | Std PPL | Min    | Max    |
|-------|----------|---------|--------|--------|
| 0.0   | 268.86   | 85.66   | 200.19 | 364.85 |
| 0.5   | 327.94   | 118.99  | 198.14 | 431.85 |

**alpha=0.5 is 22% WORSE than uniform (alpha=0.0)**

### Sensitivity Pattern

From 100-iter baseline training:
- Layer 0: S = 0.010555 (highest, 90x above minimum)
- Layers 1-11: S ~ 0.0001-0.0005 (decreasing)

The sensitivity signal is real and shows clear layer-wise structure.

### Budget Allocations (matched total ~96)

- alpha=0.0: uniform [8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8]
- alpha=0.5: [16, 12, 8, 7, 7, 7, 7, 6, 6, 6, 6, 6]
- alpha=1.0: [16, 16, 10, 8, 7, 7, 6, 6, 5, 5, 5, 5]

## A3 Results: Variance + Budget Schedule

**SKIPPED per stopping rules.**

Phase 1 showed no improvement and no tail reduction from variance weighting,
triggering the stopping rule:
> "If Phase 1 shows alpha_var>0 gives no improvement and no tail reduction,
> stop variance work and write up 'variance prior not predictive for RGSA.'"

## Analysis

### Why Variance Weighting Failed

1. **Sensitivity != Importance for attention**
   Adam exp_avg_sq measures gradient magnitude, not attention importance.
   Layer 0 having high sensitivity means its parameters change rapidly,
   not that it needs more attention budget.

2. **Over-allocation to early layers**
   alpha=0.5 allocates 2x budget to layer 0 (16 vs 8).
   This may be counterproductive: early layers may need less routing
   because they're processing low-level features.

3. **Massive seed variance dominates**
   Coefficient of variation is ~30-35% across seeds.
   Any signal from variance weighting is swamped by training noise.
   500 iterations may be insufficient for stable comparisons.

4. **Quick vs Full sweep reversal**
   Quick sweep (seed 1 only) suggested alpha=0.5 helps.
   Full sweep (seeds 1,2,3) shows it hurts by 22%.
   This demonstrates the danger of single-seed evaluation.

### Tail Risk Analysis

alpha=0.5 has WORSE tail risk:
- Max PPL: 431.85 (alpha=0.5) vs 364.85 (alpha=0.0)
- Std PPL: 118.99 (alpha=0.5) vs 85.66 (alpha=0.0)

Variance weighting increases instability.

## Explicit Decision

**NO: Variance prior is not predictive for RGSA budget allocation.**

The sensitivity signal from Adam exp_avg_sq does not provide useful guidance
for attention budget allocation. The signal exists but does not correlate
with beneficial attention patterns.

## Recommendations

1. **Do not use variance-weighted allocation for RGSA** at this scale.

2. **If revisiting**, consider:
   - Longer training (5000+ iters) to reduce variance
   - Different sensitivity metrics (attention entropy, gradient w.r.t. loss)
   - Per-head rather than per-layer allocation
   - Fisher information computed directly on attention outputs

3. **For stable RGSA**, use uniform budget allocation (alpha=0.0).

## Files and Artifacts

- Implementation: `utils/sensitivity.py`, `gpt2/rgsa.py`
- Sweep script: `scripts/rgsa_v14_a2_sweep.py`
- Results: `rgsa_v14_results/a2_full_results.json`
- Progress log: `rgsa-v14.txt`

## Reproducibility

Training command:
```bash
python gpt2/train.py --architecture vanilla --dataset finewebedu \
  --max-iters 500 --variance-alpha 0.5 \
  --sensitivity-path <sensitivity.json> \
  --save-sensitivity
```

Sensitivity extraction:
```bash
python -c "
from utils.sensitivity import extract_sensitivity
extract_sensitivity('checkpoint.pt', 'sensitivity.json')
"
```
