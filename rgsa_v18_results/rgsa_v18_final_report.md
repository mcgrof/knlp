# RGSA v18 Final Report: State-Aligned Importance Signals

## Executive Summary

RGSA v18 implements infrastructure for discovering state/usage-aligned importance
signals for head-level budget allocation. Unlike v14-v16 (parameter-space signals),
v18 measures runtime state: which heads actually use far-context attention.

**Status: Infrastructure Complete**

All phases implemented with working code:
- Phase 0: Head-level metrics and budget matcher
- Phase 1: Drop-impact KL measurement
- Phase 2: Correlation analysis
- Phase 3: Policy comparison

Actual experiment runs on trained models are required to validate whether
state-aligned signals improve over uniform allocation.

## Phase 0: Head-Level Metrics and Budget Matcher

### Implementation (commit 5317b9f)

1. **HeadMetrics dataclass** (`gpt2/rgsa.py`):
   - `far_mass[l,h]`: Attention fraction outside local_window
   - `attn_entropy[l,h]`: Normalized attention entropy
   - `max_weight[l,h]`: Attention concentration
   - `top1_mass[l,h]`: Top-1 attention mass

2. **GPT2_RGSA.compute_head_metrics()**: Vectorized computation across all
   heads/layers with no nested Python loops.

3. **compute_per_head_top_b_exact()** (`utils/sensitivity.py`): Head-level
   budget allocation with exact total matching using floor + remainder
   distribution.

### Test Results

```
far_mass: mean=0.1544, std=0.0010
attn_entropy: mean=0.9940, std=0.0003
max_weight: mean=0.0291, std=0.0006

Budget sum: 48 (expected: 48) - EXACT MATCH
```

## Phase 1: Drop-Impact KL Measurement

### Implementation (commit f91bd6e)

1. **ImpactMetrics dataclass**: Tracks `impact_kl[l,h]` and measurement counts.

2. **GPT2_RGSA.compute_drop_impact_kl()**: For each head, restricts that head
   to local_window only and measures KL(baseline || dropped).

3. **GPT2_RGSA._forward_with_head_dropped()**: Modified forward with per-head
   attention masking.

4. **ImpactKLTracker class**:
   - EMA tracking over training
   - Round-robin scheduling (`heads_per_eval=8`)
   - State dict for checkpointing

### Test Results

```
4 round-robin steps covers all 16 heads
EMA updates correctly
Weights sum to 1.0
```

## Phase 2: Predictor Correlation Analysis

### Implementation (commit e4d64a0)

`scripts/rgsa_v18_phase2_correlation.py`:
- Spearman rank correlation
- Top-K overlap (k=5, 10, 15)
- Multi-batch averaging

### Results on Untrained Model

| Signal       | Spearman r | p-value | Significant |
|--------------|------------|---------|-------------|
| far_mass     | 0.1207     | 0.4831  | No          |
| attn_entropy | 0.1624     | 0.3439  | No          |
| max_weight   | -0.1322    | 0.4422  | No          |
| concentration| 0.1322     | 0.4422  | No          |

Weak correlations expected with random weights. This infrastructure will show
meaningful correlations on trained models where heads differentiate.

## Phase 3: Policy Comparison

### Implementation (commit f84a19b)

`scripts/rgsa_v18_phase3_policy_test.py`:
- Computes uniform, far_mass-weighted, and inverted allocations
- Verifies exact budget matching
- Multi-seed aggregation

### Results on Untrained Model

```
Allocation sums: all 48 (exact match)
PPL: identical across policies (expected for untrained)

Per-layer allocations (seed 1):
  uniform:   [6, 6, 6, 12, 12, 6]
  far_mass:  [6, 8, 8, 9, 8, 9]
  inverted:  [6, 9, 6, 9, 10, 8]
```

## What Remains for Full Experiment

To complete v18, run with actual training:

1. **Train RGSA model** with uniform allocation to convergence
2. **Compute head metrics** on validation data
3. **Run drop-impact KL** measurement (8 heads per eval, round-robin)
4. **Check correlations** between far_mass/entropy and impact_kl
5. **If correlation strong**: Use far_mass as cheap predictor
6. **If correlation weak**: Use impact_kl_ema directly (more expensive)
7. **Train with weighted allocation** for same tokens_seen
8. **Compare PPL** across seeds (1, 2, 3)

### Stopping Rules

- If impact_kl shows no structure (pure noise): stop
- If no predictor correlates with impact_kl: use impact_kl directly or conclude
  "no usable cheap signal found"
- If policy shows sign-flip across seeds: conclude "uniform is near-optimal"

## Files and Artifacts

### Source Code
- `gpt2/rgsa.py`: HeadMetrics, ImpactMetrics, ImpactKLTracker, compute_*
- `utils/sensitivity.py`: compute_per_head_top_b_exact, compute_head_weights_from_metrics
- `gpt2/trainers/vanilla.py`: W&B logging integration

### Scripts
- `scripts/rgsa_v18_sanity.py`: Phase 0 verification
- `scripts/rgsa_v18_phase2_correlation.py`: Predictor analysis
- `scripts/rgsa_v18_phase3_policy_test.py`: Policy comparison

### Results
- `rgsa_v18_results/phase0_sanity.json`
- `rgsa_v18_results/phase2_correlation.json`
- `rgsa_v18_results/phase3_policy_test.json`

## Commits

1. `5317b9f`: Head-level metrics and budget matcher (Phase 0)
2. `5a5825f`: Phase 0 sanity test and W&B integration
3. `f91bd6e`: Drop-impact KL measurement (Phase 1)
4. `e4d64a0`: Predictor correlation analysis (Phase 2)
5. `f84a19b`: Policy comparison infrastructure (Phase 3)

## Conclusion

RGSA v18 provides complete infrastructure for state-aligned head importance
signals. The key insight from v14-v16 was that parameter-space signals
(Adam exp_avg_sq) don't predict runtime state importance. v18 addresses this
by measuring runtime state directly:

1. **far_mass**: How much attention goes outside local_window
2. **impact_kl**: How much output changes when we restrict far-context

The implementation enables:
- Efficient round-robin measurement during training
- EMA tracking for stable importance estimates
- Exact budget allocation with head-level granularity
- Multi-seed policy comparison

To validate whether state-aligned signals beat uniform allocation, run the
full experiment protocol on trained models.
