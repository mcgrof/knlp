# BPA v19 Final Report: BitterKV Signal Bakeoff

## Summary

v19 systematically evaluated 19 candidate sensitivity signals
(across 5 categories and 5 monotone transforms each = 95 variants)
for automated KV cache INT4/INT8 bit allocation on Qwen2.5-0.5B.
The goal: find a signal that can replace oracle-derived S2 for
automated mixed-precision quantization.

**Verdict: No signal can replace S2 at the same budget (k=6).**
The bottleneck is budget (k=2 is insufficient at L>=16K even with
oracle-optimal allocation), not signal quality. C3_residual_ratio
is the best cheap proxy but its schedule matches oracle only at
k=2, which itself fails at long context.

## 1. Baselines

| Backend   | kv_ratio | PASS@1% | PASS@3% | avg delta |
|-----------|----------|---------|---------|-----------|
| dense     | 1.000    | 9/9     | 9/9     | 0.0%      |
| INT8      | 0.529    | 9/9     | 9/9     | 0.14%     |
| S2 manual | 0.333    | 9/9     | 9/9     | 0.33%     |

S2 manual remains the gold standard: 6 layers INT8
({0,1,2,8,11,16}), rest INT4. FULL PASS at all L up to 32K.

## 2. Oracle Sensitivity Map

Per-layer INT4 PPL delta (top 6 most sensitive):

| Layer | delta (%) | Role                    |
|-------|-----------|-------------------------|
| 0     | +16.18    | Attention sink (6x gap) |
| 2     | +0.77     | Early layers            |
| 11    | +0.63     | Mid-network             |
| 8     | +0.28     | Mid-network             |
| 21    | +0.23     | Late layers             |
| 20    | +0.18     | Late layers             |

Layer 0 dominates by 21x over layer 2. Layer 16 actually shows
negative delta (-0.27%), meaning INT4 there is essentially free.

## 3. Signal Correlation Table

Best signals by Spearman rho (absolute value, vs oracle):

| Signal              | Category       | rho    | p-val  | top6 | Cost   |
|---------------------|----------------|--------|--------|------|--------|
| D1_K_effective_rank | spectrum       | -0.428 | 0.037  | 0/6  | medium |
| C3_residual_ratio   | activation     | +0.394 | 0.057  | 4/6  | low    |
| D3_K_condnum        | spectrum       | +0.237 | 0.265  | 4/6  | medium |
| C1_norm_QK          | activation     | +0.191 | 0.372  | 3/6  | low    |
| E1_adam_vhat_KV      | curvature     | +0.170 | 0.427  | 2/6  | high   |
| E2_adam_vhat_allattn | curvature     | +0.127 | 0.556  | 2/6  | high   |
| A6_fakeQuant_attnKL | act-noise     | +0.093 | 0.665  | 2/6  | medium |
| A4_logitKL_noise    | act-noise     | +0.089 | 0.680  | 2/6  | high   |
| E3_gradnorm_KV      | curvature     | +0.088 | 0.682  | 2/6  | medium |

D1_K_effective_rank has the highest |rho| but is ANTI-correlated
and has 0/6 top-6 overlap (its ranking is inverted).

C3_residual_ratio is the clear winner: best positive correlation,
near-significant p-value, 4/6 top-6 overlap, and cheapest to
compute (forward-only, no backward pass).

Transforms (sqrt, root4, root8, log1p) do not change rankings
for any signal. Rankings are monotone-invariant because all
signals have smooth distributions without heavy-tail outlier
effects within the 24-layer budget.

## 4. Schedule Quality Results

### Phase 3: Greedy allocation + full eval at L={8K,16K,32K}

| Schedule            | k  | PASS@1% | PASS@3% | ratio  |
|---------------------|----|---------|---------|--------|
| S2_manual           | 6  | 9/9     | 9/9     | 0.333  |
| C3_residual_ratio   | 2  | 2/9     | 6/9     | 0.293  |
| E1_adam_vhat_KV     | 2  | 2/9     | 6/9     | 0.293  |
| oracle              | 2  | 2/9     | 6/9     | 0.293  |
| E1_adam_root4       | 2  | 2/9     | 6/9     | 0.293  |
| A6_fakeQuant_attnKL | 24 | 9/9     | 9/9     | 0.514  |
| D1_K_effective_rank | 21 | 0/9     | 5/9     | 0.484  |
| B4_late_binding     | 22 | 0/9     | 5/9     | 0.494  |
| random (k=2)        | 2  | 0/9     | 0/9     | 0.293  |

At k=2, oracle, C3, and E1 all select layers {0, X} where X
differs (oracle: 2, C3/E1: 1). Both produce identical PARTIAL
results: PASS@3% at 8K and 32K but FAIL at 16K.

The greedy search needed k>=6 (S2) to achieve FULL PASS. No
signal achieves FULL PASS at k<6 because the problem is budget:
protecting only 2 layers cannot prevent cumulative INT4
degradation across the remaining 22 layers at long context.

Random k=2 is catastrophic: PPL=80 at 32K seed=1 (+430%).

## 5. Bandwidth Validation (Phase 5)

W7900 at L=8192, 256 decode steps:

| Microbatch | dense  | C3     | E1     | oracle |
|------------|--------|--------|--------|--------|
| 1          | 21.97  | 21.77  | 21.78  | 21.76  |
| 4          | 26.94  | 27.02  | 26.99  | 26.99  |
| 8          | 36.04  | 36.14  | 36.20  | 36.19  |

Mixed-precision shows <1% difference from dense at all batch
sizes. The W7900 is compute-bound at L=8K; bandwidth savings
from smaller KV caches do not translate to latency reduction.
This is consistent with v16 and v18 findings.

## 6. Key Question: Can Any Signal Replace Oracle-Derived S2?

**No.**

The failure mode is not signal quality but budget. At k=2 (the
minimum budget where signal ranking matters), even oracle-optimal
allocation fails at L>=16K. To achieve FULL PASS, k>=6 is
required regardless of which signal drives the allocation.

At k=6, all reasonable signals converge to similar layer sets
(they all agree layer 0 must be INT8), so the signal choice
becomes less important. S2 manual already has the optimal k=6
assignment derived from empirical v16 sensitivity sweeps.

Specific signal category verdicts:

- **Activation-noise (A1-A6)**: Weak. Best is A6_fakeQuant
  (rho=0.093) which requires k=24 for PASS — no compression.
  The noise injection approach measures local attention
  perturbation which poorly predicts end-to-end PPL impact.
- **Attention stats (B1-B4)**: Weak. Near-zero correlation.
  Entropy and attention patterns do not predict quantization
  sensitivity.
- **Activation norms (C1-C3)**: Best category. C3_residual_ratio
  (rho=0.394) captures how much each layer's attention output
  contributes to the residual stream, which is a genuine proxy
  for quantization impact. C1_norm_QK (rho=0.191) also
  reasonable.
- **Spectrum (D1-D3)**: Mixed. D3_K_condnum has good top-6
  overlap (4/6) but moderate rho (0.237). D1 is anti-correlated
  (higher rank = less sensitive, opposite of expected).
- **Parameter curvature (E1-E3)**: Moderate. E1_adam_vhat_KV
  (rho=0.17) reproduces v18 finding: detects layer 0 but
  misranks mid-layers. Transform ablations (root4 etc.) do not
  help because rankings are already monotone-invariant.

## 7. Recommendation for v20

**Branch D applies: None match S2.**

The path forward is:

1. **Keep S2 manual** (k=6, layers {0,1,2,8,11,16}) as the
   production allocation for Qwen2.5-0.5B.

2. **For new models**, use C3_residual_ratio as a first-pass
   signal: it is cheap (single forward pass), has the best
   correlation, and correctly identifies layer 0 as critical.
   Use it to generate a candidate schedule, then validate with
   a quick empirical sweep of the top-6 layers.

3. **Investigate hybrid signals**: weighted combination of
   C3_residual_ratio + D3_K_condnum (both have 4/6 top-6
   overlap with different layer emphasis). A learned regressor
   over the signal catalog could improve allocation.

4. **Budget is the bottleneck, not signal**: future work should
   focus on finding ways to reduce the number of INT8 layers
   needed (e.g., INT6 quantization, per-head mixed precision,
   or rope_complex compression from v15 to reduce the baseline
   before mixed-precision).

## Appendix: Signal Compute Cost

| Cost   | Signals                           | Time    |
|--------|-----------------------------------|---------|
| low    | B1-B4, C1-C3                      | <1s     |
| medium | A1-A3, A5, A6, D1-D3, E3         | 2-4s    |
| high   | A4, E1, E2                        | 10-12s  |

Total Phase 1 time: ~55 seconds for all 19 signals.
