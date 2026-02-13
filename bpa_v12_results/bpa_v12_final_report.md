# BPA v12 Final Report: Frontier Stress Test

## 1. Hardware + Model Configuration

GPU: AMD Radeon Pro W7900, 48.3GB VRAM, ROCm/HIP
Model: Qwen/Qwen2.5-0.5B (494M params, max_ctx=32768)
  24 layers, 896 hidden, 14 heads, 2 KV heads (GQA 7:1)

Qwen2.5-1.5B (1544M params) loaded successfully but not used
for experiments due to v11 tuning configs targeting 0.5B only.

## 2. Bandwidth-Sensitivity Analysis

Phase 1 measured decode latency scaling across L={1K..16K} and
batch={1,4,8}. Acceptance criterion: doubling L increases
latency >= 1.6x.

| Batch | L_from | L_to  | Latency ratio | Bandwidth-bound? |
|-------|--------|-------|---------------|-----------------|
|   1   |  8192  | 16384 |    1.611x     |      YES        |
|   4   |  8192  | 16384 |    1.649x     |      YES        |
|   8   |  4096  |  8192 |    1.534x     |      NO (OOM at 16K) |

Bandwidth-bound regime confirmed at L>=8192 for batch 1 and 4.
At batch=8, L=16384 hits OOM (37GB allocation exceeds 45GB VRAM).

Source: `bandwidth_scaling.json`

## 3. Retrieval Predictor Metrics

Phase 2 recorded attention maps from Qwen2.5-0.5B in eager
mode across 9 configs (L={1K,2K,4K} x 3 seeds, 128 decode
steps each). Far attention mass threshold set at p75 (0.4819),
yielding 25% positive rate (288/1152 samples).

Three feature sets tested with logistic regression:

| Feature Set   | # Features | ROC-AUC | Precision | Recall |
|---------------|-----------|---------|-----------|--------|
| Cheap (no attn)| 5         | 0.6263  | 0.486     | 0.062  |
| Attention only | 2         | 0.4029  | 0.000     | 0.000  |
| All combined   | 8         | 0.4029  | 0.000     | 0.000  |

Cheap features: entropy, resid_norm, logit_margin, top5_mass,
step_position. Attention features: avg_attn_entropy,
attn_entropy_std.

The attention-derived entropy features contained NaN values
(constant across samples), rendering them useless. Even the
cheap feature set achieves only AUC=0.63 — well below the
0.75 acceptance threshold.

The logit margin weight (+0.37) was the strongest signal:
tokens where the model is less confident about top-1 vs top-2
weakly correlate with far attention need. But the effect is
marginal and not actionable for a reliable controller.

Source: `retrieval_predictor.json`, `attention_features_raw.json`

## 4. Matched-Quality Tables

BPA kept tokens at 1% tolerance (v11 tuned configs):

|     L | Kept (mean) | % of L | Dense PPL |
|-------|-------------|--------|-----------|
|   512 |         306 |  59.7% |      21.4 |
|  1024 |        1013 |  98.9% |      13.2 |
|  2048 |        1960 |  95.7% |      10.4 |
|  4096 |        3883 |  94.8% |      10.3 |
|  8192 |        6832 |  83.4% |       8.5 |
| 16384 |       13412 |  81.9% |      12.4 |

Only L=512 shows meaningful eviction (40%). At L>=1024, BPA
must keep 82-99% of tokens to maintain quality.

Note: L=8192 and L=16384 used conservative defaults (no tuned
config) and produced degraded PPL (100-1100 on some seeds).

Source: `scaling_exponent.json`

## 5. Scaling Exponent Beta

Fit: kept_tokens ~ L^beta across L={512..16384}

beta = 1.0435
R^2  = 0.9829

Interpretation: LINEAR. No sublinear scaling evidence.

The v11 tuned configs at L={512,1024,2048,4096} produce a kept
fraction that decreases only from 99% to 95%. At larger L with
conservative defaults, kept fraction drops to 82% but with
quality collapse. The exponent exceeds 1.0 because L=512 is an
outlier where BPA can actually evict significantly.

Source: `scaling_exponent.json`

## 6. Layer-Adaptive W Comparison

Tested gradient (early=small W, late=large W) vs uniform W.
HF DynamicCache requires uniform cache length across layers,
so gradient profile uses mean W as effective eviction target.

| L    | Method           | PPL    | Kept  | p50 (ms) |
|------|------------------|--------|-------|----------|
| 1024 | dense            |  13.0  |  1152 |    9.85  |
| 1024 | gradient         |  13.1  |  1000 |    9.75  |
| 1024 | uniform (95%)    | 2230.7 |   974 |    9.69  |
| 2048 | dense            |  11.0  |  2176 |   11.44  |
| 2048 | gradient         | 816.1  |  1742 |   10.89  |
| 2048 | uniform (95%)    |  10.9  |  2010 |   11.24  |
| 4096 | dense            |   9.4  |  4224 |   14.40  |
| 4096 | gradient         | 226.8  |  3226 |   13.06  |
| 4096 | uniform (95%)    |   9.4  |  4084 |   14.28  |

The gradient profile sometimes matches quality (L=1024: PPL
13.1 vs 13.0) but catastrophically fails at L>=2048. The
inconsistency comes from random far chunk selection interacting
with the eviction boundary. Layer-adaptive W does NOT meet the
acceptance criterion of >=10% token reduction at same quality.

Source: `layer_adaptive_comparison.json`

## 7. 32K Context Stress

| Method | L     | PPL   | p50 (ms) | Kept  | GPU (MB) |
|--------|-------|-------|----------|-------|----------|
| dense  | 32768 | 19.6  | 58.03    | 32832 | 11513    |
| BPA    | 32768 | 266.2 | 53.45    | 29768 | 11777    |

BPA at 32K: W=90-95%, keeps 91% of tokens. PPL degrades 13.6x
despite minimal eviction. The 8% latency improvement is real
(53ms vs 58ms) but the quality degradation makes it unusable.

Source: `phase5_32k_stress.json`

## 8. Branch Tree Conclusions

**Assigned branch: C (Retrieval predictor weak, AUC < 0.7)**

All five v12 acceptance tests produced negative results:
1. Bandwidth-bound regime: FOUND (the only positive)
2. Retrieval predictor: FAIL (AUC=0.63 < 0.75)
3. Sublinear scaling: FAIL (beta=1.04, linear)
4. Layer-adaptive W: FAIL (quality collapses with eviction)
5. 32K stress: FAIL (PPL=266 vs 20)

BPA as a heuristic KV eviction controller has reached its
fundamental limit on this model. The 0.5B Qwen model uses
attention diffusely across context — there are no clear
"unimportant" tokens that can be safely evicted without an
attention-informed scorer. Random far chunk selection is too
noisy for production use.

Next steps require moving beyond heuristic eviction toward
either learned eviction scoring (H2O-style) or KV compression
(quantization, MLA latent projection, or paged KV).
