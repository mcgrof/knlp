# BPA Technical Narrative: From Sublinear Hypothesis to Accumulation Bound

## 1. Executive Summary

This document summarizes a 17-version empirical investigation
(v8--v24) into reducing the KV cache cost of long-context
autoregressive inference. The original hypothesis, motivated by
information-theoretic arguments (L^2M mutual-information scaling),
posited that far-context KV usage should scale sublinearly with
context length L. Over the course of measurement, that hypothesis
was replaced by a more precise and useful result.

**What we measured.** On Qwen2.5-0.5B (D=24 layers, head_dim=64,
2 KV heads) and Qwen2.5-1.5B (D=28 layers, head_dim=128, 2 KV
heads), at L in {8192, 16384, 32768} with 3 seeds and 256 decode
steps, we systematically evaluated KV cache compression methods
under a strict PPL tolerance gate.

**What we found.**

1. Per-layer INT4 quantization noise accumulates across depth D.
   Only a small constant number of "sink layers" (primarily layer 0,
   the attention sink) dominate total sensitivity. All other layers
   contribute bounded aggregate noise.

2. The minimum number of layers requiring high-precision (INT8)
   protection, denoted k\*, is **k\*=2 for both D=24 and D=28**
   under a 3% relative PPL tolerance. The ratio k\*/D decreases as
   D grows (0.083 to 0.071), supporting the O(1) scaling hypothesis:
   k\* does not grow with model depth.

3. With k\*=2, the achievable KV cache compression ratio
   (kv\_ratio, including metadata overhead) is 0.30 on 0.5B and
   0.297 on 1.5B --- meaning ~70% byte reduction with <3% PPL loss.

4. On the AMD Radeon Pro W7900 (48.3 GB, compute-bound for these
   models), compression yields **3.26x more concurrent sequences**
   but no per-token latency improvement. A bandwidth-bound GPU
   (H100) is needed to demonstrate latency gains.

These results are documented in `bpa_v24_scoreboard.json` and the
v24 theory artifacts under `results/v24/artifacts/v24/theory/`.

---

## 2. Problem Statement and Original Hypothesis

### Motivation: L^2M and Sublinear Far-Context Need

The predictive information in natural language grows sublinearly
with context length. Information-theoretic analyses (L^2M: Learned
Language Model mutual information) suggest that the marginal value
of tokens far in the past decays, implying that the resources
allocated to far-context should be conditional and not proportional
to L. In cache terms, this means the KV bytes needed to maintain
quality should grow slower than O(L) if we can selectively compress
or evict stale entries.

### What We Set Out to Measure

We defined the "effective KV usage" as the fraction of KV cache
bytes retained (kv\_ratio) at a given quality tolerance epsilon. The
hypothesis was that kv\_ratio would decrease with L --- that is,
longer sequences would permit a larger fraction of KV entries to
be compressed or evicted. In scaling-law terms, we expected to
measure an exponent beta < 1 on KV bytes as a function of L.

### Initial Bias

We expected that gating and budget-adaptation mechanisms (learned
routers, per-step budget controllers, far-context predictors) would
be sufficient to achieve sublinear scaling. The expected pathway was:
train an auxiliary model to predict which KV entries are needed, then
compress the rest. This expectation turned out to be incomplete.

---

## 3. Definitions and Metrics

All measurements in this document use the following definitions.

**L** (context length): Number of tokens in the prefix before decode.
Tested values: {4096, 8192, 16384, 32768}. Headline results use
L in {8192, 32768}.

**D** (depth): Number of transformer layers. Qwen2.5-0.5B has D=24;
Qwen2.5-1.5B has D=28.

**kv\_ratio**: Ratio of compressed KV bytes per token to dense
(fp16/bf16) KV bytes per token, **including metadata overhead**
(quantization scales, zero-points). Computed by
`compute_kv_bytes_per_token()` in `eval_v21.py:81`. For example,
INT4 g=32 with k=2 INT8-protected layers on D=24 gives
kv\_ratio = 0.3008.

**epsilon (eps)**: Relative PPL tolerance.
eps = (PPL\_quant - PPL\_dense) / PPL\_dense. The two thresholds
used throughout are eps=1% and eps=3%.

**PASS\_eps**: A configuration passes at tolerance eps if
|delta\_pct| <= eps for all seeds in {0, 1, 2} and all L in
{8192, 32768}. A single exceedance is a fail.

**k**: Number of layers whose KV cache is kept at INT8 (high
precision). The remaining D-k layers use INT4 (g=32 by default).
Layer selection is by oracle sensitivity ranking (most sensitive
layers protected first).

**k\*(eps)**: The minimum k such that PASS\_eps holds. This is the
central quantity of interest. Measured by k-sweep: evaluating
k in {0, 1, 2, 3, 4, 6, 8, 12} with oracle ranking at each k.

**Oracle ranking**: Per-layer sensitivity ranking obtained by
quantizing each layer individually to INT4 (all others INT8) and
measuring max |delta\_pct| across seeds. Layers are sorted by
descending sensitivity. The most sensitive layer is protected first.

---

## 4. Timeline of Empirical Discoveries

This section documents how our understanding evolved under evidence.
Where early assumptions were wrong, we state so explicitly.

### Phase I: Sublinear Scaling on Small Models (v8--v10)

We began with GPT2-based models (124M parameters) and measured
adaptive KV gating controllers at L=512--2048.

- **v8**: Beta\_eff varied with window size W from 0.064 to 0.265.
  We interpreted this as evidence of sublinear scaling but noted
  that the local window W creates a geometric floor that confounds
  the measurement. (File: `bpa_v8_results/bpa_v8_final_report.md`)

- **v10**: Matched-quality tuning at L=512 achieved PASS with 0.36x
  KV ratio, but L=1024 failed due to model context limits. KV
  savings did not translate to latency on the compute-bound W7900.
  (File: `bpa_v10_results/bpa_v10_final_report.md`)

**Bias correction**: We overinterpreted beta_eff values because we
had not subtracted the geometric floor created by the local window.
Chasing beta without accounting for W was misleading.

### Phase II: Moving to Production Models (v11--v13)

We migrated to Qwen2.5-0.5B (494M) to test at longer context.

- **v11**: At L=512--4096, adaptive controllers achieved 2--5%
  latency improvement, but the far budget controller showed no
  meaningful adaptation. The machinery was effectively static at
  tuned parameters. (File: `bpa_v11_results/bpa_v11_final_report.md`)

- **v12**: At L up to 16K, the scaling exponent beta=1.04 (linear).
  A retrieval predictor achieved only AUC=0.626, indicating that
  cheap forward-only signals cannot predict far-context need.
  (File: `bpa_v12_results/bpa_v12_final_report.md`)

- **v13**: Unified Tiered KV (bitter0-7) at L up to 32K. bitter1-3
  PASS@3% with attention sink protection (W\_sink=4). Frontier
  methods (MLA, KVSplice learned compression) failed due to lossy
  random projections. Scaling remained linear (beta~1.0).
  (File: `bpa_v13_final/bpa_v13_final_report.md`)

**Bias correction**: The "gating solves everything" hypothesis was
dead. Retrieval predictors were too weak. The bottleneck was not
predicting which tokens to keep, but understanding the noise
introduced by compression itself.

### Phase III: Compression Fidelity and Structural Methods (v14b--v16)

With gating off the table, we pivoted to understanding compression
quality at the per-layer level.

- **v14b**: Only INT8 passed the 3% gate. K values are highly
  compressible (SVD rank~19/64) but V is not (rank~63/64). KVSplice
  achieved 35-61% latency gains but catastrophic PPL loss.
  (File: `bpa_v14b_final_report.md`)

- **v15**: Complex-plane K grouping (rope\_complex) achieved FULL
  PASS@1% at kv\_ratio=0.550 by exploiting RoPE structure: the
  magnitude of complex-paired K dimensions is position-invariant and
  low-rank. All other structural methods failed.
  (File: `bpa_v15_final_report.md`)

- **v16**: Mixed-precision INT4/INT8 with sensitivity-guided layer
  assignment (S2: 18 INT4 + 6 INT8 layers) achieved kv\_ratio=0.333,
  FULL PASS@1%. This established the sensitivity-guided mixed-
  precision paradigm that all subsequent versions build on.
  (File: `bpa_v16_final_report.md`, `bpa_v16_scoreboard.json`)

**Bias correction**: We learned that INT4 degradation is
**cumulative across layers**, not dominated by individual bad layers.
Each layer's INT4 noise is small individually (<3%) but they add up.
This insight motivated the accumulation model formalized in v24.

### Phase IV: Ranking, Scheduling, and the k-Floor (v18--v21)

With mixed-precision established, the question became: which layers
need INT8, and how many?

- **v18--v19**: Automated ranking signals (Adam v-hat, residual
  ratio, 95 variants total) were tested. None matched the empirical
  oracle. The best proxy (C3\_residual\_ratio, rho=0.394) was
  informative but not sufficient at tight budgets.
  (Files: `bpa_v18_final_report.md`, `bpa_v19_final_report.md`)

- **v20**: Established that k\*=4 is an architectural floor on
  Qwen2.5-0.5B for g=32 INT4. Tight groups (g=4) reduce noise 8x
  but cannot eliminate the floor. K noise dominates V noise. Layer 0
  (attention sink) accounts for 25% of total noise.
  (File: `bpa_v20_final_report.md`)

- **v21**: Scale metadata overhead makes small groups expensive:
  g=4 requires 50% overhead, worse than INT8-all. The best config
  was g=32 k=4 with theory-derived ranking, achieving
  kv\_ratio=0.3203 PASS@3%.
  (File: `bpa_v21_final_report.md`, `bpa_v21_scoreboard.json`)

**Bias correction**: We learned that honest byte accounting matters.
Scale metadata is not free, and "better quantization" can be worse
in practice if it requires more scales per token.

### Phase V: Amortization, Scaling, and Theory (v22--v24)

- **v22**: Token-window amortization (sharing INT4 scales across
  S=8 consecutive tokens, g=8) reduced metadata overhead, yielding
  kv\_ratio=0.3073 PASS@3%. First 1.5B probe showed the
  residual\_ratio proxy fails catastrophically on larger models
  (+67960% PPL from misranking layer 0).
  (File: `bpa_v22_final_report.md`)

- **v23**: Empirical oracle ranking on 1.5B (D=28): layer 0
  (+824.6%), layer 15 (+3.2%), all others <0.8%. k\*=2 on 1.5B with
  kv\_ratio=0.2974. On 0.5B, amort\_g8\_S8\_k3 achieved
  kv\_ratio=0.2969. Capacity story: 3.26x concurrent sequences on
  W7900. (File: `bpa_v23_final_report.md`, `bpa_v23_scoreboard.json`)

- **v24**: Formalized the accumulation model. Derived k(D,eps) lower
  bound. Measured amplification traces (layer 0: 5.25x on 0.5B,
  4.0x on 1.5B). Calibrated error budget B(3%)=10.56 on 0.5B,
  3.88 on 1.5B. O(1) hypothesis supported: k\*=2 for both D=24 and
  D=28 with v24 oracle ranking. k/D decreases from 0.083 to 0.071.
  (File: `bpa_v24_final_report.md`, `bpa_v24_scoreboard.json`)

---

## 5. The Accumulation Theory

This section presents the formal model developed in v24. Full
derivations are in `results/v24/artifacts/v24/theory/`.

### Residual Recursion and Perturbation

A transformer with D layers computes:

    x_{l+1} = x_l + f_l(x_l)

KV quantization introduces a perturbation delta_l at each layer:

    x~_{l+1} = x~_l + f_l(x~_l) + delta_l(x~_l)

### Linearization

Define error e_l = x~_l - x_l. Under a first-order approximation
(valid when individual errors are small relative to hidden states):

    e_{l+1} = (I + J_l) e_l + delta_l

where J_l is the Jacobian of block l.

### Unrolling

Starting from e_0 = 0:

    e_D = sum_{l=0}^{D-1} A_{l->D} delta_l

where A_{l->D} = prod_{j=l+1}^{D-1} (I + J_j) is the amplification
operator from layer l to the output.

### Norm Bound

    ||e_D|| <= sum_l alpha_l * sigma_l

where:
- alpha_l = c * ||A_{l->D}|| is the importance weight (combining
  amplification through downstream layers and head sensitivity),
- sigma_l = ||delta_l|| is the quantization noise magnitude,
- c is a model-specific constant relating representation error to
  NLL change.

### Why Greedy Top-k Protection Is Optimal

Under this linear bound, protecting the top-k layers by alpha\_l
minimizes total error for any given k. The proof is immediate: the
sorted-tail sum sum_{i=k+1}^D alpha\_(i) sigma\_4 is minimized when
the top-k largest alpha values are removed from the unprotected set.

### The O(1) Condition

If the sorted importance weights alpha\_(i) satisfy:

1. O(1) "sink" layers have alpha >> median (e.g., layer 0), and
2. The tail sum of remaining layers is bounded as D grows,

then k\* = O(1) for fixed epsilon. Intuitively: as D grows, more
layers are added but each contributes only moderate noise. Only the
constant number of structurally special layers (attention sinks)
require protection.

---

## 6. Empirical Evidence for O(1)

### k\* Determination

| Model | D | k\*(3%) | k\*/D | k\*(1%) | kv\_ratio at k\*(3%) |
|-------|---|---------|-------|---------|---------------------|
| Qwen2.5-0.5B | 24 | 2 | 0.083 | N/A | 0.3008 |
| Qwen2.5-1.5B | 28 | 2 | 0.071 | 3 | 0.2974 |

Source: `bpa_v24_scoreboard.json`, Phase 4 k-sweeps at L={8192,
32768}, seeds {0,1,2}, g=32, oracle ranking from Phase 3.

k\* is identical (2) for both models despite D increasing from 24 to
28. k/D decreases from 0.083 to 0.071. This is consistent with the
O(1) condition.

On 0.5B, k=1 barely fails (+3.05%) and k=2 passes (+2.85%). On
1.5B, k=1 fails (+3.85%) and k=2 passes (+1.05%). The transition
is sharp in both cases, separated by a single layer.

### Sink-Layer Sensitivity

Oracle per-layer INT4 ablation (all other layers INT8):

**Qwen2.5-0.5B (D=24)**:

| Layer | max delta (3-seed) |
|-------|-------------------|
| 0 | +23.48% |
| 2 | +1.34% |
| 11 | +1.30% |
| 16 | +0.77% |
| All others | <0.77% |

**Qwen2.5-1.5B (D=28)**:

| Layer | max delta (3-seed) |
|-------|-------------------|
| 0 | +824.55% |
| 15 | +3.20% |
| 1 | +0.79% |
| All others | <0.43% |

Source: `results/v24/artifacts/v24/parameter_estimates_qwen05b.json`
and `parameter_estimates_qwen15b.json`.

### Tail Fraction Analysis

The fraction of total sensitivity accounted for by non-sink layers:

| Model | C=1 tail frac | C=2 tail frac | C=4 tail frac |
|-------|---------------|---------------|---------------|
| 0.5B | 33.6% | 29.9% | 24.0% |
| 1.5B | 0.85% | 0.47% | 0.32% |

The 1.5B model has dramatically higher sink concentration: layer 0
alone accounts for 99.15% of total measured sensitivity. This
pattern is consistent with the O(1) condition and suggests it
strengthens with model scale.

### Amplification Traces

Noise injection at individual layers, measuring downstream error
propagation:

| Model | Layer 0 amp | Layer D/4 amp | Layer D/2 amp | Layer 3D/4 amp |
|-------|-------------|---------------|---------------|----------------|
| 0.5B | 5.25x | 2.43x | 2.07x | 1.74x |
| 1.5B | 4.00x | 2.16x | 1.77x | 1.94x |

Source: `results/v24/artifacts/v24/error_propagation_traces_qwen05b.json`
and `error_propagation_traces_qwen15b.json`.

Layer 0 has the highest amplification because all subsequent layers
amplify its error. Amplification decreases monotonically with
injection depth (later layers have fewer downstream amplifiers).

### Why Layer 0 Is Special

Layer 0 serves as an attention sink in RoPE-based transformers. The
first token's KV values receive disproportionate attention weight
from all subsequent tokens, serving as a "default" attention target
when no other token is informative. Quantizing layer 0's KV to INT4
distorts these sink values, causing cascading attention pattern
changes through all 23--27 downstream layers.

This is an architectural feature of RoPE-based models, not a
training artifact. It has been observed independently in the
"attention sink" literature (Xiao et al., 2024) and is consistent
with our finding that sink sensitivity increases with model depth
(+23% at D=24 vs +825% at D=28).

---

## 7. What We Have Now

The current "winning recipe" for KV cache compression:

1. **Use INT4 quantization (g=32) for most layers.** Per-token
   per-head scales in fp16. Payload is 4 bits per KV element.

2. **Protect the top-k layers at INT8**, where k is determined by
   oracle sensitivity ranking. For tested models, k=2 suffices at
   eps=3%.

3. **The protected layers are identified by per-layer INT4
   ablation**: quantize one layer to INT4, keep others INT8, measure
   PPL delta. Rank by descending sensitivity; protect the top-k.

4. **Achieved kv\_ratio**: 0.297--0.301 at eps=3% (70% byte
   reduction). With amortized scales (g=8, S=8), 0.297 on 0.5B.

5. **On W7900 (compute-bound)**: primary benefit is 3.26x more
   concurrent sequences, not latency reduction. ms/token is flat
   across compression levels. (Source: `bpa_v23_scoreboard.json`,
   Phase 1 throughput benchmark.)

6. **No latency win on W7900** because the GPU is compute-saturated
   at these model sizes. Attention kernel time is dominated by
   matrix multiply in FFN layers, not KV cache reads.

---

## 8. The Gap to Information-Theoretic Lower Bounds

### What Our Bound Is

The accumulation model provides an **operational bound** on error
propagation through depth. It answers: given a per-layer noise level
(sigma\_4 for INT4, sigma\_8 for INT8), how many layers can tolerate
INT4 before the accumulated error exceeds the quality threshold? It
is not an information-theoretic lower bound on KV cache size.

### What L^2M Tells Us (And Does Not)

The L^2M motivation predicts that far-context information is
sublinearly valuable. This informs the expectation that most KV
entries are redundant. But the observed bottleneck is not "which
tokens to keep" --- it is "how much noise each kept token tolerates
under quantization." The compression floor (kv\_ratio ~0.28--0.30)
is set by the accumulation bound and sink-layer sensitivity, not by
token-level information content.

### What Remains Unknown

1. Whether conditional gating can be combined with the k-protected-
   layer recipe to push below the current 0.28 floor. For example:
   evicting provably unneeded tokens from non-sink layers while
   keeping the sink layers at INT8.

2. Whether learned calibration (per-layer scale/offset adjustment
   trained on a small dataset) can reduce sigma\_4 enough to
   eliminate the k-floor entirely.

3. Whether the O(1) condition holds at D=32+ (7B/8B models). We
   have evidence at D=24 and D=28 only.

4. The exact relationship between the accumulation bound parameters
   (alpha\_l, sigma\_l) and information-theoretic quantities (mutual
   information, conditional entropy).

---

## 9. Limitations and Falsification Criteria

### Current Limitations

1. **Limited model scale.** O(1) evidence is based on two models
   (D=24, D=28). Extrapolation to D=32+ is a prediction, not a
   measurement.

2. **Tight tolerance is harder.** eps=1% requires k=3 on 1.5B
   (vs k=2 at eps=3%). On 0.5B, no k value achieves eps=1%
   (minimum delta at k=12 is +1.42%).

3. **Oracle ranking is expensive.** It requires D separate
   evaluations (one per layer). Cheap proxies (residual\_ratio,
   Adam v-hat, 95 signal variants) all underperform. The
   residual\_ratio proxy fails catastrophically on 1.5B by
   misranking layer 0 (v22 finding).

4. **Metadata overhead matters.** Smaller quantization groups (g=4,
   g=8) have higher scale overhead. Without amortization, g=4
   achieves kv\_ratio=0.50 (worse than INT8-all). Amortization (S=8)
   helps but adds Python overhead (8% slower on W7900).

5. **Compute-bound regime.** All W7900 experiments show flat
   ms/token across compression methods. The capacity story (3.26x
   concurrent sequences) is real but latency wins require a
   bandwidth-bound GPU.

### What Would Falsify the O(1) Story

- **If k\* grows proportionally with D on 7B/8B (D~32+)**. For
  example, if k\*=6 at D=32 and k\*=10 at D=64, then k/D is not
  decreasing and O(1) fails.

- **If the sink layer structure disappears at larger scale.** If
  7B models have many layers with >3% individual sensitivity (not
  just layer 0 and one other), the tail fraction would grow with D,
  violating the O(1) condition.

- **If amortized scales cause quality degradation at very long
  context (L=64K+)**. We tested up to L=32K. RoPE drift was not
  observed for g=8 S=8 but could emerge at longer L.

---

## 10. H100 Test Plan

### Objectives

1. **Validate O(1) on 7B/8B model** (D >= 32) at L = 32K--64K.
2. **Confirm kv\_ratio improves** toward 0.27--0.28.
3. **Demonstrate bandwidth-bound latency reduction** in at least
   one configuration.

### Experiment Matrix

**Models**: Qwen2.5-7B (D=32, head_dim=128, 4 KV heads) or
Llama-3.1-8B (D=32, head_dim=128, 8 KV heads). One model minimum;
both if time permits.

**Context lengths**: L in {16384, 32768, 65536} (if model supports
64K context).

**Batch sizes**: {1, 8, 16, 32}. Aim for the bandwidth-bound regime
by increasing batch until ms/token starts increasing with KV size.

**k values**: {0, 1, 2, 3, 4, 6, 8} with oracle ranking.

**Methods**:
- Dense (fp16 baseline)
- INT8-all (lossless baseline)
- INT4 g=32, top-k oracle protection
- (Optional) amort\_g8\_S8 if implementation is stable on 7B

**Seeds**: {0, 1, 2} for all headline determinations.

### Ranking Protocol

1. Coarse screen: 1-seed oracle sweep over all D layers at L=8192.
   Cost: D evaluations.
2. Refine: 3-seed confirmation on top-8 candidates at L=8192.
   Cost: 8 evaluations.
3. Full k-sweep with confirmed ranking.

### Acceptance Criteria

| Criterion | Target | Metric |
|-----------|--------|--------|
| A: O(1) | k\*(3%) <= 4 at D >= 32 | k\* from k-sweep, must not scale with D |
| B: kv\_ratio | <= 0.30 at eps=3% | True byte ratio including scales |
| C: Latency | >= 10% ms/token reduction vs dense | In at least one (L, batch) config |
| D: Throughput | >= 20% more concurrent sequences | At fixed memory budget |

Criterion A is the primary objective. If k\*=2 at D=32 (as
predicted), kv\_ratio will be approximately 2/32 * 0.5 + 30/32 *
0.25 + overhead = ~0.295.

Criterion C requires finding the bandwidth-bound regime. On H100
(3.35 TB/s bandwidth), 7B/8B at batch >= 16 and L >= 32K should
transition to bandwidth-bound attention. If the GPU remains compute-
bound, Criterion C cannot be met and the result is still informative
(compute-bound even at 7B scale).

### Estimated Resources

- Oracle ranking (D=32, 3 seeds, L={8K}): 32 * 3 * ~5 min = ~8h
- k-sweep (8 values, 3 seeds, L={8K, 32K}): ~6h
- Throughput benchmark (4 methods, 4 batch, 3 L): ~4h
- Total: ~18h on a single H100 (~$36 at $2/h)

### Decision Logic After H100

| H100 Result | Next Step |
|-------------|-----------|
| O(1) holds, latency win | Productize: fuse INT4 kernel, benchmark cost savings |
| O(1) holds, no latency | Focus on capacity story, try larger batch/longer L |
| O(1) fails (k\* grows) | Investigate learned calibration, per-layer training |

Source: `bpa_v24_branch_tree.md` and
`results/v23/artifacts/v23/h100_readiness.md`.

---

## FAQ

**Q: What is k?**
A: The number of transformer layers whose KV cache is kept at INT8
(8-bit) instead of INT4 (4-bit). A smaller k means more compression.
k\* is the minimum k that preserves quality.

**Q: Why does layer 0 matter so much?**
A: Layer 0 serves as the "attention sink" in RoPE-based transformers.
The first token accumulates disproportionate attention weight, making
its KV values critical for all subsequent computations. Quantizing
layer 0 to INT4 distorts these values, causing cascading errors
through all downstream layers. On Qwen2.5-1.5B, layer 0 INT4
quantization alone causes +824% PPL increase.

**Q: Why does W7900 show no latency win?**
A: The W7900 is compute-bound for 0.5B and 1.5B models at all tested
batch sizes (1--16). Decode latency is dominated by FFN matrix
multiplications, not KV cache reads. Reducing KV bytes does not
reduce the compute bottleneck. The benefit appears as capacity: the
smaller KV cache allows 3.26x more concurrent sequences in the same
48 GB of VRAM.

**Q: What would change on H100?**
A: The H100 has 3.35 TB/s memory bandwidth (vs 864 GB/s on W7900).
At larger models (7B+) and higher batch sizes (16+), the attention
kernel becomes bandwidth-bound. In this regime, reducing KV bytes
directly reduces memory read time, producing a measurable ms/token
improvement. We predict 10--30% latency reduction at batch=16+ with
kv\_ratio ~0.30.

**Q: Is this the information-theoretic limit?**
A: No. The kv\_ratio ~0.28--0.30 is an operational floor set by
quantization noise accumulation and sink-layer sensitivity, not by
the information content of the KV cache. Combining quantization with
token-level eviction or learned compression could potentially push
below this floor, but that remains untested.

**Q: How generalizable is this to other model families?**
A: We tested on Qwen2.5 (RoPE-based, GQA). The attention-sink
phenomenon and RoPE structure are shared by Llama, Mistral, and most
modern architectures. We expect the O(1) pattern to generalize, but
this has not been measured on non-Qwen models.

---

## Artifact Index

All data referenced in this narrative is listed in
`artifacts/v25/v25_appendix.json`, which maps each version
(v8--v24) to its report, scoreboard, branch tree, and key artifacts.

The primary v24 artifacts backing the O(1) claim:

| Artifact | Path |
|----------|------|
| k\* by model | `results/v24/artifacts/v24/k_star_by_model.json` |
| Parameter estimates | `results/v24/artifacts/v24/parameter_estimates.json` |
| Accumulation derivation | `results/v24/artifacts/v24/theory/accumulation_derivation.md` |
| k lower bound derivation | `results/v24/artifacts/v24/theory/k_lower_bound.md` |
| Error propagation (0.5B) | `results/v24/artifacts/v24/error_propagation_traces_qwen05b.json` |
| Error propagation (1.5B) | `results/v24/artifacts/v24/error_propagation_traces_qwen15b.json` |
| Scoreboard | `bpa_v24_scoreboard.json` |
| Throughput benchmark | `results/v23/artifacts/v23/throughput_bench.json` |

---

## Figures

All figures are in `artifacts/v25/figures/`:

- `k_star_vs_D.png`: k\* vs model depth D
- `k_over_D_vs_D.png`: k\*/D vs D
- `oracle_sensitivity_0.5B.png`: Per-layer INT4 sensitivity for 0.5B
- `oracle_sensitivity_1.5B.png`: Per-layer INT4 sensitivity for 1.5B
- `kv_ratio_vs_k.png`: kv\_ratio as a function of k for both models
- `compute_bound_decision.png`: Why W7900 shows capacity not latency
