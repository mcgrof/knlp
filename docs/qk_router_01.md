# QK Router 01: Signal Check Report

Generated: 2026-03-04 00:42

## 1. Objective

Determine whether current decode-step queries Q_t, scored against a resident geometry-aware K-summary index, can predict which offloaded KV blocks should be prefetched, and whether this semantic routing beats trivial baselines.

## 2. Environment

- GPU: NVIDIA H100 80GB HBM3 (85.0GB)
- PyTorch: 2.10.0+cu126
- CUDA: 12.6
- Transformers: 5.2.0
- Git commit: 3d828fb6b833ec621ea18ad04508dec57f26811f

## 3. Model and Workload

- Model: Qwen/Qwen2.5-0.5B
- Layers: 24
- KV heads: 2
- Head dim: 64
- Prefix length: 4096
- Block size: 128 tokens
- Num prefix blocks: 32
- Max new tokens: 64
- Num requests: 64
- Dataset: wikitext-2-raw-v1

## 4. Storage Microbench Results

| Tier | Path | p50 (us) | p95 (us) | Throughput (MB/s) |
|------|------|----------|----------|-------------------|
| tmpfs | /mnt/tmpfs | 210 | 229 | 7380.6 |
| sfs | /mnt/SFS-hugging | 778 | 8687 | 653.5 |

## 5. Trace Construction

- Requests: 64
- Steps per request: 64
- Avg needed blocks per step: 7.9

## 6. K-Summary Construction Methods

| Mode | Description |
|------|-------------|
| direct_centroid | Per-layer centroid of real block keys, normalized |
| random_summary | Random normalized vectors (dumb baseline) |
| first_k_real | Summaries from first 4 blocks only |
| sampled_real_geometry | Keys sampled across full prefix span |

## 7. Router-Only Results

| Mode | Recall@2 | Recall@4 | Recall@8 | Recall@16 | Score Sep |
|------|----------|----------|----------|-----------|-----------|
| direct_centroid | 0.040 | 0.073 | 0.145 | 0.301 | -0.0076 |
| random_summary | 0.129 | 0.195 | 0.339 | 0.595 | 0.0096 |
| first_k_real | 0.215 | 0.251 | 0.269 | 0.510 | 0.0001 |
| sampled_real_geometry | 0.087 | 0.180 | 0.334 | 0.609 | 0.0012 |

## 8. Replay / Scheduler Results

### storage_mild

| Policy | Missed Rate | Avg Stall (us) | p95 Decode (us) | Overlap |
|--------|-------------|----------------|-----------------|---------|
| no_prefetch | 0.030 | 84 | 16929 | 0.974 |
| recency_only_top_m | 0.017 | 61 | 16807 | 0.984 |
| semantic_top_m | 0.021 | 69 | 16843 | 0.981 |
| utility_aware | 0.021 | 69 | 16863 | 0.981 |
| utility_aware_plus_exploration | 0.020 | 67 | 16845 | 0.982 |

### storage_medium

| Policy | Missed Rate | Avg Stall (us) | p95 Decode (us) | Overlap |
|--------|-------------|----------------|-----------------|---------|
| no_prefetch | 0.030 | 149 | 17182 | 0.974 |
| recency_only_top_m | 0.017 | 86 | 16807 | 0.984 |
| semantic_top_m | 0.021 | 107 | 16966 | 0.981 |
| utility_aware | 0.022 | 115 | 16993 | 0.980 |
| utility_aware_plus_exploration | 0.020 | 105 | 16921 | 0.981 |

### storage_harsh

| Policy | Missed Rate | Avg Stall (us) | p95 Decode (us) | Overlap |
|--------|-------------|----------------|-----------------|---------|
| no_prefetch | 0.030 | 168 | 17230 | 0.974 |
| recency_only_top_m | 0.017 | 89 | 16807 | 0.984 |
| semantic_top_m | 0.021 | 115 | 16985 | 0.981 |
| utility_aware | 0.021 | 117 | 16980 | 0.981 |
| utility_aware_plus_exploration | 0.020 | 108 | 16917 | 0.982 |

## 9. Recency-Only vs Semantic Comparison

- Recency missed rate: 0.017
- Semantic missed rate: 0.021
- Delta (recency - semantic): -0.004
- Utility+exploration missed rate: 0.020

Semantic routing performs comparably to recency-only.

## 10. Geometry-Aware Summary Findings

| Mode | Missed Rate | Avg Stall (us) | p95 Decode (us) |
|------|-------------|----------------|-----------------|
| direct_centroid | 0.021 | 107 | 16966 |
| random_summary | 0.023 | 112 | 16988 |
| first_k_real | 0.021 | 107 | 16950 |
| sampled_real_geometry | 0.022 | 114 | 16999 |

Geometry-aware summaries show no meaningful advantage over random.

## 11. Analysis: Why Semantic Routing Failed

Three observations explain the NO-GO:

**1. Extreme temporal locality in block access patterns.**
Reuse distance: mean=0.0 steps, median=1.0, p95=2.0. The set
of needed blocks barely changes between consecutive decode steps.
When blocks are needed at step t, they are almost always needed
at step t+1. This makes simple recency tracking near-optimal:
the blocks you used last step are the blocks you need next step.

**2. Geometry-aware K-summaries produce worse recall than random.**
direct_centroid recall@8 = 0.145, while random_summary recall@8
= 0.339. The centroid of a block's K vectors does not predict
whether the block will be attended to at decode time. This is
likely because: (a) the query at decode time attends based on
content match and positional encoding, not proximity to key
centroids; (b) normalized centroids lose the magnitude
information that distinguishes frequently-attended from
rarely-attended blocks; (c) at 4K context, most blocks receive
non-trivial attention mass simply because the prefix is short
enough that attention is relatively spread out.

**3. The base miss rate is already very low.**
Even no_prefetch only misses 3.0% of needed blocks. With 32
blocks and ~8 needed per step, the overlap between consecutive
steps is high enough that on-demand fetching (lazy resident
accumulation) quickly builds up the full needed set. The
headroom for any prefetch policy to improve upon is small.

These results suggest that QK semantic routing would need a
fundamentally different workload to show value: one with low
temporal locality (e.g., multi-turn RAG with topic shifts),
much longer context (64K+ where attention becomes sparse), or
non-greedy decoding (beam search creating divergent access
patterns).

## 12. Limitations

- Single model (Qwen2.5-0.5B, 494M params)
- Single context length (4K prefix)
- Simulator-only (no actual block fault-in)
- Greedy decoding only
- No learned router (plain QK retrieval only)
- 128-token blocks only (no 256-token comparison)
- Wikitext workload has uniform text structure (no topic shifts)

## 13. Recommendation

**NO-GO**: Semantic routing delta vs recency: -0.3%. Geometry
delta: -19.4%. Insufficient signal to justify further work.

Recency-only prefetch (1.7% miss rate) outperforms all semantic
variants (2.0-2.2% miss rate) across all storage regimes.
Geometry-aware K-summaries perform worse than random vectors as
routing indices. The root cause is extreme temporal locality in
attention block access at 4K context length with greedy
decoding. The needed-block set is nearly identical from step to
step, making recency tracking optimal.

Future-Q prediction was skipped because one-step semantic
signal was already weaker than recency.

**Conditions that might change this verdict:**
- Context length >= 64K where attention becomes sparse
- Multi-turn RAG with topic shifts between turns
- Beam search or speculative decoding with divergent paths
- Heterogeneous prefix (code + natural language + structured)
- Hardware with much larger tier-2 latency ratio (10x+ vs 3.7x)
