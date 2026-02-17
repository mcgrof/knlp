# BPA v24 Branch Tree

## Current State

k* changes by only 0 while D grows by 4. k/D decreases (0.083 -> 0.071), consistent with O(1) scaling.

O(1) Verdict: **SUPPORTED**

## Decision Tree

```
v24 outcome: SUPPORTED
|
+-- A: O(1) SUPPORTED (RECOMMENDED PATH)
|   |
|   +-- A1: Move to 7B/8B model on H100
|   |   k* should remain ~2-4 for D=32-80.
|   |   kv_ratio will improve to ~0.27-0.28.
|   |   -> v25: 7B replication + bandwidth-bound latency proof
|   |
|   +-- A2: Formalize theory for publication
|   |   Accumulation bound + O(1) condition is novel.
|   |   -> v25+: write up as short paper / tech report
|   |
|   +-- A3: Develop cheap proxy for oracle ranking
|       Empirical oracle is correct but slow (D evals).
|       -> v25+: proxy from attention entropy + activation norms
|
+-- B: H100 experiments
    |
    +-- B1: Bandwidth-bound latency proof
    |   7B at batch=16-32, L=32K-64K should be bandwidth-bound.
    |   Predict 10-30% latency reduction from kv_ratio~0.27.
    |   -> v25: run on H100 cloud GPU
    |
    +-- B2: Fused INT4 attention kernel
        Eliminate 8% Python overhead with CUDA/HIP kernel.
        -> v25+: after latency proof
```

## H100 Decision Memo

**Recommendation: HIGH ROI for H100 experiments.**

Evidence supports k* ≈ O(1), meaning kv_ratio improves with model
size. A 7B model (D~32) should achieve ratio~0.27 with k*~2-4.
On a bandwidth-bound GPU like H100, this translates to real latency
gains (not just capacity gains as on W7900).

Priority experiments:
1. 7B oracle ranking + k-sweep (4-8h GPU time)
2. Throughput/latency benchmark at batch=16-32 (2-4h)
3. Fused INT4 kernel development (if latency gain confirmed)
