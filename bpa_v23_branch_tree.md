# BPA v23 Branch Tree

## Current State

Two sub-0.30 PASS configs achieved:
- 0.5B: amort_g8_S8_k3 (ratio=0.2969, +2.38%)
- 1.5B: g32_k2_oracle (ratio=0.2974, +1.05%)

k* scales with model depth: k*=4 (0.5B, D=24) -> k*=2 (1.5B, D=28).
W7900 still compute-bound. Capacity gain: 3.26x concurrent sequences.

## Decision Tree

```
v23 outcome
|
+-- A: Sub-0.30 achieved on both 0.5B and 1.5B
|   |
|   +-- A1: Push to 7B/8B model (RECOMMENDED)
|   |   k/D scaling predicts k*~2 for 7B (D~32), ratio~0.281.
|   |   Need H100 or quantized-weight 7B on W7900.
|   |   Oracle ranking required (proxy unreliable).
|   |   -> v24: 7B replication + bandwidth-bound proof
|   |
|   +-- A2: Fuse INT4 + amortized-scale kernel
|   |   Current backend is pure-PyTorch (8% overhead).
|   |   Custom HIP/CUDA kernel eliminates dequant overhead.
|   |   Worth doing only after bandwidth-bound proof.
|   |   -> v24+: kernel optimization after latency evidence
|   |
|   +-- A3: Push k* below 2 on 1.5B
|       Only layer 0 (sink, +824%) and layer 15 (+3.2%)
|       need INT8. Layer 15 is marginal at 3.2%.
|       Learned calibration for layer 15 might reduce to k*=1.
|       But layer 0 is structurally uncompressible.
|       -> v24+: learned calibration for marginal layers
|
+-- B: k/D scaling confirmed
|   |
|   +-- B1: Extrapolate to production models (RECOMMENDED)
|   |   At D=56 (70B): k/D=2/56=0.036, ratio=0.274.
|   |   At D=80 (405B): k/D=2/80=0.025, ratio=0.268.
|   |   Larger models benefit more from INT4 KV compression.
|   |   -> v24: validate on 7B/13B with oracle ranking
|   |
|   +-- B2: Develop cheap proxy for oracle ranking
|       Empirical oracle is correct but slow (D evals).
|       Needed: fast proxy that correctly identifies sink +
|       one or two secondary sensitive layers.
|       Candidate: attention entropy of layer 0 (known sink)
|       plus activation magnitude for remaining layers.
|       -> v24+: proxy development after more model data
|
+-- C: W7900 still compute-bound
|   |
|   +-- C1: H100 bandwidth-bound experiment (RECOMMENDED)
|   |   H100 has 3.35 TB/s bandwidth. At batch=16-32 and
|   |   L=32K-64K on 7B, attention should become bandwidth-
|   |   bound. KV byte cuts then translate to latency gains.
|   |   -> v24: H100 run with 7B + amort_g8_S8
|   |
|   +-- C2: Consumer GPU experiment
|   |   RTX 4090 (1 TB/s, narrower bus) may hit bandwidth
|   |   limits sooner with 7B + long context.
|   |   -> v24: alternative to H100 if cloud credits unavailable
|   |
|   +-- C3: Accept capacity-only story
|       3.26x more concurrent sequences is valuable for
|       production serving even without latency improvement.
|       Report throughput (sequences/GPU/sec) as primary metric.
|       -> Any version: reframe for production deployment
|
+-- D: Amortization knobs well-characterized
    |
    +-- D1: g=8 S=8 is the sweet spot
    |   g=4 shows RoPE drift at small S.
    |   g=8 S=16 fails (scale drift too large).
    |   Scale mode (pre/post/norm) doesn't matter for g=8.
    |   -> Settled: use g=8 S=8 post_rope going forward
    |
    +-- D2: k=3 works on 0.5B (was k*=4 before)
        amort_g8_S8_k3 passes at ratio=0.2969.
        The amortization overhead reduction allows k*=3.
        This suggests scale overhead contributed to the
        apparent k*=4 floor in earlier versions.
        -> Understanding: k-floor is partially overhead-driven
```

## Recommended v24 Plan

Priority order:
1. Replicate on 7B model (Qwen2.5-7B or Llama-3.1-8B) with
   empirical oracle ranking. Predict k*~2, ratio~0.28.
2. Run on H100 or consumer GPU to demonstrate bandwidth-bound
   latency gains from KV compression.
3. If bandwidth-bound: show ms/token improvement. If not:
   strengthen the capacity/throughput story with production
   serving benchmarks.
4. Fuse INT4+amortized-scale kernel for zero-overhead decode.
