# BPA v22 Branch Tree

## Current State

Winner: amort_g8_S8_k4 (kv_ratio=0.3073, max_delta=+2.43%, PASS@3%)
- Beats v21 best (g32_k4, 0.3203) by 4.1%
- Beats S2_k6 (0.333) by 7.7%

k-floor remains k*=4. Bandwidth-bound regime not reached on W7900.
1.5B replication needs model-specific sensitivity (proxy fails).

## Decision Tree

```
v22 outcome
|
+-- A: Scale amortization works (kv_ratio=0.3073, PASS)
|   |
|   +-- A1: Push to sub-0.30 on 0.5B
|   |   Requires k<=3, which fails at +4.25% (g32_k3).
|   |   Scale amortization doesn't change k*.
|   |   DEAD END for 0.5B without new mechanism.
|   |
|   +-- A2: Port amort_g8_S8 to 1.5B/7B (RECOMMENDED)
|   |   k/D shrinks: k=4/28=0.143 gives ratio=0.296 (sub-0.30).
|   |   Need proper sensitivity sweep (per-layer PPL ablation,
|   |   not residual_ratio proxy which fails catastrophically).
|   |   W7900 can run 1.5B; 7B needs INT4 weights.
|   |   -> v23: amort_g8_S8 + model-specific sensitivity on 1.5B
|   |
|   +-- A3: Fuse amortized INT4 kernel for latency
|       Current implementation is pure-PyTorch simulation.
|       Custom HIP/CUDA kernel could decode INT4+amortized scales
|       in a single pass. Worth doing only if bandwidth-bound
|       regime is reached on target hardware.
|       -> v23 or later: kernel work after bandwidth proof
|
+-- B: Residual_ratio proxy fails on 1.5B (+67960%)
|   |
|   +-- B1: Use per-layer PPL ablation instead (RECOMMENDED)
|   |   Run each layer individually at INT4 and measure PPL.
|   |   This is what v19/v20 did for 0.5B (produced the oracle
|   |   ranking). Slower but correct.
|   |   -> v23: empirical sensitivity sweep on 1.5B
|   |
|   +-- B2: Fix the proxy
|       Residual_ratio measures reconstruction error magnitude.
|       Layer 0 (sink) has small absolute values but huge
|       functional importance. A better proxy would measure
|       attention weight divergence or output logit KL.
|       Lower priority than B1 (empirical always wins).
|       -> v24+: better proxy as convenience optimization
|
+-- C: Still compute-bound on W7900
|   |
|   +-- C1: Move to H100 or consumer GPU (RECOMMENDED)
|   |   H100 has 3.35TB/s bandwidth vs W7900's 864GB/s.
|   |   Paradoxically, consumer GPUs (RTX 4090: 1TB/s,
|   |   narrower bus) may hit bandwidth limits sooner with
|   |   large batch + long context.
|   |   -> v23: H100 or RTX 4090 bandwidth experiment
|   |
|   +-- C2: Use much longer context (64K-128K)
|   |   Qwen2.5-1.5B supports 128K context.
|   |   At L=128K the KV cache is 4x larger than at 32K.
|   |   May shift the compute/bandwidth balance.
|   |   -> v23: L=64K/128K on 1.5B
|   |
|   +-- C3: Accept compute-bound finding
|       For production deployment, the memory savings from
|       kv_ratio=0.3073 are valuable even without latency
|       improvement: more sequences can run concurrently.
|       Report throughput (sequences/GPU) instead of latency.
|       -> Any version: reframe metric as throughput gain
|
+-- D: k*=4 is architectural (unchanged from v21)
    |
    +-- D1: Accept k*=4, optimize around it
    |   Best path: larger models where k/D shrinks.
    |   At D=56 (70B): k/D=4/56=0.071, ratio=0.286.
    |
    +-- D2: Learned scale calibration (B3 from v21)
        For the top-4 sensitive layers, learn optimal INT4
        scales on calibration data. Targets sigma reduction
        at the layers that matter most. Could reduce k* to 3
        if layer 0 sigma drops below threshold.
        -> v24+: only if B1+A2 fails to reach sub-0.30
```

## Recommended v23 Plan

Priority order:
1. Port amort_g8_S8_k4 to Qwen2.5-1.5B with per-layer PPL
   sensitivity sweep (not residual_ratio proxy).
2. Test at L=32K and L=64K if memory allows.
3. Profile bandwidth on 1.5B at batch=8+ (larger KV cache
   may shift compute/bandwidth balance).
4. If 1.5B passes at k=4 with ratio<0.30, that's the headline.
5. If bandwidth-bound: show latency win. If not: report
   throughput (concurrent sequences) win.
