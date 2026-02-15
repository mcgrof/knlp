# BPA v21 Branch Tree: Outcome-Based Decision

## Observed Outcome

v21 hypothesis (g=4 + k<4 beats S2_k6) is FALSE due to scale
overhead. But g=32 k=4 with theory ranking BEATS S2_k6 (0.320
vs 0.333).

```
v21 Exploit g=4, beat kv_ratio=0.33
├── Phase 0: True byte accounting
│   ├── g=4 scale overhead = 50% → kv_ratio=0.50 (WORSE than INT8)
│   ├── g=8 scale overhead = 33% → kv_ratio=0.375 (worse than S2)
│   └── g=32 scale overhead = 11% → kv_ratio=0.281-0.340
├── Phase 2: Grid search (g×k)
│   ├── g=32 k=4 → PASS@3%, ratio=0.320 → BEATS S2_k6!
│   ├── g=32 k=3 → FAIL (+4.25%), ratio=0.311
│   ├── g=8 k=2 → PASS@3%, ratio=0.387 (too high)
│   └── g=4 k=2 → PASS@3%, ratio=0.501 (useless)
├── Phase 3: k-floor attribution
│   └── k=3→k=4 removes layer 2 (sigma=1.44), crossing threshold
└── Winner: g=32 k=4 theory ranking → kv_ratio=0.3203 PASS@3%
```

## Why g=32 k=4 Beats S2_k6

S2_k6 from v16 used 6 INT8 layers selected by empirical per-layer
PPL ablation. g=32 k=4 uses only 4 INT8 layers selected by theory
ranking (residual norm measurement from v20). The improvement comes
from better layer selection, not from any novel quantization method.

Theory ranking [0,8,1,2] protects the 4 layers with highest residual
distortion under INT4. The v19 empirical ranking [0,2,11,8] includes
layer 11 (rank 3) which has lower residual norm than layer 1 (theory
rank 3). The theory ranking is strictly better because it measures
the actual signal that matters (KV reconstruction error) rather than
a proxy (per-layer PPL delta, which conflates multiple effects).

## v22 Action Plan

### Path A: Transfer to larger models
- Test Qwen2.5-1.5B and Qwen2.5-7B with the same g=32 + theory
  ranking approach. More layers means each layer contributes
  proportionally less noise, potentially reducing k*/D ratio.
- Predict k* for each model using the accumulation theory from v20.
- Target: kv_ratio < 0.30 on 7B (would require k*/D < 0.15).

### Path B: Amortized scales for small group sizes
- g=4 noise reduction is real (8x) but scale overhead kills it.
- If scales are shared across S tokens (segment-level), the per-token
  scale cost drops by 1/S. At S=8 tokens per segment, g=4 scale
  overhead drops from 50% to ~8%, making kv_ratio ~0.30.
- Risk: segment-level scales may lose per-token precision for RoPE
  position-dependent K values.

### Path C: INT8 scale representation
- Replace fp16 scales with INT8 scales, halving scale overhead.
- g=4 with INT8 scales: overhead drops from 50% to 33%.
- Still not competitive (kv_ratio ~0.375 vs g=32 k=4 at 0.320).
- Could combine with Path B for aggressive scale compression.

### Recommended: Path A (transfer to larger models)
The g=32 k=4 recipe with theory ranking is the Pareto-optimal point
for Qwen2.5-0.5B. The remaining compression headroom (0.320 → 0.28)
requires k<4, which is architecturally blocked. Larger models with
more layers offer the best path to lower kv_ratio.
