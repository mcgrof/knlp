# BPA v13 Branch Tree

## Outcome Summary

- bitter1-3: PASS@3% at all L (2048-32768), 0% learned
- bitter4: PASS@3% up to 8192, OOM at 16384+
- bitter5-7: FAIL everywhere (lossy compression / OOM)
- Best method: bitter3 (two-signal heuristic), 22% faster at 32K

## Branch Assignment: C — Compression operators too lossy

The frontier methods (bitter5/6/7) that use tiered compression
(MLA + KVSplice) all fail because the compression operators
introduce catastrophic reconstruction error. Random orthogonal
projections and segment averaging are not suitable approximations
for KV cache compression.

## Next Steps

### Branch C.1: Learned MLA projections (HIGH PRIORITY)

Replace random orthogonal projections with PCA-trained projections
calibrated on actual KV distributions. The MLAProjector currently
uses random QR-decomposed matrices. Instead:
- Collect K,V distributions from a calibration set
- Compute SVD/PCA of K and V independently
- Use top-r singular vectors as projection matrices
- This retains maximum variance in the latent space

Expected impact: MLA reconstruction error should drop significantly
since PCA captures the actual data distribution, not random subspace.

### Branch C.2: Better splice operators (MEDIUM PRIORITY)

Replace segment averaging with attention-weighted merging. Current
KVSplicer averages consecutive tokens uniformly. Instead:
- Weight each token's contribution by its V-norm
- Or use a small attention mechanism to produce segment summaries
- Preserve positional information via position-aware merging

### Branch C.3: Lighter router training (LOW PRIORITY)

bitter4/7 OOM at long sequences because router training does dense
decode. Fix: subsample training data (use only every Nth step),
or train on shorter sequences and transfer.

### Branch C.4: Quantization as compression (ALTERNATIVE)

Instead of low-rank projection or segment merging, compress KV
entries via quantization (INT8/INT4). This preserves positional
information better than rank reduction while still reducing memory.
Modern GPUs (A100, H100) have native INT8 matmul support.

### What NOT to pursue

- More sophisticated routing policies. The bottleneck is compression
  quality, not routing quality. bitter1-3 already achieve near-dense
  PPL with trivial heuristics.
- End-to-end differentiable training of routing+compression jointly.
  The router doesn't help when compression is destructive.
- Larger recency windows. At budget=0.9*L, the recency window
  (W_min=L/8) is already sufficient.

## Decision Matrix

```
Outcome at L=32K            -> Next v14 focus
---------------------------    -------------------------
bitter5/6/7 all FAIL [NOW]  -> C.1: PCA-trained MLA
  with random MLA               projections
                                C.2: Better splice operators

bitter4 OOM at 16K+ [NOW]   -> C.3: Subsample router
                                training data

bitter1-3 PASS [NOW]        -> Baseline established.
                                No further heuristic work.
```
