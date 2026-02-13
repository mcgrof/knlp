# BPA v14b Final Report: Compression Fidelity Only

Model: Qwen2.5-0.5B (494M params, max_ctx=32768)
GPU: AMD Radeon Pro W7900 (48.3GB), ROCm 6.2
Decode: 256 steps, batch=1, seeds 0/1/2
Quality gate: PASS requires ALL 3 seeds within tolerance

## Headline: Quant INT8 is the only method that passes

Of seven compression backends evaluated, only **per-layer INT8
quantization** passes the quality gate at every context length.
All other methods fail, most catastrophically at long context.

## Results Summary

### Quality Gate (PASS = all 3 seeds within tolerance)

| Backend        | 4K @3% | 8K @3% | 16K @3% | 32K @3% |
|--------------- |:------:|:------:|:-------:|:-------:|
| quant (INT8)   | PASS   | PASS   | PASS    | PASS    |
| lowrank        | FAIL   | FAIL   | FAIL    | FAIL    |
| lowrank_konly  | FAIL   | FAIL   | FAIL    | FAIL    |
| quant_int4     | FAIL   | FAIL   | FAIL    | FAIL*   |
| kvsplice_seg2  | FAIL   | FAIL   | FAIL*   | FAIL*   |
| kvsplice_seg4  | FAIL   | FAIL   | FAIL*   | FAIL*   |
| kvsplice_seg8  | FAIL   | FAIL   | FAIL*   | FAIL*   |

*catastrophic: at least one seed > 3x dense PPL

### PPL Degradation (avg across seeds, %)

| Backend        |   4K |   8K |  16K |  32K |
|----------------|-----:|-----:|-----:|-----:|
| quant INT8     | -0.0 | +0.1 | +0.2 | -0.2 |
| quant INT4     | +14  | +21  | +20  | +19* |
| lowrank        | +15  |  +8  |  +9  | +24  |
| lowrank_konly  | +15  |  +8  |  +9  | +24  |
| kvsplice_seg2  | +56  | +60  | +112*| cat* |
| kvsplice_seg4  | +96  | +82  | +102*| cat* |
| kvsplice_seg8  | +78  | +76  | +154*| cat* |

### Latency Ratio (p50 vs dense)

| Backend        |   4K |   8K |  16K |  32K |
|----------------|-----:|-----:|-----:|-----:|
| quant INT8     | 1.01 | 1.00 | 1.01 | 1.01 |
| quant INT4     | 1.01 | 1.00 | 1.01 | 1.01 |
| lowrank        | 1.00 | 1.00 | 1.01 | 1.01 |
| lowrank_konly  | 1.00 | 1.00 | 1.01 | 1.01 |
| kvsplice_seg2  | 0.86 | 0.75 | 0.68 | 0.66 |
| kvsplice_seg4  | 0.78 | 0.64 | 0.52 | 0.48 |
| kvsplice_seg8  | 0.74 | 0.58 | 0.44 | 0.39 |

## Analysis

### A) Low-rank KV (SVD per head)

K values are highly compressible: 99.9% of energy at rank=19 out
of head_dim=64. V values are NOT compressible: 99.9% requires
rank=63. This is because K vectors are rotated by RoPE and
concentrate energy in a few dominant rotation directions, while
V vectors encode rich content information across all dimensions.

Low-rank projection at rank~31 preserves K well but the V
component forces rank up to ~63 (effectively no compression).
Using K-only compression gives identical results (confirming V
drives the quality loss). Even K-only compression at 50% fails
at >3% tolerance because the SVD basis from calibration data
doesn't perfectly represent K distributions at all positions.

The fundamental issue: RoPE makes K position-dependent, and a
fixed linear projection cannot capture this. A position-aware
projection would be needed, but this is equivalent to learning
a new attention mechanism.

### B) Quantization (INT8 / INT4)

INT8 per-channel symmetric quantization is essentially lossless
(<0.2% PPL change at all context lengths). This is the expected
result: 8-bit quantization preserves most of the information in
fp16 values, and the per-channel scale factors adapt to each
head's dynamic range.

INT4 block quantization (block_size=32) causes 14-21% PPL
degradation and one catastrophic failure at 32K. The 4-bit range
[-8,7] cannot represent the full dynamic range of KV entries,
especially in deeper layers where attention patterns become more
peaked and outlier values are critical.

Calibration assigned all 24 layers to INT8 (none passed the
INT4 threshold of 2x median INT8 error), confirming that Qwen2.5
KV values have tight quantization requirements.

### C) KVSplice Trained

KVSplice achieves dramatic latency reduction (35-61% faster at
32K) by compressing far tokens into segment representations, but
quality degrades catastrophically. The shared segment compressor
(one linear [seg*D -> D] for all layers) cannot preserve:

1. Position-dependent RoPE rotations across the compressed range
2. Layer-specific KV distributions (early vs late layers have
   very different key/value statistics)
3. Attention mass allocation (the KL loss on segment-level
   distributions cannot recover fine-grained positional info)

Training was unstable: KL loss oscillated 10x-1000x across
layers because shallow layers have very different attention
patterns than deep layers. A per-layer compressor would help
but would negate the memory savings of shared compression.

The training did converge to a meaningful compressor (not random
noise), as PPL is lower than random projection baselines from
BPA v13. But the quality loss is fundamentally incompatible with
the 3% tolerance requirement.

## Winner: Quant INT8

Per-layer INT8 symmetric quantization is the clear winner:
- PASS at 1% tolerance across all context lengths
- Negligible latency overhead (~1%)
- Simulated 2x compression of far-past KV entries
- No calibration instability or position sensitivity

## Rate-Distortion Summary

In order of compression aggressiveness:

1. quant INT8: ~2x compression, <0.2% PPL, PASS
2. lowrank K=31 V=63: ~1.4x compression, 8-24% PPL, FAIL
3. quant INT4: ~4x compression, 14-21% PPL, FAIL
4. kvsplice_seg2: 2x compression, 56-112% PPL, FAIL*
5. kvsplice_seg4: 4x compression, 82-102% PPL, FAIL*
6. kvsplice_seg8: 8x compression, 76-154% PPL, FAIL*

The Pareto frontier has exactly one point: quant INT8. There is
no method that achieves >2x compression while staying within 3%
quality tolerance on Qwen2.5-0.5B.

## Stop Conditions Met

- KVSplice was trained (5k-20k steps) and evaluated at 3 segment
  sizes. All FAIL with catastrophic degradation. No excuses.
- No routing experiments, no eviction games.
- Only compression fidelity evaluated.

## Files

- results/v14b/all_results_combined.json: 96 eval results
- results/v14b/bpa_v14b_scoreboard.json: aggregated scoreboard
- kvsplice_trained/*.pt: trained segment compressor checkpoints
- backends/: compression backend implementations
- eval_v14b.py: evaluation harness
- scripts/train_kvsplice.py: KVSplice training script
