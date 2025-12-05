# KV Cache Compression: Breaking Point Analysis

**UPDATE (v6)**: The cache bypass bug described below has been fixed.
See [Post-Bugfix Report](kv_plugin_v6_postbugfix.md) for validation results.

## Executive Summary

Direct measurement of KV cache compression quality reveals that **V tensors
are significantly more sensitive to compression than K tensors**. The failure
boundary occurs much earlier than previously reported PPL measurements
suggested, because the plugin's inference path was not actually using
compressed values.

## Critical Finding: Plugin Architecture Issue

**The PPL measurements in earlier benchmarks were invalid.** The
`CompressedAttentionWrapper` compresses K/V on write but does not expand
from the compressed cache on read. HuggingFace's `DynamicCache` bypasses
our compressed storage, meaning inference runs with uncompressed values
while we measure theoretical compression ratios.

This explains why PPL showed 0% degradation even at "896x compression" -
no compression was actually happening during the forward pass.

## Direct Compression Quality Results (Qwen2.5-0.5B)

### Reconstruction Quality vs Rank

| Rank | Ratio | K MSE | K CosSim | V MSE | V CosSim |
|------|-------|-------|----------|-------|----------|
| 64 | 1.0x | 0.000 | 1.000 | 0.000 | 1.000 |
| 32 | 2.0x | 1.297 | 0.907 | 0.336 | **0.741** |
| 16 | 4.0x | 2.789 | 0.788 | 0.563 | **0.512** |
| 8 | 8.0x | 4.896 | 0.591 | 0.671 | 0.353 |
| 4 | 16.0x | 9.488 | 0.450 | 0.718 | 0.236 |
| 2 | 32.0x | 26.67 | 0.349 | 0.739 | 0.167 |
| 1 | 64.0x | 56.34 | 0.207 | 0.750 | 0.115 |

### Key Observations

1. **V degrades 2x faster than K**: At 2x compression, K retains 91%
   similarity while V drops to 74%.

2. **Quantization has minimal impact**: Int8 and int4 quantization add
   <1% additional error on top of rank reduction.

3. **Head dimension limits compression**: Qwen2.5-0.5B has head_dim=64,
   so maximum theoretical compression is 64x (rank 1).

## Failure Boundaries

### Cosine Similarity Thresholds

| Threshold | K Failure | V Failure |
|-----------|-----------|-----------|
| 99% | rank < 64 (any compression) | rank < 64 |
| 95% | rank ≤ 32 (2x) | rank ≤ 64 (1x) |
| 90% | rank ≤ 16 (4x) | rank ≤ 64 (1x) |
| 75% | rank ≤ 8 (8x) | rank ≤ 32 (2x) |
| 50% | rank ≤ 4 (16x) | rank ≤ 16 (4x) |

### Recommended Safe Zones

Based on cosine similarity ≥ 0.90 for K and ≥ 0.70 for V:

| Zone | Rank Range | Compression | Expected Quality |
|------|------------|-------------|------------------|
| **Safe** | 48-64 | 1.0-1.3x | Minimal degradation |
| **Moderate** | 32-48 | 1.3-2.0x | V at 70-80% fidelity |
| **Aggressive** | 16-32 | 2.0-4.0x | Noticeable degradation |
| **Extreme** | <16 | >4.0x | Significant quality loss |

## Comparison to Literature Claims

### Palu (ICLR 2025)
- Claims 6x compression with <2% PPL loss
- Our data: 6x would require rank ~11, giving V CosSim ~0.45
- This would cause significant PPL degradation if properly measured

### AsymKV (NeurIPS 2025)
- Advocates asymmetric K/V compression (compress V more)
- Our data **contradicts this**: V is more sensitive, not less
- K tolerates compression better than V

### MiniCache (NeurIPS 2024)
- Uses cross-layer KV merging
- Different approach, not directly comparable

## Implications

### For Plugin Development

1. **Fix inference path**: The `CompressedAttentionWrapper` must expand
   from compressed cache on read, not just compress on write.

2. **Recalibrate presets**: Current "balanced" and "aggressive" presets
   assume much higher compression is safe than reality shows.

3. **Separate K/V compression**: Given V's higher sensitivity, consider
   asymmetric compression with K compressed more than V (opposite of
   AsymKV's recommendation).

### For Production Deployment

1. **Conservative compression only**: Until plugin is fixed and
   properly validated, recommend rank ≥ 48 (1.3x compression max).

2. **Monitor reconstruction error**: Add runtime checks for cosine
   similarity to catch quality degradation.

3. **Task-specific validation**: Compression tolerance may vary by
   task - test on target workload before deployment.

## Methodology

### Direct Compression Test

```python
# Extract real K/V from forward pass
kv_pairs = extract_kv_from_model(model, input_ids)

# Compress and expand at each rank
for rank in [64, 32, 16, 8, 4, 2, 1]:
    compressor = OrthogonalCompressor(d_compressed=rank)
    compressor.calibrate(kv_data)

    k_compressed = compressor.compress(k)
    k_reconstructed = compressor.expand(k_compressed)

    # Measure quality
    cos_sim = cosine_similarity(k, k_reconstructed)
```

### Why PPL Was Invalid

```
[Inference Path - EXPECTED]
Input → Attention → Compress K/V → Store → Expand → Attention → Output

[Inference Path - ACTUAL]
Input → Attention → Store in DynamicCache → Attention reads from DynamicCache → Output
                 ↓
        Compress K/V → Store in our cache (UNUSED)
```

The wrapper stores compressed values but HuggingFace's cache management
reads from its own `DynamicCache`, bypassing our compressed storage.

## Next Steps

1. **Fix plugin architecture** to properly intercept cache reads
2. **Re-run PPL benchmarks** with fixed inference path
3. **Validate against downstream tasks** (not just perplexity)
4. **Consider hybrid approach**: compress V less than K

## Files

- `scripts/kv_compression_direct_test.py` - Direct quality measurement
- `results/direct_compression_test.json` - Raw results
- `plots/compression_failure_curves/` - Visualizations (TODO)
