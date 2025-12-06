## Quality Benchmark Results (Qwen2.5-7B on B200)

**Baseline PPL**: 7.8849

### V-Only Calibrated Compression

| Rank | Compression | Savings | PPL | ΔPPL |
|------|-------------|---------|-----|------|
| 120 | 1.03x | 3.1% | 8.35 | +5.9% |
| 112 | 1.07x | 6.2% | 8.99 | +14.0% |
| 96 | 1.14x | 12.5% | 10.61 | +34.6% |
| 64 | 1.33x | 25.0% | 58.89 | +647% |

### Calibrated vs Random Orthogonal (Rank 112)

| Method | PPL | ΔPPL |
|--------|-----|------|
| Calibrated PCA | 8.99 | +14.0% |
| Random orthogonal | 9.09 | +15.2% |

Calibration provides ~8% relative improvement over random projections.

### Key Findings

1. **Calibration helps modestly**: ~8% relative improvement over random orthogonal
2. **V-only is essential**: Compressing K causes catastrophic quality degradation
3. **Conservative compression only**: Rank 120 (+6% PPL) is the practical limit
4. **Aggressive compression fails**: 2x compression (rank 64) causes +647% PPL
