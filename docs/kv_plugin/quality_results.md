## Quality Benchmark Results (Qwen2.5-7B on B200)

### Summary

| Metric | Rank 120 | Rank 96 |
|--------|----------|---------|
| Memory Savings | 3.1% | 12.5% |
| Throughput Penalty | -13% | -14% |
| PPL Increase | +6% | +35% |
| Task Degradation | 0% | Unknown |

### Perplexity Results

**Baseline PPL**: 7.8849

| Rank | Compression | PPL | ΔPPL |
|------|-------------|-----|------|
| 120 | 1.03x | 8.35 | +5.9% |
| 112 | 1.07x | 8.99 | +14.0% |
| 96 | 1.14x | 10.61 | +34.6% |
| 64 | 1.33x | 58.89 | +647% |

### Task Performance (lm-eval)

200 samples per task, zero-shot:

| Task | Baseline | Compressed (r120) | Delta |
|------|----------|-------------------|-------|
| HellaSwag | 68.0% | 68.0% | 0.0% |
| ARC-Easy | 76.0% | 76.0% | 0.0% |
| WinoGrande | 78.0% | 78.0% | 0.0% |
| PIQA | 83.0% | 83.0% | 0.0% |

**Key finding**: +6% PPL increase shows 0% task degradation. Perplexity and
task performance don't correlate linearly at conservative compression levels.

### Calibrated vs Random

| Method | Rank 112 PPL | ΔPPL |
|--------|--------------|------|
| Random orthogonal | 9.09 | +15.2% |
| Calibrated PCA | 8.99 | +14.0% |

Calibration provides ~8% relative improvement.
