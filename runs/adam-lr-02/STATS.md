# adam-lr-02 Statistical Analysis
## AUC(log(val_ppl)) over Token Interval
AUC = integral of log(val_ppl) over tokens, computed via trapezoidal rule.
Lower AUC = better convergence efficiency.

| Config | Mean AUC | StdDev | 95% CI | N |
|--------|----------|--------|--------|---|
| Baseline | 1.40e+09 | 2.62e+07 | [1.36e+09, 1.42e+09] | 3 |
| Fisher p=1 | 1.27e+09 | 2.64e+07 | [1.25e+09, 1.30e+09] | 3 |
| Fisher p=2 | 1.37e+09 | 8.14e+07 | [1.28e+09, 1.48e+09] | 3 |
| Random shuffle (C1) | 1.32e+09 | 6.59e+06 | [1.31e+09, 1.32e+09] | 3 |
| Depth ramp (C2) | 1.37e+09 | 4.11e+07 | [1.31e+09, 1.40e+09] | 3 |
| Frozen@200 (C3) | 1.36e+09 | 5.50e+07 | [1.32e+09, 1.44e+09] | 3 |

## Final Validation PPL

| Config | Mean | StdDev | 95% CI | N | vs Baseline |
|--------|------|--------|--------|---|-------------|
| Baseline | 82.60 | 8.13 | [71.20, 89.64] | 3 |  |
| Fisher p=1 | 56.73 | 2.39 | [54.15, 59.90] | 3 | -31.3% |
| Fisher p=2 | 85.42 | 29.72 | [56.80, 126.38] | 3 | +3.4% |
| Random shuffle (C1) | 66.42 | 3.34 | [62.34, 70.51] | 3 | -19.6% |
| Depth ramp (C2) | 73.23 | 10.08 | [59.02, 81.26] | 3 | -11.3% |
| Frozen@200 (C3) | 75.08 | 18.47 | [61.87, 101.20] | 3 | -9.1% |

## Bootstrap Procedure
- 10,000 bootstrap resamples
- 95% confidence intervals (percentile method)
- Seeds used: [0, 1, 2]
