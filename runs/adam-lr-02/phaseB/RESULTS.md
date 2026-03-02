# Adam LR Per-Layer Fisher Scaling — Phase B Results (1B Tokens)

## Summary
Fisher-derived per-layer learning rate scaling provides NO benefit at 1B tokens.
The Phase A improvement (-31% at 250M tokens) vanishes as Adam's own second-moment
estimates converge, making external LR preconditioning redundant at scale.

## Final Results

| Config           | Seeds | Mean PPL | Std  | vs Baseline |
|-----------------|-------|----------|------|-------------|
| **Baseline**     | 5/5   | 37.17    | 0.58 | —           |
| Fisher p=1       | 5/5   | 37.69    | 0.23 | +1.4% worse |
| Random shuffle   | 4/4   | 37.57    | 1.00 | +1.1% worse |

## Individual Seeds

### Baseline (R0)
- s0: 38.34, s1: 36.74, s2: 37.00, s3: 36.92, s4: 36.84

### Fisher p=1 (R1)
- s0: 37.95, s1: 37.92, s2: 37.59, s3: 37.50, s4: 37.47

### Random Shuffle (C1)
- s0: 36.70, s1: 38.93, s2: 37.69, s3: 36.96

## Key Findings
1. Fisher and random shuffle are statistically indistinguishable — the Fisher
   signal specifically does not matter at 1B tokens.
2. Random shuffle has much higher variance (std 1.00) confirming it is a dice
   roll — some lucky assignments, some terrible.
3. Fisher is consistently mediocre (std 0.23) but neither approach beats baseline.
4. Phase A result (Fisher -31% at 250M) was real but transient — Adam self-corrects.

## Conclusion
Per-layer Fisher-scaled learning rates are a valid optimization for SHORT training
runs (<500M tokens) where Adam has not yet converged its own estimates. At production
scale (1B+ tokens), stock AdamW is sufficient and simpler.
