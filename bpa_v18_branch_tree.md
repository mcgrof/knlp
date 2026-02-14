# BPA v18 Decision Tree

## Branch Points

```
START: Can Adam v-hat predict KV quantization sensitivity?
  |
  +-- Phase 0: Baselines reproducible?
  |     YES -> Continue
  |
  +-- Phase 1: Does v-hat show meaningful layer variation?
  |     YES -> 1467x dynamic range, layer 0 dominant
  |     Root4 compresses to 6.2x, preserves ranking
  |
  +-- Phase 2: Does adam-guided allocation beat random?
  |     |
  |     +-- At k=2 on 0.5B:
  |     |     adam: PASS@3% at L={8K,32K}, FAIL at 16K
  |     |     random: FAIL everywhere (catastrophic at 32K)
  |     |     -> YES, adam >> random at tight budget
  |     |
  |     +-- Does adam match S2 manual?
  |           S2 k=6: FULL PASS@1% everywhere
  |           adam k=2: partial PASS@3% only
  |           -> NO, adam needs more INT8 layers
  |
  +-- Phase 3: How well does adam correlate with empirical?
  |     rho=0.40, p=0.052 (borderline significant)
  |     Critical miss: layer 8 ranked #13 by adam (#2 empirical)
  |     -> MODERATE signal, not sufficient for tight allocation
  |
  +-- Phase 4: Does signal generalize to 7B?
  |     adam_root4 k=6 vs random k=6 at L=8K:
  |       adam: +1.40%  random: +1.38%
  |     -> NO, indistinguishable from random at generous budget
  |     -> Budget k=6/28 is too loose to test signal quality
  |
  +-- Phase 5: Does KV reduction produce latency gain?
  |     W7900: <0.3% delta across all regimes
  |     -> NO, hardware is compute-bound
  |
  +-- Phase 6 (Fisher ablation): SKIPPED (time)
  |
  CONCLUSION: CASE B
    Adam v-hat is cheap and better than random, but inferior
    to empirical sweeps. Not a replacement, but a reasonable
    first-pass heuristic for conservative allocation.
```

## Headline Mapping

| Case | Description | v18 Status |
|------|-------------|------------|
| A | Adam matches S2 | NOT achieved |
| B | Adam worse but cheaper | ACHIEVED |
| C | Adam signal weak | Partially (7B) |
| D | 8B confirms scaling | NOT achieved |

## Key Numbers

| Metric | Value |
|--------|-------|
| Spearman rho (adam vs empirical) | 0.40 |
| p-value | 0.052 |
| Minimal k (0.5B) | 2 |
| Minimal k (7B) | 6 |
| adam@8K PPL delta (0.5B) | +1.25% |
| S2@8K PPL delta (0.5B) | +0.30% |
| random@8K PPL delta (0.5B) | +21.0% |
| KV bytes ratio (adam k=2) | 0.32 |
| KV bytes ratio (S2 k=6) | 0.36 |
| Latency delta (W7900) | <0.3% |
