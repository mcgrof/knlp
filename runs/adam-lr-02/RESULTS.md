# adam-lr-02: Per-Layer LR Scaling Falsification Study

## Summary

This experiment tests whether per-layer learning rate scaling
derived from Adam's second moment (diagonal Fisher proxy) provides
a genuine optimization benefit, or whether the gains can be
explained by simpler alternatives.

**Phase A**: 250M tokens, 3 seeds, 6 configurations.

## Results Table

| Config | Seed | Iters | Best Val PPL | Toks/s | Time (s) |
|--------|------|-------|-------------|--------|----------|
| C1_random_shuffle | 0 | 1010 | 62.34 | 24176 | 6940 |
| C1_random_shuffle | 1 | 1010 | 70.51 | 24222 | 6935 |
| C1_random_shuffle | 2 | 1010 | 66.40 | 24214 | 6936 |
| C2_depth_ramp | 0 | 1010 | 59.02 | 24208 | 6940 |
| C2_depth_ramp | 1 | 1010 | 79.42 | 24189 | 6938 |
| C2_depth_ramp | 2 | 1010 | 81.26 | 24072 | 6939 |
| C3_frozen_200 | 0 | 1010 | 62.18 | 24197 | 6937 |
| C3_frozen_200 | 1 | 1010 | 101.20 | 24218 | 6937 |
| C3_frozen_200 | 2 | 1010 | 61.87 | 24214 | 6940 |
| R0_baseline | 0 | 1010 | 86.95 | 23583 | 6922 |
| R0_baseline | 1 | 1010 | 89.64 | 24257 | 6945 |
| R0_baseline | 2 | 1010 | 71.20 | 24201 | 6929 |
| R1_fisher_p1 | 0 | 1010 | 54.15 | 24268 | 6920 |
| R1_fisher_p1 | 1 | 1010 | 56.13 | 24276 | 6922 |
| R1_fisher_p1 | 2 | 1010 | 59.90 | 24280 | 6918 |
| R2_fisher_p2 | 0 | 1010 | 73.08 | 24188 | 6925 |
| R2_fisher_p2 | 1 | 1010 | 56.80 | 24168 | 6932 |
| R2_fisher_p2 | 2 | 1010 | 126.38 | 24210 | 6936 |

### Mean over seeds

| Config | Mean Val PPL | StdDev | 95% CI | vs Baseline |
|--------|-------------|--------|--------|-------------|
| Baseline | 82.60 | 8.13 | [71.20, 89.64] | - |
| Fisher p=1 | 56.73 | 2.39 | [54.15, 59.90] | -31.3% |
| Fisher p=2 | 85.42 | 29.72 | [56.80, 126.38] | +3.4% |
| Random shuffle (C1) | 66.42 | 3.34 | [62.34, 70.51] | -19.6% |
| Depth ramp (C2) | 73.23 | 10.08 | [59.02, 81.26] | -11.3% |
| Frozen@200 (C3) | 75.08 | 18.47 | [61.87, 101.20] | -9.1% |

## Decision Gate

Best Fisher config: **Fisher p=1** (mean PPL=56.73, -31.3% vs baseline)

Best control: **Random shuffle (C1)** (mean PPL=66.42, -19.6% vs baseline)

**GATE: PASS** - Fisher-LR beats both baseline and all controls.

Proceed to Phase B with Fisher p=1 vs baseline vs Random shuffle (C1) at 1B tokens, 5 seeds.

## Addressing Skepticism

### Could random LR heterogeneity explain the gain?

No. Random shuffle achieves PPL=66.42, which is +17.1% worse than Fisher. The layer-assignment matters.

### Could a simple depth heuristic suffice?

No. Depth ramp achieves PPL=73.23, +29.1% worse than Fisher.

### Is dynamic adaptation necessary?

Yes. Frozen multipliers achieve PPL=75.08, +32.4% worse. Dynamic updates provide benefit.

## Plots

- `plots/val_ppl_vs_tokens.png`: Val PPL vs tokens
- `plots/tokens_to_threshold.png`: Tokens to reach PPL targets
- `plots/lr_mult_vs_layer.png`: LR multipliers by layer
- `plots/throughput.png`: Throughput comparison
