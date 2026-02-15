# Accumulation Theory: k-floor Bound

## Model

Quantization introduces per-layer noise epsilon_l to KV cache.
The total output distortion accumulates approximately additively
across layers (confirmed empirically in v15):

  Delta_total ≈ sum_l w_l * epsilon_l

where w_l ≈ 1 (amplification factor, close to unity by v12 finding).

## Noise Estimates

| Precision | Total noise (RMS) | Per-layer mean |
|-----------|-------------------|----------------|
| INT4      | 16.8812        | 0.703385           |
| INT8      | 1.3961        | 0.058170           |

INT4/INT8 noise ratio: 12.1x

## Per-layer sigma (top 6 by delta):

| Layer | sigma4    | sigma8    | delta     |
|-------|-----------|-----------|-----------|
|     0 | 4.306610 | 0.268596 | 4.038014 |
|     8 | 2.409251 | 0.342527 | 2.066724 |
|     1 | 1.946713 | 0.234471 | 1.712242 |
|     2 | 1.550453 | 0.106902 | 1.443551 |
|     3 | 0.575639 | 0.038192 | 0.537446 |
|     4 | 0.472757 | 0.032773 | 0.439984 |

## Fitted Bound

Linear relationship between remaining noise and PPL degradation:

  max_delta_pct ≈ 26.6505 * remaining_noise + -186.5985

For PASS@3%: need remaining_noise <= 7.1143
For PASS@1%: need remaining_noise <= 7.0392

## Predicted vs Observed k*

| Tolerance | Predicted k* | Observed k* |
|-----------|-------------|-------------|
| 3%        | 5           | 3           |
| 1%        | 6           | 3           |

## Implications

To reduce k*, interventions must either:
1. Reduce sigma4 per layer (Path B: better quantization)
2. Reduce the number of layers contributing noise (Path A: selective protection)
3. Reduce amplification w_l (Path C: structural)

The theory predicts that sigma4 reduction by factor X reduces k* by
approximately X layers (since the top layers dominate).
