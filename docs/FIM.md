# Fisher Information Matrix (FIM) in Attention

## What is the Fisher Information Matrix?

The Fisher Information Matrix (FIM) measures how much information an observation
carries about an unknown parameter. In the context of attention mechanisms, it
quantifies the sensitivity of attention distributions to changes in the input.

Think of it as a "curvature map" of the optimization landscape. High curvature
means small changes in parameters cause large changes in outputs (sensitive,
hard to optimize). Low curvature means the landscape is flatter (stable, easier
to optimize).

## Why We Care About FIM in Attention

The SPDA paper ("Scaled Dot-Product Attention as One-Sided Entropic Optimal
Transport") proves that attention solves an Entropic Optimal Transport (EOT)
problem. The attention scores enter a log-sum-exp potential, and the Hessian
(second derivative) of this potential is exactly the Fisher Information Matrix.

For attention with probabilities `p = softmax(scores / τ)`:

```
FIM = (1/τ²) * (diag(p) - p * p^T)
```

This matrix tells us:
- **How sensitive** the attention distribution is to perturbations
- **How curved** the optimization landscape is at this point
- **Which directions** in token space carry the most information

## FIM Metrics We Track

We log several FIM metrics during training to understand attention geometry:

### eigmax (Maximum Eigenvalue)

**What it is**: The largest eigenvalue of the Fisher Information Matrix.

**What it means**: The sharpest curvature direction in the optimization
landscape. High eigmax = sharp peaks/valleys = sensitive to small changes.
Low eigmax = flatter landscape = more stable optimization.

**Why it matters**: Learning rate needs to be smaller than 2/eigmax for
stability. Lower eigmax allows larger learning rates and more stable training.

**FIM eigmax (mean)**: Average of eigmax across all heads in a layer, computed
over the last 100 training samples. This gives a per-layer summary of
optimization difficulty.

### trace (Total Fisher Information)

**What it is**: Sum of all eigenvalues of the FIM.

**What it means**: Total amount of information in the attention distribution
across all directions. Think of it as the "total curvature mass."

**Why it matters**: Higher trace means more total information, but also more
directions that need careful optimization.

### energy_r8, energy_r16 (Energy Concentration)

**What it is**: Fraction of total Fisher energy (trace) captured in the top 8
or 16 eigenmodes.

**What it means**: How concentrated the information is. High energy
concentration (e.g., 90% in top 8 modes) means most information is in a few
directions. Low concentration (e.g., 37% in top 16 modes) means information is
diffuse across many directions.

**Why it matters**: High concentration suggests we could compress effectively
using low-rank approximations (keep top-k modes, discard the rest). Low
concentration means compression is harder—information is spread out.

### decay (Spectral Concentration)

**What it is**: Ratio of maximum eigenvalue to the 5th eigenvalue (eigmax / λ_5).

**What it means**: How quickly eigenvalues drop off. High decay = eigenvalues
drop fast = information concentrated in top modes. Low decay = eigenvalues
decrease slowly = information spread across many modes.

**Why it matters**: Another indicator of whether low-rank compression will work
effectively.

## Why We Added FIM Metrics

We initially hypothesized that Reciprocal Attention (RA) might concentrate
Fisher Information into fewer modes, making attention more compressible. The
SPDA theoretical framework suggested that alternating Q@K.T and K@Q.T could
change the information geometry.

**Hypothesis**: RA produces higher energy concentration (energy_r16 → 1.0),
enabling better low-rank compression of the KV cache.

**Result**: Hypothesis rejected. RA shows:
- **Lower eigmax** (0.0352 vs higher in MLA) = flatter optimization, not sharper
- **Similar trace** = no increase in total Fisher information
- **Low energy concentration** (37% in top 16 modes) = information diffuse, not concentrated

The FIM metrics revealed that RA's value comes from **smoother optimization
geometry** (lower eigmax, easier training), not from concentrating information
into compressible modes.

## Mathematical Introspection: What Does RA Add?

**Motivation**: The SPDA result provides a principled way to analyze attention
geometry. If alternating Q@K.T and K@Q.T changes the optimization landscape,
FIM metrics should reveal how.

**Research Question**: Does RA change the Fisher Information geometry in ways
that enable better cache compression?

### FIM Data from Experiments

Source: test_matrix_results_20251123_231956 (W&B project:
gpt2-kvsplice-ablation-w7900-mla-fixed)

| Architecture | eigmax | energy_r8 | energy_r16 | Interpretation |
|-------------|--------|-----------|------------|----------------|
| RA+MLA | 0.0352 | 0.223 | 0.373 | Low concentration, flat curvature |
| RA+MLA+KVSplice | 0.0341 | 0.220 | 0.370 | Similar geometry, compression orthogonal |

**Key Findings**:

1. **Low energy concentration**: Only ~37% of Fisher energy in top 16 modes
   across all architectures. Would need r>16 to capture 90% energy.

2. **No FIM improvement from compression**: KVSplice shows nearly identical
   energy concentration (0.370 vs 0.373) despite 50% cache reduction.

3. **Slight eigmax reduction**: KVSplice has marginally lower eigmax (0.0341 vs
   0.0352), suggesting flatter curvature, but effect is small.

### Conclusion

**FIM analysis did not provide clear guidance for compression decisions**.
Despite low energy_r16 (~0.37), KVSplice with d=128 (50% compression)
empirically improves quality by 11%. The learned compression appears to find
task-specific structure that FIM-based metrics don't capture.

**Interpretation**: Fisher Information measures optimization geometry, not
necessarily task-relevant information. Learned compression (KVSplice) acts as
beneficial regularization that forces representations into information-dense
subspaces, but this structure isn't visible in variance-based or FIM-based
metrics. The value comes from end-to-end learning, not from following
prescribed compression directions indicated by FIM.

## Relationship to RA's Inductive Bias

RA (Reciprocal Attention) alternates between forward and reverse EOT problems:

```
F_fwd  from softmax(Q * K^T / τ)    # Forward geometry
F_rev  from softmax(K * Q^T / τ)    # Reverse geometry
```

Each layer experiences one geometry, alternating across depth. The FIM metrics
revealed that this alternation produces:

- **Flatter curvature** (lower eigmax) = easier optimization
- **Better gradient flow** = compensates for compression losses in MLA
- **More stable training** = particularly helpful for compressed representations

But RA does **not** produce:
- Higher total Fisher Information (trace similar/lower)
- Concentrated information modes (energy_r16 remains ~37%)
- Improved compressibility from geometric changes alone

This explains why RA helps MLA (optimization benefits) but doesn't predict or
enable further compression (no structural changes to information geometry).

## Practical Implications

**For RA usage**:
- Expect optimization benefits (flatter curvature, better gradient flow)
- Do not expect FIM-guided compression opportunities
- Learned compression (KVSplice) works independently of RA's geometric properties

**For compression research**:
- FIM metrics don't predict learned compression effectiveness
- Task-specific information structure differs from geometric information (FIM)
- End-to-end learning finds compressible structure that variance/FIM miss

## References

- SPDA Paper: "Scaled Dot-Product Attention as One-Sided Entropic Optimal Transport"
- Code for FIM metrics: `gpt2/trainers/ra.py` (FisherMetricsCallback)
- Experimental results: [test_matrix_results_20251123_231956](https://github.com/mcgrof/knlp-key-results/tree/main/key_results/test_matrix_results_20251123_231956)
