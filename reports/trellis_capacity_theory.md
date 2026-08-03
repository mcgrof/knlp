# Trellis capacity theory audit

This report classifies the theoretical claims behind a proposed
Hopfield/Krotov-style capacity analysis of the Trellis memory update
(arXiv:2512.23852), before any capacity experiment built on that analysis is
trusted. Each claim gets exactly one label: **proved identity**, **asymptotic
under stated assumptions**, **special-case analogy**, **empirical hypothesis**,
or **unsupported/incorrect transfer**. Every proved identity is implemented in
[trellis_lm/capacity_theory.py](../trellis_lm/capacity_theory.py) and verified
in float64 with asymmetric dimensions (m != d) by
[trellis_lm/tests/test_capacity_theory.py](../trellis_lm/tests/test_capacity_theory.py).

## Notation

One letter never names two things (in particular, `D` is never used both for a
dimension and for a pattern count):

    P   number of stored associations
    d   write/key dimension
    m   target/code dimension (Trellis slots)
    W   = [w_1, ..., w_P]  in R^{d x P}   (keys, unit-norm columns)
    A   = [a_1, ..., a_P]  in R^{m x P}   (target codes, columns)
    G_W = W^T W,  G_A = A^T A             (P x P Gram matrices)
    M   in R^{m x d}                      (memory state)

## Claim ledger

| # | Claim | Label | Status |
|---|---|---|---|
| 1 | Hebbian readout `M_H W = eta A G_W`; per-item signal/crosstalk split | proved identity | tested exact (P <= d and P > d) |
| 2 | Matched-code scores `S = A^T M_H W = eta G_A G_W` | proved identity | tested exact |
| 3 | Hebbian vector SNR ~ sqrt(d/(P-1)) | asymptotic under assumptions | second moment is exact for iid isotropic unit keys/codes; the SNR reading is a concentration statement, not a capacity theorem; Monte-Carlo verified |
| 4 | Delta recurrence equals the closed-form unrolling over `R_t = beta_t I - gamma_t w_t w_t^T` | proved identity | tested exact, multiple gamma/beta schedules, nonzero `M_0`, cyclic passes |
| 5 | `M_LS = A W^+` interpolates (`M_LS W = A`) for P <= d, full column rank | proved identity | tested; for P > d only the row-space projection `A W^+ W` is reached (normal equations tested) |
| 6 | One-pass delta reaches the pseudoinverse solution | unsupported/incorrect transfer | **rejected**; converged cyclic LMS does (P <= d), one pass leaves real residual — both tested |
| 7 | Orthonormal keys, gamma=1, beta=1: one-pass delta stores each pair exactly | proved identity | tested; also equals the Hebbian memory (zero crosstalk) |
| 8 | One-step update Jacobian is `beta I - gamma (w w^T kron H_ell)` | proved identity (column-major vec) | tested against full autograd for identity, SiLU, LN-SiLU; the row-major ordering is provably wrong (also tested) |
| 9 | Classical Hopfield ~0.14N and pseudoinverse N capacities bound Trellis | special-case analogy | binary autoassociative attractor results under different recall criteria; historical anchors only |
| 10 | Krotov N^(n-1) polynomial capacity applies to Trellis | unsupported/incorrect transfer | **rejected** (structural mismatch, below) |
| 11 | LN-SiLU has an "effective polynomial degree" (~1+) implying constant-factor capacity gain | unsupported/incorrect transfer | **rejected**; LayerNorm couples all m coordinates and removes input scale — a degree does not exist |
| 12 | Unnormalized SiLU "becomes identity" at high load | unsupported/incorrect transfer | **rejected**; for mixed-sign z at large magnitude, SiLU is ReLU-like (z for z>0, ~0 for z<0), not identity |
| 13 | The 0.78-PPL write-nonlinearity ablation measures a capacity multiplier | unsupported/incorrect transfer | **rejected**; it is a language-model quality delta under one training configuration |
| 14 | m = P, A = I bridge: coordinates of z are per-pattern overlaps | proved identity for the construction; special-case analogy for the Dense-Associative-Memory reading | tested; unbounded-state caveat below |
| 15 | Welch bound; simplex frame attains it; tight frames attain the mean-square level | proved identity (classical) | tested (orthonormal, simplex, frame-potential-optimized) |
| 16 | LN-SiLU stabilizes the update-Jacobian spectra across load, and any nonlinear writer improves a real capacity metric at matched state | empirical hypothesis | to be measured; nothing in this report supports or refutes it |

## Exact identities

**Hebbian.** `M_H = eta A W^T` gives the exact readout

    Z = M_H W = eta A G_W
    z_j = eta a_j + eta sum_{i != j} a_i <w_i, w_j>

— an exact signal/crosstalk decomposition needing no attractor dynamics. The
matched-code score matrix is `S = A^T Z = eta G_A G_W`; correct-item margins
and impostor scores follow from the two Gram matrices, with no assumption that
the code dimension m cancels.

**Delta / online LMS.** For phi = identity the Trellis update is

    M_t = beta_t M_{t-1} - gamma_t (M_{t-1} w_t - a_t) w_t^T

and with `R_t = beta_t I - gamma_t w_t w_t^T` (d x d, acting on M from the
right) the exact unrolling is

    M_P = M_0 R_1 R_2 ... R_P + sum_{i=1}^{P} gamma_i a_i w_i^T R_{i+1} ... R_P.

Each write's contribution is its outer product attenuated by the *suffix*
product of every later interference-and-decay operator. This replaces any
hand-waving about the one-pass delta rule "being the pseudoinverse": the
suffix products are what one pass does *not* undo.

**Least squares and RLS.** `M_LS = A W^+` interpolates exactly for P <= d with
full column rank; for P > d it lands on the row-space projection `A W^+ W`
with residual orthogonal to the data (`(M_LS W - A) W^T = 0`). Recursive least
squares reaches the ridge solution `A W^T (W W^T + lam I)^{-1}` online but
carries a d x d inverse-covariance state on top of the m x d memory — it is an
oracle, not a matched-state Trellis baseline.

**Update Jacobian.** With inner loss `l(z, a) = 1/2 ||phi(z) - a||^2`, update
direction `u = grad_z l = J_phi(z)^T (phi(z) - a)`, and `H_ell = du/dz` (the
inner-loss Hessian, symmetric: `J_phi^T J_phi + sum_k (phi_k - a_k)
Hess(phi_k)`), one step `F(M) = beta M - gamma u(M w, a) w^T` has the local
perturbation rule

    dF = beta dM - gamma H_ell (dM w) w^T

whose vectorized Jacobian, in the **column-major** vec convention, is

    beta I_{md} - gamma (w w^T kron H_ell).

Convention trap, verified in float64: flattening the autograd Jacobian with a
row-major reshape (torch's default) produces the transposed Kronecker order
`H_ell kron w w^T`, which is *not* equal for a coordinate-coupling phi such as
LN-SiLU. The test suite checks both that the column-major form matches
autograd (identity, SiLU, LN-SiLU) and that the row-major order does not.
For identity phi, `H_ell = I` and the rule collapses to `dM R_t` — consistent
with the unrolling above.

## Random-geometry SNR: what is exact, what is asymptotic

For iid isotropic unit keys and unit codes, the crosstalk second moment is
exact, not asymptotic:

    E || z_j - eta a_j ||^2 = eta^2 (P - 1) / d,

since `E <w_i, w_j>^2 = 1/d` exactly and the cross terms vanish in
expectation. The familiar reading

    SNR_vector ~ sqrt(d / (P - 1))

is the concentration/typical-case statement layered on that exact moment. It
is a *vector reconstruction SNR under random-geometry assumptions* — not a
universal storage-capacity theorem, and it says nothing about learned
(non-random) key or code geometry. The Monte-Carlo test verifies the moment to
10% and the RMS SNR to 5% at d=64, m=48, P=33.

## The m = P bridge, and its caveat

There is one exact bridge to Dense Associative Memory: set `m = P`, `A = I_P`.
Then `M_H = eta W^T` and each coordinate of `z = M q` is one stored pattern's
overlap, `z_i = eta <w_i, q>` (tested exactly). Only in this regime is an
elementwise nonlinearity on z genuinely a *per-pattern overlap* nonlinearity —
the structure Krotov-style polynomial capacity actually requires.

The caveat that keeps this a diagnostic and not an architecture: the state is
then P x d and **grows with the number of stored patterns**. That is an
unbounded explicit-pattern memory, not bounded-memory compression. Any claimed
Dense-Associative-Memory-like benefit must additionally survive the compressed
regime (fixed m while P grows), which is an open empirical question, not a
corollary.

## Explicit rejections

- **No Krotov N^(n-1) transfer to Trellis.** Dense Associative Memory applies
  its nonlinearity to each stored pattern's overlap separately,
  `sum_mu xi^mu F'(<xi^mu, x>)`; that per-pattern structure is what yields
  polynomial capacity. Trellis keeps a fixed m x d matrix, computes `z = M w`,
  and applies phi inside an online regression loss and its VJP
  (`u = J_phi(z)^T (phi(z) - a)`). Outside the m = P bridge, z coordinates are
  not pattern overlaps, and the polynomial-capacity argument has no object to
  attach to.
- **No LN-SiLU "effective degree".** LayerNorm couples all m coordinates,
  removes input-dependent mean and scale, and reshapes the Jacobian and
  Hessian; a polynomial-degree analogy is not defined for it. Separately,
  unnormalized SiLU does not "become identity" at high load — with mixed-sign
  inputs it is asymptotically ReLU-like.
- **No 0.78-PPL-as-capacity-multiplier.** The paper's write-nonlinearity
  ablation (11.65 vs 10.87 perplexity) is a language-model quality delta under
  one training configuration. It is not a measured storage capacity, and no
  capacity multiplier can be read off it.
- **One-pass != pseudoinverse.** The exact unrolling shows each write filtered
  through the suffix operators `R_{i+1} ... R_P`. Converged cyclic LMS
  approaches `A W^+` for P <= d (tested); a single pass leaves an orders-of-
  magnitude-larger residual (tested). Claims that assume the online update
  reaches the offline optimum are wrong at P of order d.
- **Hopfield numbers are anchors, not bounds.** The classical ~0.14N and
  pseudoinverse-rule N capacities concern binary autoassociative attractor
  memories under basin-of-attraction recall. Trellis retrieval is one-shot
  from a query vector; the criteria do not match, so the numbers do not
  transfer.

## Standing wording

Classical Hopfield, pseudoinverse, and Dense Associative Memory results provide
useful controls and limiting cases, but their capacity theorems do not directly
apply to Trellis. Trellis is a one-pass, decayed, learned-code, nonlinear online
regression memory. Its relevant capacity must be measured as a function of key
dimension, code dimension, update budget, code geometry, decay, and retrieval
criterion.

## Implementation

[trellis_lm/capacity_theory.py](../trellis_lm/capacity_theory.py) implements
the reference algorithms (Hebbian; one-pass / K-cyclic / converged delta; the
exact unrolling; pseudoinverse and RLS oracles; the nonlinear Trellis update
reusing the closed-form VJPs in
[trellis_lm/activations.py](../trellis_lm/activations.py)), the recall and
frame metrics (score matrix, coherence, frame potential, effective rank,
tight-frame distance, Welch bound), the update-Jacobian helpers, and the key /
code ensembles (orthonormal, random unit, frame-potential-optimized, clustered,
near-duplicate, simplex, range-matched targets). One measured detail worth
recording: projected gradient on the frame potential converges to a unit-norm
*tight* frame — the mean-square off-diagonal coherence attains its Welch level
exactly, but the max coherence does not approach the Welch bound
(equiangularity is strictly stronger than tightness), so "optimized
low-coherence frame" must be read as low mean-square coherence.

## References

- Behrouz et al., "Trellis" (arXiv:2512.23852).
- Krotov, Hopfield, "Dense Associative Memory for Pattern Recognition"
  (arXiv:1606.01164) — states the ~0.14N, pseudoinverse N, and N^(n-1)
  results in the binary autoassociative setting.
- Kanter, Sompolinsky, "Associative recall of memory without errors" (1987) —
  the projection rule reaching load p/N = 1 for linearly independent patterns.
- Welch, "Lower bounds on the maximum cross correlation of signals" (1974).
