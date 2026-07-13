# SymKV: symmetry-adapted KV-head mode compression

SymKV is a falsification-first study of whether the KV cache can be compressed
along its *head* axis by reserving one basis mode for head consensus. It is R&D,
not a shipping claim: the null result is a perfectly good outcome and the design
is built to make that null easy to reach.

## The idea in one paragraph

At each decode position a model holds, per layer and separately for K and for V,
a set of `H` key/value heads, each a `D`-dimensional vector. Stack them into a
matrix `X` of shape `H x D`. Instead of storing all `H` heads, SymKV stores a
rank-`m` projection: pick an `H x m` orthonormal basis `B`, keep the code
`Z = Bᵀ X` (shape `m x D`), and reconstruct `X̂ = B Z` when attending. The only
thing that distinguishes SymKV from ordinary per-head PCA is *which* basis: SymKV
forces the consensus direction `u0 = ones(H)/√H` — the mode along which every head
contributes equally — to be column 0, then fills the remaining `m-1` columns with
the top eigenvectors of the head covariance restricted to `u0`'s orthogonal
complement, `P⊥ C P⊥`. The bet is that the average-of-heads signal is both large
and unusually stable across data, so nailing it with a fixed, calibration-free
mode buys predictive quality that a generic PCA basis — which spends its whole
budget chasing raw variance — spreads too thin at small `m`.

## Why it is falsification-first

There is a theorem working against SymKV, and we state it up front rather than
discovering it late. PCA is the rank-`m` basis that minimizes raw reconstruction
error (Eckart–Young). SymKV is PCA with one column constrained to `u0`, so it can
*never* reconstruct `X` better than PCA at the same `m`, and the Gate-0 test suite
asserts exactly that ordering (`pca ≤ symkv_raw ≤ random_sym` on raw MSE). So a
SymKV win cannot come from reconstruction. It can only come from *predictive*
quality — next-token KL / NLL through the real model — being governed by
something other than raw KV MSE, such that the consensus mode is worth more to the
output than its variance share implies. If that gap does not exist, SymKV loses,
and the study should say so.

The stopping rule is pre-registered: if `symkv_raw` does not beat the `random_sym`
control on held-out predictive metrics, and never beats `pca_head` at any
non-trivial budget, the verdict is negative and the line stops before any
alignment, causal-probe, or kernel work.

## What is built (Gate 0)

The `symkv/` package is the math core, locked by CPU unit tests before any GPU or
model runs:

- `basis.py` — the six bases behind one orthonormal interface: `full` (identity
  control), `mean_only` (`u0` alone), `pca_head` (the raw-MSE optimum and the
  ceiling to beat), `symkv_raw` (the hypothesis), and the `random_sym` /
  `grouped_mean` controls that a real structural win must clear.
- `covariance.py` — `HeadCovariance`, a streaming float64 accumulator for the
  uncentered head second moment `C = E[x xᵀ]`; uncentered because `u0` lives in
  the mean and SymKV models it rather than removing it.
- `codec.py` — encode/decode, reconstruction MSE, and honest per-token byte
  accounting that amortizes the shared basis over a layer's tokens and never
  credits a saving to a dense reference path we only materialize to measure error.
- `config.py` — `SymKVConfig` (`H`, `D`, `m`, method, dtypes, seed).

`tests/test_symkv_math.py` is Gate 0: 11 CPU tests (unit-norm `u0`, `P⊥`
projector identities, orthonormality, forced consensus mode, exact full-rank
reconstruction, MSE monotonicity, byte accounting, and the PCA lower-bound
premise). All pass under the prune `w7900-ml` env on CPU.

## The model lane (Phase 1, next)

The first experiment that can kill the idea runs on Llama-3.2-3B-Instruct
(`num_key_value_heads = 8`, `head_dim = 128`, 28 layers — so the head-mode budget
`m ∈ {1,2,4,8}` is a real 8×/4×/2×/1× head-axis payload, not a degenerate one).
Calibrate `C` per layer and per K/V on ~32 prompts, hold out ~64 disjoint prompts
at contexts 512 and 1024, and teacher-force ≥8 future positions. Compare `full`,
`mean_only`, `random_sym` (×5 seeds), `grouped_mean`, `pca_head`, and `symkv_raw`
at each `m`. Primary axes: bytes vs next-token KL, bytes vs ΔNLL, raw-KV-MSE vs
predictive-KL (the decoupling the whole hypothesis rides on), `u0` energy per
layer, and the angle between `u0` and the PCA subspace. The verdict is written to
`VERDICT.md` in the dated result directory under `knlp-key-results`, whichever way
it falls.

Provenance and the full brief live in
`/data/knlp-key-results/symkv-20260713/` (`BRIEF.md`, `PHASE0_AUDIT.md`). This
line runs on the free local W7900; it carries no cloud-spend tag.
