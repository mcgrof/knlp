# Trellis implementation fidelity

This document states, equation by equation, how the `trellis_lm` implementation
relates to the Trellis paper (arXiv:2512.23852), which parts are faithful, which
are configurable choices, and which the paper simply does not specify. It exists
because the published text is internally inconsistent in several places, so a
bit-exact reproduction of the reported baseline is not possible from the paper
alone — and it is better to say exactly where the reconstruction begins than to
imply a fidelity the source does not support.

## Where the paper is under-specified or inconsistent

Reading the method and the ablations together, the following are genuinely not
pinned down by the text:

- the baseline write nonlinearity `φ` is never defined (only that it is
  nonlinear, and that `φ = identity` is an ablation);
- the reported baseline chunk size `C` is never stated;
- the source, granularity, and whether-learned of the inner learning rate `γ`;
- the exact parameterization and initialization of the forget gate `β`;
- whether the intermediate `Norm`/`LayerNorm` carries affine parameters;
- the short-convolution kernel width and whether it is depthwise;
- the initial memory state `M₀` and the exact multi-head lifting of the
  single-head equations.

There are also two internal inconsistencies worth flagging so future readers do
not chase them: the method text defines the inter-pass map `f` one way while the
ablation table's reported baseline uses another (below), and the decay
coefficient's derivation from the stated regularizer does not reduce to the
`β = 1 − λ` the update equation uses. Treat `β` as the primary forget gate from
the update equation; do not try to recover it literally from the regularizer.

## The recurrence (faithful)

Per token, with a fixed `m`-slot state `M ∈ ℝ^{m×d}` and write vector `w`
(the key `k` on the first pass, the value `v` on the second):

```
z = M @ w                      # pre-update readout code, [m]
u = J_φ(z)^T (φ(z) - α)        # VJP of the compression loss ||φ(M w) - α||^2
M = β · M - γ · u wᵀ           # gated online-gradient update
y = read(M, ·)                 # write-before-read: readout uses the updated M
```

`run_trellis_memory` implements exactly this: negative-gradient update, the
`u wᵀ` outer-product orientation, the VJP taken from the pre-update state, and
write-before-read (`M @ q` on the key pass, `Mᵀ @ r` on the value pass). Using
an autograd VJP rather than materializing `J_φ` matters when `φ` includes a
LayerNorm, whose Jacobian is not diagonal. The literal loss carries a factor of
2 in the gradient; the implementation folds it into `γ` (the half-squared-loss
convention), which is a rescaling of the learning rate, not a different rule.

For end-to-end (bilevel) training the inner VJP must keep its graph
(`exact_inner=True`); the stale/detached inner update is a first-order
approximation, not the paper's full bilevel objective.

## φ versus f (the one real gap that was fixed)

`φ` (the write nonlinearity in the compression loss) and `f` (the inter-pass map
applied to the key readout before the value pass) are **distinct** functions in
the paper. The method text writes `f = SiLU(x)/‖SiLU(x)‖` (L2-normalized SiLU),
but the ablation table's reported baseline uses `f = LayerNorm(SiLU(x))`, and
lists L2-SiLU only as a modification (10.98 vs the 10.87 baseline). So `ln_silu`
is the better-supported `f` for reproducing the tables; L2-SiLU is a documented
alternative, not the baseline. The baseline `φ` is never stated at all.

The implementation now exposes `phi_activation` and `f_activation` as
independent knobs (each falls back to `activation` when unset, preserving prior
behaviour). This is required to run the paper's own ablations — varying `f`
while holding `φ` nonlinear, and separately setting `φ = identity`. A single
shared activation cannot express either, and tying them was the substantive
infidelity in the earlier code.

The final `φ` on the value readout, `y = φ(Mᵀ r)`, is `value_readout_act`
(default `none`; the faithful profile sets it).

## Configurable choices and the faithful profile

Most remaining paper-vs-default differences are already knobs; they are simply
not the defaults, because the defaults favour the exact/legacy paths used during
Phase-0 correctness work. `TrellisConfig.faithful_baseline()` sets the
reported-baseline choices the text does pin down:

- `f_activation = ln_silu` — the reported-baseline inter-pass map;
- `value_readout_act = ln_silu` — the final `φ` on the value read;
- `output_path = "paper"` — the Fig. 1 shell (Trellis → Norm → GeLU-gated branch
  → Linear, with SwiGLU as the separate feed-forward sublayer);
- `trellis_beta_min = 1e-3` — the forget gate spans `(0,1)` and can erase memory;
  it must not be floored near 1 (the default `0.90` floor is an engineering
  restriction, not paper-faithful).

`phi_activation` is deliberately left unset (ties to `activation`) because the
paper never defines the baseline `φ` — it is a sweep parameter, not a known
value. `γ` (learned, positive, per head via softplus) is a defensible
reconstruction of the scalar `γ_t`, not a stated design; keep it configurable
and do not describe it as paper behaviour.

## Exact recurrence versus the chunked baseline

The pure sequential recurrence (`chunk_size = 1`) is, in paper terms, the "fully
non-linear recurrence" ablation (`B = 1`), which the paper reports at *slightly
better* perplexity (10.75 vs 10.87) at the cost of parallel-training efficiency.
It is therefore not the reported baseline: that uses the stale-gradient chunk
approximation with `C > 1`, and since `C` is unstated, its exact numbers cannot
be reproduced from the text. Name the `C = 1` path for what it is — the exact/
`B = 1` ablation — not "the paper baseline". The chunk kernel is validated
against the sequential path as the trusted oracle; the printed chunk equation is
dimensionally inconsistent as typeset and should not be copied literally.

## Fidelity tests

`trellis_lm/tests/test_trellis_paper_fidelity.py` asserts the paper's stated
properties as one labelled suite:

- **C = 1 chunk == sequential** and the **auto C−1 (nilpotent-exact) chunk ==
  sequential** — the chunk kernel reproduces the serial recurrence (also covered
  by `test_trellis_chunked.py`);
- **`φ = identity` reduces to the gated delta rule** — compared against an
  independent closed-form delta-rule loop;
- **write-before-read** — perturbing `w_t` moves `y_t` but leaves `y_{t-1}`
  untouched (also covered by `test_trellis_causality.py`);
- **`φ`/`f` decouple** — the two knobs resolve independently and the faithful
  profile sets the reported-baseline values.

## Bottom line

The core recurrence, the two-pass key-then-value read, write-before-read, and
the Fig. 1 shell are faithful. The one substantive fix was decoupling `φ` and
`f`. Everything else is either a default to flip (final `φ`, `β` range), a
labelling correction (`C = 1` is an ablation), or a detail the paper leaves
open, which is documented here as a reconstruction rather than presented as the
paper's specification.

## Addendum: the clean write-only ablation, now run

Decoupling `φ` from `f` made the write-only ablation runnable for the first
time — vary the write nonlinearity `φ` while holding the inter-pass map `f` at
LN-SiLU. Run on the synthetic associative-recall task (2 key/value pairs,
d_model 256, 2 layers, 4 heads, 64 slots, batch 256, 3000 steps, W7900, fp32),
mean ± std over 3 seeds of the last-6-eval recall accuracy:

| write `φ` | `f` | recall_acc | final loss |
|---|---|---|---|
| identity (linear) | LN-SiLU | **1.000 ± 0.000** | 0.000 |
| LN-SiLU (nonlinear) | LN-SiLU | **0.617 ± 0.130** | 0.781 |
| dense transformer (reference) | — | 0.725 ± 0.142 | 0.482 |

With `f` held fixed, the linear write (which at `φ = identity` is the gated
delta rule) solves the binding task perfectly and beats the dense baseline,
while the nonlinear write does not converge and trails both. So the isolation
the fix enables makes the nonlinear write look *worse*, not better — fixing the
confound did not rescue it.

Scope, stated plainly: this is the binding/recall probe, not LM perplexity, and
not the Trellis-vs-Gated-DeltaNet comparison the line's closure rests on (that
needs the FLA baselines, unavailable on this ROCm box). It is mechanism-level
evidence, consistent with the closure, not a re-run of the headline comparison.
Note the task-dependence: the paper reports the nonlinear write *helping* LM by
~0.78 ppl, whereas here it *hurts* associative recall — the write
nonlinearity's value has opposite sign on the two tasks.

Reproduce (per arm, `phi_activation` in {identity, ln_silu}):

```
python -m trellis_lm.train --task recall --model trellis --steps 3000 \
  --batch 256 --lr 3e-3 --n_pairs 2 --d_model 256 --n_layers 2 --n_heads 4 \
  --d_head 64 --n_slots 64 --dtype fp32 \
  --phi_activation <identity|ln_silu> --f_activation ln_silu
```

## Outer-gradient semantics: the meta-learning provenance

Trellis has the nested-gradient structure of gradient-based
meta-learning: an inner gradient update to the fast memory, then an
outer loss evaluated after that update. The literal end-to-end bilevel
derivative therefore corresponds to
[MAML](https://proceedings.mlr.press/v70/finn17a/finn17a.pdf)-style
differentiation through the inner gradient — a "gradient through a
gradient" carrying Hessian-vector terms — and detaching the inner
correction is the named first-order approximation of
[FOMAML](https://arxiv.org/abs/1803.02999). (Reptile, from the same
paper, is related first-order precedent but moves the initialization
toward adapted parameters; it is not the direct stop-gradient
analogue.) The provenance runs deeper than external convention: the
architectural predecessor Trellis builds on,
[TTT](https://arxiv.org/abs/2407.04620), states outright that its
forward pass contains the gradient operator and that backward through
it takes "gradients of gradients", distinguishing the inner-loop
gradient on the fast weights from the outer-loop gradient on the slow
parameters. That is nearly a specification of what our full-bilevel
mode computes.

Two separations keep this honest. First, the
[paper's](https://arxiv.org/abs/2512.23852) chunk approximation
freezes only *which state the inner gradient is evaluated at* (the
chunk start); that is a forward-side choice and does not by itself
imply the backward is detached — which state feeds `u_t` and whether
the outer backward differentiates through `u_t` are independent
decisions, and only the first is declared. The faithful paper-style
reading is chunk-start-stale forward *with* full differentiation
through that approximate computation — exactly the
`chunk_start_stale + full_bilevel` combination the code now
implements as a reference. Second, the paper's bilevel sentence also
cites [BOME](https://arxiv.org/abs/2209.08709), a first-order bilevel
solver, so the bibliography does not unambiguously mandate
second-order training; the paper never says it applies BOME's
algorithm, never labels its training first-order, and never describes
stopping gradients through `u_t`, so the citation introduces
ambiguity rather than an answer.

The defensible statement, then: full-bilevel differentiation is the
default semantics of the method as written — by MAML convention, by
the cited TTT lineage, and by ordinary total differentiation — while
a detached first-order implementation is a reasonable engineering
approximation that is a *distinct algorithmic choice*. Because the
paper discloses neither stop-gradient nor higher-order-backward
semantics, both modes are implemented and must be reported
explicitly; the labeled comparisons will speak louder than any
reading of the text.
