# Trellis bounded-memory LM — evaluation (Phase 0)

This is the full-architecture Trellis line: a learned bounded-memory sequence
mixer that produces its output directly from a fixed-size compressed memory
state, **not** a `[B,H,T,T]` attention mask. (The mask-only "Trellis-KRI"
selector is a separate, parked line.) Phase 0's job, per the task plan, is
correctness — unit tests and a toy associative-recall task — before any
optimization or large-scale comparison. This document reports Phase 0 and
defines the staged rollout for Phases 1–4.

## 1. What was implemented

A clean `trellis_lm/` package, Kconfig-integrated (`make defconfig-trellis-tiny`
then `make trellis` / `trellis-recall` / `trellis-smoke` / `trellis-bench`):

- `config.py` — `TrellisConfig` (all knobs: n_slots, activation, alpha/beta
  modes, gamma_init, conv, post_gate, forget_gate, exact_inner, chunk_size).
- `activations.py` — `ln_silu` (default phi), `l2_silu`, `softmax`.
- `trellis_memory.py` — exact sequential bounded-memory recurrence. Per token:
  `z = M@write`, `u = J_phi(z)^T(phi(z)-alpha)` via an **exact autograd VJP**,
  gated OGD update `M <- beta*M - gamma*outer(u, write)`, write-before-read
  readout (`M@q` key pass → `[B,H,M]`; `M^T@r` value pass → `[B,H,D]`).
- `trellis_mixer.py` — two-pass mixer (RMSNorm → q/k/v/alpha/beta projections →
  causal depthwise conv on q,k → key pass → f → value pass → merge → out-proj →
  SwiGLU post-gate).
- `model.py` — `TrellisLM` and a matched `DenseTransformerTiny` baseline (same
  d_model/layers/heads/d_head, SwiGLU, RMSNorm, causal MHA).
- `train.py` (recall toy + TinyStories LM), `eval_ppl.py`,
  `eval_long_context.py` (RULER/NIAH-style recall vs context, incl. a
  repeated-key condition), `bench_memory_latency.py`.
- `tests/` — causality, shapes, memory/grad-flow.

## 2. What is faithful to the paper, what is approximated

Faithful: fixed-size `m`-slot memory; two-pass (key then value) recurrent
compression; the online-gradient update with a forget gate `beta` and learning
rate `gamma`; the inner objective `||phi(M@write)-alpha||^2` and its exact VJP
`u`; readout from the compressed state; memory is `O(layers*heads*slots*d_head)`,
independent of `T`.

Approximated / deferred (Phase 1+): the inner update is computed by per-token
autograd VJP (correct but slow) rather than the paper's chunked stale-gradient
linearization (`chunk_size>1`); `gamma` is a positive scalar per head
(softplus), not a learned schedule; the activations are parameter-free; no
custom kernels. These are speed/scale items, explicitly deferred until
correctness is proven (it now is).

## 3. Phase 0 results (W7900, fp32, exact sequential)

**Unit tests: 10/10 pass** — no-future-dependence (causality) with a live
perturbation check, shape correctness (trellis + dense + per-slot/softmax
variants), gradient flow through q/k/v/alpha/beta/gamma, deterministic eval,
bounded memory `<<` full-KV at T=2048, and a forget-gate-off behavioural change.

**Synthetic associative recall (MQAR), answer-position-supervised:**

| model | pairs | steps | final loss | recall acc | mem state / seq |
|---|---|---|---|---|---|
| dense (control) | 2 | 800 | 0.55 | **0.78** | grows with T (KV) |
| **Trellis** | 2 | 800 | 1.99 | **0.22** | **262 KB, flat in T** |
| dense | 8 | 400 | ~2.6 | chance | — |
| Trellis | 8 | 400 | ~2.7 | chance | — |

Reading: with answer-only supervision the dense control learns 2-pair recall
(loss 3.6→0.55, acc→0.78), proving the harness is correct. **Trellis binds
key→value** — loss drops well below the value prior (log 16 ≈ 2.77) to 1.99 and
recall rises to ~0.22, clearly above chance — but in this minimal exact form it
is **weaker than dense** at the toy. The 8-pair case is under-trained for both
at 400 steps (it was a pipeline-debugging artifact: full-sequence CE drowned the
single recall position; fixed by answer-only loss).

**Minimum bar (Phase 0): MET.** Trains stably; loss decreases; key→value binding
occurs; memory is bounded and independent of sequence length; gradients flow
through every projection; the forget-gate ablation is live.
**Strong bar (beat dense): not yet** — and not expected at Phase 0; it is the
Phase-2 matched-budget question.

## 4. Memory footprint (the structural win, analytic)

Trellis state `= 2 * L * B * H * M * d_head * elem` (two passes), **flat in T**.
Dense KV cache `= 2 * L * B * H * T * d_head * elem`, **linear in T**. They cross
at `T = M`; beyond that Trellis uses strictly less state. For the tiny config
(L=2,H=4,d=64,M=64) the per-seq state is 262 KB regardless of context, while a
dense 8K-token KV cache for the same dims is ~16 MB and growing. This is the
bounded-memory property the architecture exists for.

## 5. Failure analysis

The Trellis-vs-dense gap on the toy is the thing to chase in Phase 1–2, not to
hand-wave. Leading suspects, in order: (a) the recall path binds k↔v through a
*shared* `alpha` code, but `alpha_t = alpha_proj(h_t)` differs at the key vs
value token, so the binding leans on the short conv to align them — an
alpha-binding diagnostic and a conv-kernel sweep are the first checks; (b)
under-training — Trellis loss was still decreasing at 800 steps; (c) the
exact per-token update may need a different `gamma`/`beta` init or more slots
relative to pairs. These map directly onto the planned diagnostics (beta/gamma
distributions, alpha entropy, slot-collapse, read-before-write vs
write-before-read, activation and m sweeps).

## 6. Phase 1 — TinyStories smoke (W7900)

Matched scratch comparison, gpt2-tokenized TinyStories, exact Trellis vs dense
(same d_model=256/4-layer/4-head/d_head 64, SwiGLU, RMSNorm).

| model | seq | val PPL | tok/s | memory state |
|---|---:|---:|---:|---|
| Trellis | 256 | 67.2 | 962 | 262 KB (flat in T) |
| Trellis | 512 | 41.2 | 934 | 262 KB |
| dense | 256 | 47.4 | 116K | grows with T (KV) |
| dense | 512 | 48.1 | 131K | grows with T |

Trellis PPL falls sharply with context (67→41) and crosses below dense at 512,
while dense is flat (~47–48). The per-token recurrence is ~125× slower than
dense (the speed wall). Stale-gradient mode (exact_inner=False) was then
validated: it learns the recall toy better (acc 0.53 vs exact 0.22) and is
~2.4× faster at seq 1024 (1094 vs 448 tok/s), so Phase 2 uses it.

## 7. Phase 2 — matched scratch comparison + ablations (W7900, stale)

Matched tokens (steps 500, batch 8, same seq per length for both models).

| seq | Trellis val PPL | dense val PPL | Trellis vs dense |
|---:|---:|---:|---:|
| 512 | **31.0** | 45.8 | −32% |
| 1024 | **23.4** | 39.0 | −40% |

**Scratch Trellis beats same-size dense at both lengths under matched tokens,
and the advantage widens with context** (ratio 0.68 → 0.60). This is the
headline positive and the direct bounded-memory signal.

Ablations (Trellis stale, seq 512; baseline is the 500-step run, the others are
300-step — note the step mismatch when comparing to baseline):

| variant | val PPL |
|---|---:|
| baseline (n_slots 64, ln_silu, forget on; 500 steps) | 31.0 |
| no forget gate (300 steps) | 49.0 |
| softmax phi instead of ln_silu (300 steps) | 42.5 |
| n_slots 32 (300 steps) | 35.7 |
| n_slots 128 (300 steps) | 34.4 |

The **forget gate is load-bearing** (removing it is the worst single change),
**ln_silu ≫ softmax** for the inner activation, and **slot count 32–128 barely
matters** at this scale. Caveats: tiny model, single seed, TinyStories (weak
long-range structure), Trellis carries ~8% more params (the alpha projection),
and Trellis ran in stale mode vs the dense standard path.

## 8. Phase 3 — long-context recall (inconclusive on LM-trained tinies)

`eval_long_context.py` on the Phase-2 checkpoints, associative recall with
0/8/32/64 filler tokens and a repeated-key condition. **Recall accuracy was
0.000 (chance) for every model and condition**, trellis and dense alike. These
checkpoints were trained for language modeling on TinyStories, not for a recall
objective, and are tiny, so they cannot do this synthetic retrieval — the eval
is honest but uninformative here. A real recall test needs models trained on a
recall objective at length, which is loop-speed-gated.

## 9. Phase 4 — GPT-2 retrofit + distillation (weak warm-start)

`TrellisRetrofit.from_gpt2` warm-starts a 124M Trellis from GPT-2 (transfers
wte, tied lm_head, and attention q/k/v/o) and distills against the frozen GPT-2
full-cache teacher (CE + KL). Bounded memory state 2.36 MB/seq (flat in T).

| mode | trainable | CE @1 | CE @200 | KL @200 | tok/s |
|---|---:|---:|---:|---:|---:|
| full FT | 166 M | 20.7 | **5.95** | 3.42 | 343 |
| LoRA r16 | 1.6 M | 20.9 | 20.8 (stuck) | 17.6 | 347 |

The warm-start CE (~20.8) is **worse than random** (log 50257 ≈ 10.8): pouring
GPT-2's attention weights into the bounded-memory operator does not produce
attention-like behavior (different operator; Trellis also has no positional
embedding). **LoRA cannot move it** (the transferred mixer is frozen garbage),
while **full fine-tune recovers** (CE 20.7→5.95) — but that is the mixer
learning essentially from scratch, not a cheap adaptation. So GPT-2→Trellis
retrofit is **not a cheap PEFT win like KRI-FT**; a useful retrofit needs
substantial full training, which is loop-speed-gated.

## 10. Overall verdict

Bounded-memory Trellis is **genuinely promising**: it is faithfully implemented
and correct (Phase 0), trains stably, keeps memory bounded and flat in
sequence length, and — the real positive — **beats a same-size dense
Transformer at matched tokens at 512 and 1024, with the gap widening as context
grows** (Phase 2). The forget gate and ln_silu activation are the load-bearing
design choices.

But it is **not yet decisively validated**, and every decisive next test is
blocked by the same wall: the exact/stale **per-token recurrence is too slow**
(W7900 ~1094 tok/s stale @1024; an A100 pod was 4× *slower* — this is a
kernel-launch-bound Python loop, so a bigger GPU does not help). Consequently
the strong-bar test (beats dense at ≥4k context) and a real Phase-4
retrofit-vs-KRI-FT at scale **cannot be run** with the current loop. The
**chunked stale-gradient kernel** (parallelize the recurrence within chunks,
validated against the exact version) is the single gating build that unlocks
both. Phase 3's recall test also needs recall-objective training at length,
again gated on that speed.

**Recommendation: build the chunked kernel next, or park here.** The Phase-2
matched win justifies the investment; without the chunked kernel, Trellis stays
at "promising at 512–1024, undecided at the lengths that matter." Park is
defensible too — the result is documented and reproducible (`make
defconfig-trellis-tiny && make trellis`). Not recommended: more single-GPU
per-token-loop runs (they cannot reach the deciding regime).

## 11. Phase 2x — long context with the chunked kernel (the ≥4k test)

The chunked stale-gradient kernel (chunk_size=64) made ≥4k training feasible:
~17K tok/s on the W7900, flat across context (vs ~140 tok/s sequential at
8192). It is validated — chunk_size=1 exactly reproduces the sequential path,
the C>1 form is causal and trains — and it fixed a latent bug (stale mode had
frozen alpha_proj; alpha now trains). Matched-token sweep (steps 400, batch 4),
trellis(chunk64) vs same-size dense, val PPL on TinyStories:

| seq | trellis | dense | trellis vs dense |
|---:|---:|---:|---:|
| 1024 | 41.3 | 42.3 | −2% |
| 2048 | 37.8 | 51.8 | −27% |
| 4096 | 39.5 | 42.4 | −7% |
| 8192 | 36.1 | 40.8 | −12% |

(Phase-2's better-trained 512 point — trellis 31.0 vs dense 45.8, −32% — used a
larger budget, so it is not directly comparable to these undertrained 400-step
runs.)

**Strong-bar verdict: PARTIAL PASS.** Scratch Trellis beats same-size dense at
every tested length **including 4096 and 8192 under matched tokens** — the
"beats dense at ≥4k" bar is met, and Trellis's bounded 262 KB state holds while
dense's KV grows with context. **But the margins are noisy and non-monotonic**
(2048 best at −27%, 1024 nearly tied), so the cleaner claim — a *widening* gap
with context — is **not** supported by this short single-seed run. The absolute
PPLs are higher than Phase 2 because the token budget here is ~2.5× smaller
(undertrained), not because of the kernel.

Caveats: single seed, tiny 4M model, TinyStories (weak long-range structure — a
real strong-bar result wants PG19/code), Trellis ~8% more params, and the C=64
stale-within-chunk approximation.

### Chunked-quality cross-check (matched budget, @1024)

Same budget (batch 4, steps 400), Trellis only, three inner modes:

| mode | val PPL |
|---|---:|
| sequential-stale (chunk 1, exact recurrence) | 28.2 |
| chunk 16 | 38.2 (+35%) |
| chunk 64 | 41.3 (+46%) |

**The chunked kernel is a speed/quality tradeoff, not free.** The
stale-within-chunk shortcut (use the chunk-start state for every token's inner
code) loses running-state information, and the cost grows monotonically with
chunk size: +35% PPL at C=16, +46% at C=64 versus the exact recurrence at
matched tokens. Two consequences. First, the earlier 23.4 (Phase 2, seq-stale)
vs 41.3 (Phase 2x, C=64) gap is **both** the token budget (Phase 2 used ~2.5×
more tokens: seq-stale drops 28.2→23.4 with more data) **and** the chunking
penalty (28.2→41.3 at matched budget). Second, the ≥4k trellis-beats-dense
result in section 11 used C=64, i.e. Trellis was *handicapped* — so that win is
conservative; a smaller chunk (or the exact path) would only help Trellis.

### The exact within-chunk solve: built, and why it can't be cheap

We then built and fully characterized the exact within-chunk solve. The key
structural fact: in the sequential path the forward `z_t = M_{t-1} @ w_t` uses
the *true* M_{t-1} — "stale" there refers only to detaching u from the gradient
graph (no 2nd-order), the forward is exact. So the chunked forward *can* match
the sequential forward exactly; the only approximation is using the chunk-start
state M0 for z inside a chunk. The within-chunk dependency z_t ← u_{s<t} is
strictly lower-triangular, so a Jacobi (all-at-once) iteration on it is
**nilpotent**: after k passes the first k tokens are bit-exact, and `C-1` passes
reproduce the sequential forward exactly. The earlier "oscillation" was a
red herring — it is the transient of Jacobi (the max over not-yet-converged
tokens bounces while the converged *prefix* grows by one per pass); the failed
experiment simply used too few passes (refine=1 at C=16 is 14 short of exact).

This is now implemented as `refine_passes < 0` → auto `C-1` (a verified-exact
oracle; new test `test_chunked_exact_mode_matches_sequential`). The convergence
curve (W7900, fp32, random probe) confirms it:

```
 C  refine   div_vs_sequential
 4    1         2.7     (prefix of 1 exact)
 4    3         1e-5    (exact: C-1)
16    1         5.0
16    2       160       (transient bounce, NOT divergence)
16   12         7e-5    (exact: C-1)
```

But exactness here buys no speed, and the wall-clock proves it (W7900, T=1024,
B4·H4·D64·M64, M_q pass):

```
            sequential   stale (r0)      exact (r=C-1)
 C=16          240 ms     22 ms (11×)     250 ms  (≈ seq, +4%)
 C=64          240 ms      6 ms (40×)     251 ms  (numerically broken)
```

The reason is structural: the refine loop is a Python loop *inside* the per-chunk
loop, so exact costs `(T/C)·(C-1) ≈ T` inner steps — the same count as the
sequential loop. A *single-pass* closed-form solve (one `(I − A)⁻¹`, as DeltaNet
uses) exists only when the inner step is **linear** in z. Trellis's nonlinear φ
makes u_t = J_φ(z_t)ᵀ(φ(z_t) − α_t) nonlinear in z, which forfeits that
delta-rule free lunch: there is **no cheap-and-exact chunkwise form for this
architecture**. A Gauss-Newton linearization would restore a single solve, but
only to first order (still approximate) and at a block-triangular `(C·M)²`
cost. Separately, the segmented-product reconstruction `P = cumprod(β)`
underflows in fp32 for large chunks (0.5⁶⁴ ≈ 5e-20), so even the exact oracle is
numerically usable only up to C ≈ 32 — a second, independent ceiling.

**Verdict.** The chunked kernel has exactly two sane regimes and nothing in
between (any fixed `0 < refine < C-1` is a trap — erratic unconverged tail):
`chunk_refine=0` (true-stale: 11–40× faster, +35–46% PPL) for fast exploration,
and `chunk_refine=-1` (exact oracle: bit-exact to sequential, but ≈ sequential
cost and C≲32) for validation. For final-quality numbers use the sequential
path. Speed and exactness are mutually exclusive for Trellis's nonlinear inner
step — that is a property of the architecture, not an implementation gap.

## 12. Phase 2y — PG19, multi-seed, held-out PPL (the firmed-up result)

The Phase 2x ≥4k win was single-seed, on a weak corpus, with train-loss PPL.
Phase 2y removes all three confounds: a real long-range corpus (PG19, Project
Gutenberg books — documents of 68k–567k characters, where bounded memory should
actually matter), three seeds, and **held-out validation PPL on a disjoint
slice** (driver `scripts/trellis_firmup.py`). Matched-size Trellis vs the dense
tiny baseline (≈4.3–5.2M params each, same d_model/layers/heads, same tokens),
400 steps, W7900 fp32. L512 uses the full-strength sequential operator; the
longer lengths use the true-stale chunked operator (the only tractable choice at
4k). It turns out the chunked operator is *not* a handicap on this corpus — see
the full-strength confirmation below, which re-runs 1024/2048 sequentially and
finds the win holds either way.

```
 length  operator            trellis ppl       dense ppl     trellis vs dense
   512   seq (full)          1542 +- 31       1581 +- 131         -2.5%   (tie)
  1024   chunk16 (handicap)  1359 +-  6       1665 +- 114        -18.4%
  2048   chunk16              534 +- 31        616 +-  11        -13.3%   (clean)
  4096   chunk16              546 +-  2        622 +-   5        -12.1%   (clean)
```

Two honest reads. (1) **The win is real and it is a long-context effect.** At
2048 and 4096 the seeds are tight and non-overlapping — Trellis's *worst* seed
beats dense's *best* — so the 12–13% margin is signal, not noise. This turns the
noisy single-seed ≥4k PARTIAL into a clean multi-seed result on a standard
long-range benchmark. (2) **The short-length "clean win" does NOT replicate on
this harder corpus.** At 512, full strength, it is a tie (dense σ=131 — one lucky
seed at 1399); PG19-512 is short-context and undertrained (PPL ~1500), so
bounded memory has little to exploit. The earlier "−32% at 512" was on an easier
corpus. Data: `knlp-key-results/trellis-lm/firmup-20260603/`.

### Full-strength confirmation — and a reversed assumption

The long-length cells above used the chunked operator. To check that the win is
not a chunking artifact, we re-ran 1024 and 2048 with the full-strength exact
sequential operator (Trellis at its strongest), same seeds/steps/tokens:

```
 length  operator      trellis ppl       dense ppl     trellis vs dense
  1024   chunk16       1359 +-  6       1665 +- 114        -18.4%
  1024   seq (full)    1517 +- 91       1635 +-  69         -7.3%
  2048   chunk16        534 +- 31        616 +-  11        -13.3%
  2048   seq (full)     585 +-  9        621 +-  11         -5.9%
```

Two things, one expected and one not. **Expected:** full-strength Trellis still
beats matched dense at both lengths (−7.3%, −5.9%), so the core "beats dense at
length" result is *not* an artifact of the chunked operator — the most important
robustness check passes. **Unexpected:** full strength is *worse* than the
chunked operator here (seq is +11.6% at 1024 and +9.5% at 2048 above chunk16),
the opposite of the assumption that chunked is a conservative handicap. This also
*reverses* the earlier-corpus chunking penalty quoted in §11 (where sequential
beat true-stale chunked). So the chunked operator's quality effect is
**corpus-dependent, not a fixed penalty**: on PG19 the stale-within-chunk write
trains to *better* held-out PPL — regularizer-like — and far more stably (chunk16
σ 6–31 vs the sequential path's σ 9–91). Caveat: lr=3e-3 was not tuned per
operator, and the high sequential variance at 1024 suggests the exact path is
partly under-tuned here, so do not read "chunked is fundamentally better" as
settled — read it as "at matched untuned hyperparameters on PG19, the chunked
operator is at least as good as exact sequential, and the dense-beating result
holds for both." The §11 "+35–46% penalty" should be understood as
corpus-specific, not universal.

## 13. External baselines — the linear-attention family (the sobering result)

"Beats matched dense" is necessary but weak. The pointed question is whether
Trellis's *nonlinear* inner step buys anything over its *linear* cousins —
DeltaNet and Gated DeltaNet, the same gated-delta-rule fast-weight memory with a
linear inner step (added in `trellis_lm/linear_baselines.py`). Same harness,
same untuned lr=3e-3, matched size/tokens/seeds, full-strength sequential, PG19:

```
            L1024 (vs dense)        L2048 (vs dense)
 dense        1635                    621
 trellis      1517  (-7.3%)          585  (-5.9%)
 DeltaNet     1393  (-14.8%)         502  (-19.1%)
 GatedDelta   1188  (-27.4%)         423  (-31.8%)
```

The ordering is **GatedDelta ≪ DeltaNet ≪ Trellis < dense.** Our Trellis is the
*weakest* member of the family: Gated DeltaNet beats it by +27% (1024) and +38%
(2048) at matched size, with far lower variance (σ 2.7 vs 8.6). So the
celebrated "Trellis beats dense" margin is the *smallest* in the family, and a
simpler *linear* model crushes dense by ~5× that margin. The irony is sharp:
Gated DeltaNet is linear in state, so it also gets the cheap exact chunkwise
kernel Trellis cannot have (§11) — it is faster, exact, *and* better quality.

**Why Trellis underperforms here — checked against the paper.** This is a
toy-scale, untuned statement, NOT an architecture verdict, and the Trellis paper
(arXiv:2512.23852) explains why:

- *Scale.* The paper's SMALLEST model is 125M params / 2.4B tokens; there are no
  small-scale results. We ran 4–5M params / ~1–3M tokens — ~25–30× below their
  param floor and ~1000× below their token budget. Even at their smallest scale
  the Trellis edge over Gated DeltaNet is modest: 125M/2.4B Pile-2k 10.87 vs
  11.31 (+0.44 ppl); 350M/7.5B 20.28 vs 21.40 (+1.12). A nonlinear
  test-time-training memory needs capacity + tokens to amortize; at toy scale the
  leaner linear model wins. Our ordering is exactly what their scaling story
  predicts — it neither refutes the paper nor validates our Trellis at scale.
- *Untuned inner loop.* The inner-loop step size γ and forget-decay λ are never
  specified in the paper (no init, no schedule); we used `gamma_init=1e-2`
  untuned. DeltaNet/Gated DeltaNet have no such inner-loop knob, so Trellis is
  uniquely exposed to a mis-set inner lr. Our *outer* lr (3e-3) matches the
  paper's 125M setting. A small γ/inner-lr sweep is the cheapest thing that could
  move our Trellis numbers; the architecture verdict should wait for it.
- *Implementation: mostly faithful.* We match the big choices — one GD step/token,
  single linear memory M, m=64 (their default), forget gate, two-pass, normalized
  -SiLU; their ablation confirms what we found load-bearing (nonlinearity −0.8 to
  −1.8 ppl, forget gate −0.4). No momentum/Muon in either. One possible minor
  discrepancy to check: the paper distinguishes φ (compression activation in
  α≈φ(Mk)) from f (inter-pass normalized-SiLU) and leaves φ's exact form unstated;
  we may use the same function for both.

So the honest verdict is scoped: **at the scale we can actually train, Gated
DeltaNet is the better bounded-memory vehicle** — simpler, faster, exact-chunkable,
lower-variance, better quality. Whether Trellis's nonlinear memory pays off is a
scale + inner-lr question this setup cannot answer. Data:
`knlp-key-results/trellis-lm/firmup-20260603/pg19_L*_linbaselines.json`.

## 14. Where it stands / next

Trellis beats matched dense on PG19 at long context (§12), robustly across
operators — that part is solid. But the external baselines (§13) recontextualize
it: Trellis is the *weakest* member of the linear-attention family, and Gated
DeltaNet beats it decisively (+27–38%) while being simpler, lower-variance, and
exact-chunkable. So the honest headline is not "Trellis is great" but "the
bounded-memory / linear-attention family beats matched dense at long context, and
**Gated DeltaNet is the standout — our Trellis underperforms its own cousins.**"

The chunked-kernel direction is closed: the exact within-chunk solve was built
and proven nilpotent-exact but costs sequential time (§11), and no cheap-and-exact
form exists for the nonlinear inner step. Note that this whole limitation is a
*Trellis-specific* tax: Gated DeltaNet, being linear, has the exact chunkwise
kernel for free — another reason it is the better vehicle.

What remains, reprioritized by §13: (1) **decide whether to keep pursuing our
Trellis at all** — the matched comparison says Gated DeltaNet is the better
bounded-memory model on every axis we measured; the case for Trellis now rests
entirely on the unverified "wins at scale with the full recipe" claim, which is
expensive to test and not clearly worth it. (2) If a bounded-memory model is the
goal, **ship / extend Gated DeltaNet from scratch**, not Trellis. (3) Trellis is
worth keeping only as (a) a faithful reference reimplementation of the paper and
(b) a teaching example of why a nonlinear inner step forfeits the parallel-kernel
free lunch. (4) Remaining hygiene if anyone continues the matched study: more
steps + per-operator lr (400 undertrains ≤1024); kvpress / Mamba2 to round out
the family. The retrofit (Phase 4) remains a weak warm-start.
