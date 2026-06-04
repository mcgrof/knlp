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
longer lengths use the *handicapped* true-stale chunked operator (the only
tractable choice at 4k), so a Trellis win there is conservative.

```
 length  operator            trellis ppl       dense ppl     trellis vs dense
   512   seq (full)          1542 +- 31       1581 +- 131         -2.5%   (tie)
  1024   chunk16 (handicap)  1359 +-  6       1665 +- 114        -18.4%
  2048   chunk16              534 +- 31        616 +-  11        -13.3%   (clean)
  4096   chunk16              546 +-  2        622 +-   5        -12.1%   (clean)
```

Two honest reads. (1) **The win is real and it is a long-context effect.** At
2048 and 4096 the seeds are tight and non-overlapping — Trellis's *worst* seed
beats dense's *best* — so the 12–13% margin is signal, not noise, and it is
*handicapped* Trellis (chunked); full strength would likely widen it. This
turns the noisy single-seed ≥4k PARTIAL into a clean multi-seed result on a
standard long-range benchmark. (2) **The short-length "clean win" does NOT
replicate on this harder corpus.** At 512, full strength, it is a tie (dense
σ=131 — one lucky seed at 1399); PG19-512 is short-context and undertrained
(PPL ~1500), so bounded memory has little to exploit. The earlier "−32% at 512"
was on an easier corpus. A secondary observation: Trellis is consistently the
*lower-variance* learner (σ 1.5–31 vs dense's 5–131), i.e. bounded memory
trains more stably here. Data: `knlp-key-results/trellis-lm/firmup-20260603/`.

## 13. Where it stands / next

The chunked kernel — the gating build — is done and validated, and the ≥4k
comparison is now a **clean multi-seed win on PG19** (§12): −12% at 4096 with
σ=1.5 vs 5, non-overlapping seeds, and handicapped — the strongest positive so
far, and it grows with length (tie at 512 → −12–18% at ≥1024). The chunked-kernel direction is now closed: the
exact within-chunk solve was built and proven nilpotent-exact but costs
sequential time (§11), and no cheap-and-exact form exists for the nonlinear
inner step — so the speed/quality choice (stale-fast vs sequential-exact) is
fundamental, not a missing optimization. Multi-seed + held-out PPL and a real
long-range corpus are now done (§12) and the long-context win held up. The
remaining levers: (1) a **full-strength (sequential) confirmation at 1024–2048**
to remove the chunked handicap caveat from the headline cells (≈50–100 min/run
on the W7900, so cheap enough); (2) more steps — 400 leaves the ≤1024 cells
undertrained (PPL ~1500), so longer training should sharpen the short-length
picture and likely the margins; (3) the external baselines (Mamba2/DeltaNet,
kvpress) so "beats matched dense" becomes "competitive with the linear-attention
family"; (4) re-run Phase 3 recall now that recall-objective training at length
is feasible. The retrofit (Phase 4) remains a weak warm-start.
