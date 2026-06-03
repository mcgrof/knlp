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
