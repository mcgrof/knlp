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

## 6. Staged rollout (Phases 1–4) — not yet run

- **Phase 1** TinyStories LM smoke: Trellis vs matched dense at 256/512/1024
  (train loss, val PPL, tok/s, peak memory). `make trellis-smoke`.
- **Phase 2** scratch comparison at matched params/tokens/context across
  512–8192; ablations (no-forget, softmax, m∈{32,64,128}, conv on/off,
  read order). This is where "beats dense at 4k+" is decided. Pod-worthy.
- **Phase 3** RULER/NIAH long-context recall incl. repeated-key; accuracy vs
  context and vs memory bytes. `eval_long_context.py`.
- **Phase 4** retrofit/distillation from GPT-2/SmolLM2 vs KRI-FT and dense PEFT,
  matched trainable params and steps; KL-to-teacher.
- Optional systems/compression baselines (NVIDIA kvpress) and architecture
  baselines (Mamba2/DeltaNet) only with reliable open implementations.

## 7. Recommendation

**Continue.** The architecture is faithful, correct, tested, and bounded, and it
demonstrably binds — the Phase-0 minimum bar. It is currently weaker than dense
on the toy, which is the expected starting point for a from-scratch minimal
implementation and is exactly what Phases 1–2 (matched-budget training +
the diagnostics + the chunked update for speed) exist to resolve. Do not claim
a win until Phase 2 shows scratch-Trellis beating same-size dense at long
context under matched compute/memory, or Phase 4 shows TrellisRetrofit beating
KRI-FT at matched memory/params. Phase 1 (TinyStories smoke, both models) is the
cheap next step and runs on the W7900; Phase 2's long-context sweep is the first
pod-worthy run.
