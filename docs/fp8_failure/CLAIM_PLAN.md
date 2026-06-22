# FP8 KV-cache failure atlas — claim ledger

This is the frozen list of scientific claims the atlas makes, each with its supporting experiment,
its negative control, the condition that would refute it, the measurement level it requires, and its
current verdict. The point is to fix what we are testing *before* reading paper-tier results, and to
never quietly move the goalposts. It also captures the open questions and the derived
recommendations so no idea from the design brief is lost.

Verdicts: **supported** / **partially-supported** / **refuted** / **not-tested**. A claim is only
"supported" once its negative control has also run and behaved.

## Claims

**C1 — `attention_bias=True` does not predict FP8-K failure.**
- Supporting: per-model `has_k_bias` flag vs K8/V8 outcome across the fleet.
- Control: Qwen2.5-14B carries a K-bias yet tolerates FP8-K; biasless GPT-J still degrades (material).
- Refuted if: the flag separates tolerant from catastrophic models cleanly.
- Level: fake_quant_teacher_forced. Verdict: **supported** (fleet + the K-bias-*magnitude* finding —
  it is the bias magnitude / where it sits, not the flag; see [[project_kbias_magnitude_fp8]]).

**C2 — Large K-bias *magnitude* causally explains the Qwen-style failure.**
- Supporting: per-tensor FP8-K catastrophic on biased Qwens; pre-bias recovery ~0.97 (Qwen2.5-7B).
- Control (MISSING): an **alpha dose-response** — scale the K-bias by α∈{0,0.25,0.5,1,1.5,2} and show
  the FP8-minus-native error tracks α; and pre-bias must beat a **random same-norm** and a
  **channel-permuted** bias subtraction (not just any subtraction).
- Refuted if: FP8 error does not track α, or random/permuted bias recovers as well as the true bias.
- Level: fake_quant_teacher_forced + activation. Verdict: **supported** (2026-06-21 controls: true
  bias recovers 0.97, random-same-norm 0.05, permuted 0.05 → bias-SPECIFIC; alpha dose-response
  0.52@α=0 → 3.28@α=1 → magnitude-driven).

**C3 — Partial-RoPE rotary/pass-through scale mismatch causally explains the dominant Phi-2 K-side
failure.**
- Supporting: Phi-2 pass-through-only K8 reproduces almost all the error (err 0.662 of 0.674);
  pass-through bias removal recovers only ~13% → it is the pass-through *distribution*, not the bias.
- Control (MISSING): the true architecture-aligned 32/48 split must beat **20 random 32/48
  partitions** and **architecture-misaligned contiguous** splits, and an **artificial same-size split
  on a full-RoPE model (Mistral/Llama)** must NOT show the effect.
- Refuted if: random partitions recover as well as the true split.
- Level: fake_quant_teacher_forced. Verdict: **supported** (2026-06-21: the TRUE pass-through tail
  damages more than 19/20 random same-size partitions — 95th percentile — architecture-aligned; and
  Phi's true-bias recovery does NOT beat random + flat alpha → not bias).

**C4 — Qwen and Phi are distinct mechanisms.**
- Supporting: Qwen recovery 0.97 (bias) vs Phi 0.13 (distribution); gauge beats per-channel on Phi
  not Qwen; GPT-J (biasless partial-RoPE) localizes to the *rotary* subspace (inverse of Phi).
- Refuted if: the same repair optimally fixes both.
- Level: fake_quant_teacher_forced. Verdict: **supported**.

**C5 — K16/V8 is a strong first fallback for K-side failures, but is NOT universally lossless:
Phi-2 retains MATERIAL residual error under K16/V8.**
- Supporting: K16/V8 tolerant for Qwen/Pythia/Mistral/GPT-J; **material for Phi-2** (top1 ~0.95).
- Open question (MANDATORY, see below): isolate Phi's V-side residual.
- Refuted if: Phi's K16/V8 residual disappears under a correct measurement (then it was an artifact
  and K16/V8 *is* lossless for Phi).
- Level: fake_quant + hf_dynamic_cache. Verdict: **RESOLVED** (2026-06-21: no V granularity/format/
  layer rescues Phi-2's K16/V8 and the residual WORSENS in the incremental cache (AR agreement 0.44)
  → it is a **real V-side sensitivity, not an artifact**. Recommend **native V (or native KV) for
  Phi-2**. K16/V8 is lossless everywhere tested EXCEPT Phi-2.)

**C6 — Mechanism-specific repairs recover more compression when correctly matched.**
- Supporting: pre-bias FP8 for Qwen (0.97); QK-gauge beats per-channel on Phi; per-channel/pre-bias
  beat the gauge on Qwen.
- Control: every repair compared against **per-channel K** (not only per-tensor) — already enforced.
- Refuted if: a generic per-tensor/per-channel baseline matches the matched repair everywhere.
- NEW repair (2026-06-21): **INT8-K rescues Qwen where FP8 fails** (top1 0.30→0.96) — the failure is
  FP8-format-specific (mantissa crushed under a bias-dominated scale), not bit-width; e5m2 is worse.
  So int8-K is a mechanism-matched repair for the bias mechanism, and finer FP8 granularity is NOT
  (per-channel stays catastrophic).
- Level: fake_quant_teacher_forced. Verdict: **partially-supported** (gauge/pre-bias/int8-K shown;
  split-scale recovery for Phi + the Pareto vs storage/metadata cost NOT yet tabulated).

**C7 — A cheap release-time preflight can identify unsafe FP8 layouts on held-out models.**
- Supporting: NONE yet.
- Control: leave-one-family-out; the key metric is the **false-safe rate** (declaring a fragile model
  safe is the foot-gun).
- Level: fake_quant on holdout families. Verdict: **not-tested** (heuristic decision-tree scanner +
  holdout eval NOT yet built; per Codex keep it a tool, not a statistical centerpiece).

## Open questions (must be resolved or explicitly left unknown)

1. **Phi-2's V-side residual under K16/V8** — real V sensitivity / scale-granularity / numerical
   (attention-accumulation) / fake-quant artifact / unknown? Probe: V per-tensor vs per-head vs
   per-channel vs per-group, e5m2-V, prefill-only vs incremental cache, per-layer V sensitivity, a
   second reference quantizer. Recommendation must follow the result (V-specific scale, or native V,
   or "artifact → K16/V8 IS lossless for Phi").

## Derived "DO NOT" guidance (confirm empirically; some already confirmed)

- Do not use `attention_bias=True` as a failure predictor (C1, confirmed).
- Do not assume biasless models are safe (GPT-J degrades; confirmed).
- Do not infer safety from family or parameter count (non-monotonic in scale; confirmed).
- Do not compare a new repair only against per-tensor FP8 (the gauge-vs-per-channel trap; confirmed).
- Do not claim QK equalization fixes DC-offset bias (it loses to pre-bias/per-channel on Qwen;
  confirmed).
- Do not call K16/V8 universally lossless while Phi's V residual is open (C5).
- Do not make FP8 *throughput* claims from an Ampere A6000 (no native FP8 Tensor Cores) — performance
  claims come from the H100/H200 decode-paper stack only.
- Do not validate only at 2K, or only teacher-forced (long-context + AR still owed).

## Recovery playbook (derived)

- Qwen-style high-K-bias-magnitude: true pre-bias K FP8; per-channel K where practical; K16/V8 fallback.
- Phi-style partial-RoPE pass-through-distribution: rotary/pass-through split scaling; QK-gauge if it
  beats per-channel; per-channel K; native-K fallback; **plus a V-specific repair if Phi's K16/V8 V
  residual is real**.
- Unknown K-side failure: preserve K, test V separately, native KV if V also fails.

## Stop conditions (from the brief)

>=12 checkpoints / 6 families in the atlas; core results with 3 seeds + CIs; Qwen causal controls
(alpha + random/permuted) complete; Phi 20-random-partition control complete; Phi K16/V8 residual
classified or explicitly left unknown; >=2 historical partial-RoPE checkpoints (have: Pythia sweep +
GPT-J + NeoX); repairs compared vs per-channel; long-context for both mechanisms; incremental-cache
correctness; >=4 holdout checkpoints for preflight; recovery methods include storage/metadata cost;
every claim has a verdict; raw per-prompt data + manifests preserved.

## Current coverage snapshot (2026-06-21)

Done: atlas (smoke/fleet/legacy), KV isolation, Qwen bias + pre-bias recovery, Phi subspace +
de-confounding, QK-gauge (+per-channel +clamp sweep), AR-divergence (per-token + margin), GPT-J
biasless control, granularity/format(+e5m2)/layer sweeps (phase-7), serving validation by reference
(phase-10), statistics module, 45 CPU tests. Pod spend ~$2.

Owed (this batch): **Phi V-residual probe** (C5), **negative controls** (C2 alpha + random/permuted
bias; C3 20-random-partition; C6 random-condition gauge). Owed (later): long-context 8K/16K,
multi-seed+CIs (needs the seed→prompt-independence fix in calib_prompts), preflight scanner (C7),
consolidated paper artifacts.
