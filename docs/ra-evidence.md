# Reciprocal Attention — evidence ledger

This page records what has actually been measured for Reciprocal
Attention (RA) at each model scale, and what each result may be cited
for. It exists because earlier RA pages mixed results of very different
quality; this ledger is the corrective. The mechanism itself is
described in [ra.md](ra.md).

## How results are classified

- **SUPPORTED** — a matched comparison with sufficient controls for the
  narrow claim being made.
- **SUPPORTED_BUT_LIMITED** — the run is usable, but single-seed,
  task-specific, or artifact-incomplete; not sufficient for a general
  claim.
- **NEUTRAL** — a valid experiment whose measured difference does not
  clear its noise floor.
- **NON_DIAGNOSTIC** — the run executed, but its design cannot answer
  the claimed question.
- **INVALIDATED** — a known harness or metric bug means the numbers
  must not be used as evidence.
- **HYPOTHESIS** — mechanistic speculation not established by
  experiment.

A result can be internally valid (the comparison was fair) without
establishing that the effect scales. The two are graded separately
throughout.

## Scaling table

| Scale | Experiment | Result | Status |
|---|---|---|---|
| 30M | GPT-2 TinyStories, 3 seeds, full RA vs "!"-gated RA | −0.47% / −0.46% PPL | SUPPORTED_BUT_LIMITED |
| 124M | GPT-2 vs Qwen-style SDPA gate, FineWebEdu | PPL 72.5 → 68.9, HellaSwag 28% → 30% | SUPPORTED (this config); scaling NOT ESTABLISHED |
| 124M | GPT-2 FIM-trace arm, 3 seeds | best seed −1.87%; multi-seed mean −0.47% | SUPPORTED_BUT_LIMITED |
| 150M | head-count sweeps (1/2/8 heads per layer) | −0.007% / −0.057% / −0.229% loss | SUPPORTED_BUT_LIMITED |
| 150M | 1-hour matched RA-8, 4xH100 | PPL 239.66 → 217.06 (−9.4%) | SUPPORTED_BUT_LIMITED (single seed, artifact-incomplete) |
| 1B | four matched single-seed variants (two selectors, 8 or 28 heads) | per-step deltas −0.80% to +0.52%, σ ≈ 1% | NEUTRAL |
| 8B | cache-transform screen | excluded | INVALIDATED |
| 32B | uniform fixed-head placement, short runs | negative in a short exploratory run | NON_DIAGNOSTIC |

## The validated small-scale result

In a matched GPT-2 small (~124M) FineWebEdu comparison, RA improved
validation perplexity from 72.5 to 68.9 (~5%) and HellaSwag from 28% to
30%, outperforming a Qwen-style SDPA output-gate control (71.8 PPL,
28.5%). This is the project's trusted RA result, from the era when runs
were tracked in Weights & Biases (project `gpt2-ra-sdpa-ablation`).

Scope matters: this supports the narrow claim that RA helped in that
exact GPT-2 configuration. Subsequent 1B experiments did not establish
a statistically clear benefit, so the result must not be extrapolated
as a scaling law.

A separate GPT-2 124M campaign (FIM-trace placement, 1-hour matched)
published its best seed at −1.87%; across the three seeds actually run
the mean was −0.47%, within the project's ~1% noise floor, and a later
on/off ablation of the trained model found the reciprocal branch
contributed slightly negatively (+0.0030 nats when enabled). That
campaign's evidence is real but fragile — it is not the same experiment
as the SDPA-gate comparison above, and neither rehabilitates nor
undermines it.

## The 150M result and its limits

A single matched 150M run (1-hour wall clock, 4xH100, same
FLASH_ATTENTION path on both arms) reported baseline PPL 239.66 vs
RA-8 217.06, a 9.4% improvement. The run was single-seed, and the
retained artifact set is incomplete — no per-step trajectory,
checkpoint, or training log survived, only summary JSON — so the number
cannot be independently revalidated today, and it was not reproduced at
1B. It is recorded here as supported-but-limited evidence, not as a
headline.

## 1B: neutral within noise

Four matched single-seed 1-hour runs at ~1B (TinyLlama-style, 22
layers, GQA) tested two selectors (FIM-trace, Fisher-eigmax of
attention distributions; also a synthetic-query JSD variant) and two
head counts (8 and 28). Mean per-step deltas vs baseline were −0.80%,
+0.45%, +0.49%, and +0.52%, with σ ≈ 1% — all within the noise band.
Raising the head count from 8 to 28 did not unlock a benefit.

A placement caveat explains part of this: at 150M the per-layer FIM
traces spanned ~15x with clear bands, so a "middle band" layer filter
meant something. At 1B the measured spread was only ~1.5x across 22
layers, so the categorical filter stopped discriminating and admitted
almost every layer.

The correct statement is: **no robust benefit was established at 1B.**
Not "RA fails at 1B", and not "RA tops out at 150M" — the experiments
cannot support either.

## What was invalidated

- **The 8B cache-transform screen.** Its intervention was a K-K cache
  transform, not the reciprocal branch; its "RA-N" arms selected layers
  while the code called them heads; its token-agreement metric reused a
  mutable cache; and its reported perplexity came from a fresh model
  call that never consumed the transformed cache. Every number from it
  is excluded from evidence.
- **The `exact_eigmax` head selector.** The batch-mean post-softmax
  attention matrix is row-stochastic, so its largest eigenvalue is
  exactly 1 for every head (Perron-Frobenius); rankings derived from it
  were random tie-breaking. Selections based on it are excluded.
- **An earlier public-page comparison** showing baseline 282.1 vs gate
  223.7 vs RA 50.5 PPL ("82% better") could not be traced to retained
  artifacts and has been removed from all pages.
- **A learned-beta mechanism test (CAPRA)** on GPT-2 124M: after harness
  bugs were fixed, beta stayed near zero, the validation delta was ~0,
  and a random head selection beat the proposed selector. A valid
  negative result for that mechanism hypothesis in that setting.

## 32B: non-diagnostic

A short exploratory run at ~32B applied RA with one fixed head in a
uniform band of middle layers — no validated selector. It trained worse
in that configuration. Because the placement was uniform rather than
selected, it carries no information about whether RA with a working
selector could help or hurt at 32B.

## Closure: the 2026-08 measurement-first program

The line was closed in August 2026 by a preregistered three-stage
program (each stage's gates and predictions were committed before its
compute ran):

1. **Credit audit.** The reciprocal branch's usefulness was measured
   directly as a signed per-head operator (the cross-covariance of
   the output gradient with the reciprocal-vs-standard delta). The
   structure is real — it beats conservative permutation nulls, is
   placement-specific, and reproduces across independent calibration
   data — but it is high-rank (spread over tens of feature
   directions) and almost perfectly sign-cancelling. The cancellation
   explains mechanistically why scalar RA coefficients trained to
   zero effect throughout this program's history: a single scalar
   sees only the trace of the operator, which the opposing modes
   cancel.
2. **Frozen-model gate test.** With the base model frozen, per-mode
   gates in the measured eigenbasis (rank 8-32) consistently improved
   held-out loss across seeds while scalar gates and matched-rank
   random bases did not: the signal exists, and the discovered basis
   matters — on a frozen model.
3. **Matched training matrix.** Three seeds, seven arms, shared
   warmup checkpoints, paired data and matched budgets (~1.5B forked
   tokens per arm at 124M): no reciprocal-attention variant beat its
   paired baseline in all seeds. The original learned-beta
   implementation was statistically indistinguishable from baseline,
   with a final on/off ablation showing the trained branch slightly
   hurting — reproducing the earlier autopsy on fresh runs. The
   frozen-basis spectral gate was consistently worse than baseline:
   a basis measured at the warmup checkpoint goes stale as the model
   co-adapts. One control won large and reproducibly in every seed:
   the Qwen-style SDPA output gate (arXiv:2505.06708), which is not
   a reciprocal mechanism.

**Status: CLOSED.** The historical small-scale measurement stands as
recorded; its mechanism did not reproduce an advantage under
controlled paired conditions, and the measurement-first program
explains why. Reopening would require a genuinely new framing — not
another selector, rank, or gating variant, all of which were tested
to preregistered stop rules.

## Canonical summary

> Reciprocal Attention produced a clear positive result in a matched
> GPT-2 small-scale FineWebEdu experiment, improving PPL from 72.5 to
> 68.9 and HellaSwag from 28% to 30% while outperforming a Qwen-style
> SDPA-gate control. Attempts to extend the result did not establish a
> robust scaling trend: several matched 1B variants were neutral within
> their observed noise, an 8B screen was invalidated by methodology,
> and the exploratory 32B run used uniform rather than validated
> surgical placement. The small-scale result stands; favorable scaling
> remains unproven.
