# Cartridges-at-Scale (CAS) replication harness

This reproduces the validated isolated-cartridge result from *Cartridges at
Scale* (arXiv:2606.04557) on Qwen3-8B / LongHealth. The primary defconfig trains
the five reported patients with the faithful optimizer regime, evaluates the
paper's 2,048-token protocol and an 8,192-token diagnostic, and runs the exact
full-document KV through the same cartridge execution path. Joint
mixed-visibility composition remains an experimental extension in this harness.

The harness wraps the HazyResearch `cartridges` package (pinned and patched by
`bootstrap.sh`) with knlp's Kconfig workflow, so every experimental knob lives in
Kconfig and the driver reads a JSON generated from `.config` — no experiment
policy in shell or Python constants.

## Quick start

```
make defconfig-cas-smoke        # or: make defconfig-cas-paper
# then on a GPU host with vLLM:
research/cartridges_cas/bootstrap.sh
research/cartridges_cas/gen_config_json.py
research/cartridges_cas/run.sh
```

`bootstrap.sh` clones `HazyResearch/cartridges@8cb6823`, installs it (leaving the
CUDA torch untouched), applies the two knlp patches, and drops the CAS scripts in
place. `gen_config_json.py` turns `.config` into `config.json`. `run.sh` runs only
the phases selected by the defconfig. See `PAPER_REGIME.md` for the exact public
reproduction commands and output contract.

## Defconfigs

- `cas-smoke` — a small legacy end-to-end development path for the synthesis and
  experimental composition code.
- `cas-paper` — the validated five-patient H100 isolated reproduction, including
  both completion caps and the exact full-document KV path control.
- `cas-paper-regime-a100` — a one-patient A100 control using the same faithful
  training regime and the 2,048-token evaluation.
- `cas-training-spread-h100` — eight concurrent patient-02 trainings: two seed-42
  determinism controls and seeds 43 through 48, followed by three evaluation
  runs per cartridge in one software session.
- `cartridge-control-screen` — the control-aware fixed-trajectory objective
  screen. The stored synthesis path serializes each target row as
  [sampled token] + [top-k], so under greedy synthesis nearly half the rows
  carry the sampled token twice and the legacy loss consumes both copies — an
  accidental confidence-weighted chosen-token anchor concentrated on
  first-answer-token and end-of-turn rows. This screen decomposes that
  objective one variable at a time: an exact legacy reproduction control,
  dedup-only, dedup scale-matched, per-row anchors on control positions, and
  count/mass-matched anchors on non-control positions — all from one starting
  cartridge, one saved zero-moment optimizer state, and one frozen example
  schedule, gated by a no-training parity arm proving
  legacy == unique + anchors in loss and cartridge gradients. Requires
  `DATA_PARQUET` and `CTRL_CART_INIT` env; evaluation reports strict
  thinking-off generation, thinking-on stress, forced-choice letter scoring,
  and control-state probe diagnostics (`control_screen_eval.json`).
- `cartridge-opt-ablation` — AdamW vs SOAP on stored-target cartridge training,
  matched arms (shared truncation init, same data order, steps, learning rate).
  Cartridge training backprops through a large frozen model to update a small
  KV prefix, so the optimizer step is a small share of total step cost — the
  regime where a second-order optimizer's per-step gains are nearly free. The
  run reports loss at matched steps, a CUDA-synchronized wall-clock split with
  the optimizer-step share, and strict letter accuracy per checkpoint
  (`opt_ablation_report.md`). Needs a stored self-study parquet (`DATA_PARQUET`
  env, or enable the synth phase). SOAP's eigenbasis refresh needs a
  deterministic backward: run on NVIDIA, not on ROCm/W7900. Offline hosts can
  stage records into `RECORDS_DIR` and point `LONGHEALTH_JSON` at a local copy
  of the LongHealth benchmark JSON.

The scale knobs (`CONFIG_CARTRIDGES_CAS_*`) are documented in the Kconfig help.
The dominant quality lever is convos-per-patient: 400 leaves a cartridge below
its no-context floor; ~8000 makes the oracle clearly beat it.

## The two patches (`scripts/apply_pod_patches.py`)

1. **Compiled FlexAttention on CUDA.** Upstream `cartridges` was written to
   `torch.compile` FlexAttention (`dynamic=False, max-autotune-no-cudagraphs` for
   training, `dynamic=True` for generation); the raw kernel is a workaround for
   AMD RDNA3, which cannot compile it. On CUDA sm≥80 this restores the compiled
   path — about a 16× training speedup. Toggle with `CONFIG_CARTRIDGES_CAS_COMPILE_FLEX`
   / env `CARTRIDGES_COMPILE_FLEX`.
2. **Teacher top-k flatten edge-case.** The self-study synthesizer keeps only the
   leading teacher logprobs whose cumulative mass reaches `min_prob_mass`; when a
   confident teacher never reaches it the original `argmax` returns 0 and keeps
   only the top-1 token, silently collapsing the distribution to a hard label. The
   patch keeps all K in that case, so distillation gets the real teacher ranking.

## Eval — read this before trusting a number

The eval is `cas_combine_eval.py`, and two mistakes silently produce garbage:

- **Prompt format.** Use the letter-answer format with `enable_thinking=True`
  (matched to how the cartridge was trained). The `<answer>` format overflows
  Qwen3's thinking budget before the answer and roughly halves and flattens every
  score. The harness eval uses the letter format.
- **Cartridge reconstruction.** A `.pt` stores `trainable_keys` + `frozen_keys`.
  For an *isolated* cartridge `frozen` is a single attention-sink token — load
  `[frozen | trainable]`; the sink is load-bearing, dropping it makes a degenerate
  control prefix. For a *joint* cartridge `frozen` is the distractor cartridges —
  load `trainable-only` (the rescued target); baking the distractors in turns each
  "oracle" into a collapse. The eval auto-detects by frozen token count (`SINK_MAX`).

## Status — what reproduces and what does not

Validated and correct: the isolated-training pipeline, compiled FlexAttention
(~16× training speedup on CUDA sm≥80), the richer-target flatten fix, the
reconstruction/evaluation path, the paper optimizer regime, the five-patient
evaluation, the exact full-document KV path control, and the eight-training seed
spread. The five faithful H100 cartridges average 0.580 at the paper's
2,048-token cap and 0.617 at an 8,192-token diagnostic cap. The full-document KV
scores 0.860 through the cartridge path, matching ordinary full-document
inference at 0.855.

The harness still contains experimental collapse and joint-training phases, but
the public reproduction does not claim a validated composition result. The
current joint trainer is an approximation of the paper's per-sample visibility
rule and is disabled in `cas-paper`.

**Open item: the isolated-cartridge quality gap.** The five-patient mean is
0.580 against the paper's 0.736. The no-context and full-document anchors match
the paper, and exact full-document KV scores 0.860 through the cartridge path,
so cache reconstruction and execution do not explain the gap. Raising the
completion cap to 8,192 moves the mean to 0.617. The remaining measured
differences include self-study data scale (about 4,400 conversations here versus
roughly 40,000 in the paper) and substantial seed-to-seed training variation.

**The optimizer regime matters.** Holding patient 02 and its data fixed, moving
from the earlier small-batch recipe to the complete paper regime raised its
score from 0.55 to **0.65 ± 0.05**. Batch size, peak learning rate, warmup, and
schedule changed together, so this comparison supports the complete regime and
does not attribute the improvement to one knob.

### The frozen single-cartridge baseline

The recipe that produced 0.65 is fixed, and everything downstream (the
five-patient confirmation, the meta-initialization study below) starts from it.
It is `scripts/cas_train_isolated.py` with:

```
PATIENT=patient_02 DATA_PARQUET=<self-study parquet> RECORDS_DIR=<records>
KV_TOKENS=auto KV_DIVISOR=20     # p = ceil(doc_tokens / 20) = 632 for patient_02
LR=0.1 GLOBAL_BS=128 EPOCHS=80   # ends at step 1020 for 4420 conversations
STEPS=5000                       # the linear-decay horizon; EPOCHS stops first
SCHEDULE=linear WARMUP_STEPS=200 WARMUP_MIN_LR=2e-3 ALPHA_F=0.02
```

The cartridge is initialized from the KV state of the first p tokens of the
document under the library's system-prompt template (three header tokens, the
content, then the `<|im_end|>\n` pair), with the library's one frozen
attention-sink token in front. The trainer keeps `STEPS` and the schedule
horizon separate (`SCHED_STEPS`) so a shorter run can stop early on an unchanged
schedule. The library itself never stops at its step limit (it only saves
there and runs on to `EPOCHS`), so the script ends the run itself once `STEPS`
updates have been applied; a run that ends at `STEPS` therefore holds exactly
that many updates. It saves the cartridge at chosen steps (`SAVE_EVERY`,
`SAVE_AT`; one write per step, holding the state after exactly that many
updates), writes the untrained start as `<patient>_init.pt` (`SAVE_INIT`), can
start from any saved cartridge (`CART_INIT`), and prints the held-out
distillation loss on a validation parquet as `VAL_CSV,<step>,<loss>` lines
(`VAL_PARQUET`, `VAL_EVERY`). Cut that validation set with `cas_split_val.py`,
which holds out whole prompt groups: the self-study parquets carry exact
duplicate rows and further rows that share a prompt, and a split by row index
leaks them into the held-out set. Evaluate with `cas_eval_table15.py` in
cartridge mode and at least three runs; a single 20-question run swings by
±0.05–0.10. On one H100 a step at global batch 128 takes about 33 s, so the
full recipe is about nine hours per cartridge.

## Meta-initialization: is part of a cartridge document-agnostic?

Cartridge training is expensive and every cartridge starts from the same
kind of place: the KV state of its own document's first p tokens. If a fixed
part of what training does is the same for every document — the cartridge
learning to condition the model toward the self-study answer distribution, say,
rather than learning its document — that part could be fitted once on trained
cartridges and applied to the start of every new one, saving a slice of every
training run. The study measures the displacement of trained cartridges from
their own starts, tests whether a shared component exists across documents, fits
the simplest correction that predicts a held-out document's displacement, and
then asks the only question that matters: does a cartridge that starts from the
corrected state reach the baseline's held-out loss in fewer steps than one that
starts from the plain state, by more than the run-to-run floor.

The instruments are small pure-torch scripts that read the library's cartridge
files directly:

- `cas_split_val.py` splits one patient's self-study conversations into a
  training set and a held-out validation set, so a loss curve on rows the
  cartridge never trained on is available.
- `cas_cart_init.py` loads, splits and saves cartridge files and provides the
  `KVFromCartFile` initializer the trainer uses for `CART_INIT`.
- `cas_make_init.py` builds the step-0 cartridges the paper ablates: the first
  p tokens of the document, p random tokens of the document, random vocabulary
  tokens, and random vectors. The random-token starts are rulers: a correction
  that lifts them as well is document-agnostic in the strongest sense.
- `cas_cart_loss.py` scores many cartridges on one fixed validation slice with
  the trainer's own per-entry distillation loss, model loaded once, and marks
  each target position as document-informative when the no-cartridge model's
  most likely token disagrees with the teacher's, so a cartridge that only
  learned the format can be told from one that learned the document.
- `cas_kv_rope.py` removes and re-applies the rotary embedding on stored keys,
  which are kept post-rotation at absolute slot positions. Its self-test
  recovers the pre-rotation keys from a real cartridge to a cosine of 0.9997
  (the bf16 floor) and detects a one-slot position error.
- `cas_meta_init.py` is the study itself: `audit` reports where the displacement
  lives (template slots, content slots, keys versus values, fast versus slow
  rotary pairs) and the across-document shared fraction against a
  random-rotation null; `fit` fits a nested family of corrections (a per-head
  bias, a per-slot bias, a per-head gain, a per-head ridge-regularized affine
  map) in a chosen key frame with positive-part James–Stein shrinkage, scoring
  each by leave-one-document-out R² and selecting the simplest family within a
  margin of the best; `apply` writes the corrected start for a new document,
  with sign-flip, key-only, value-only and norm-matched single-donor controls.
- `cas_curve_shift.py` compares the held-out loss curves: for each arm and seed,
  how many steps earlier it reaches the levels the baseline reaches, paired by
  seed, raw and with the step-0 offset removed, against the floor set by
  same-seed replicate runs.

Key displacements are compared in three frames: as stored, de-rotated to slot
positions (the pre-rotation state), and de-rotated by the document's own p (the
frame in which a query placed after the cartridge sees every slot, and the one
in which a displacement driven by such queries is shared across documents of
different lengths). The frame is chosen by leave-one-out score rather than
assumed.

### Faithful `P_iso`

The joint trainer trains all cartridges together with a per-sample
mixed-visibility mask: hold N trainable cartridges in one cache; for each training
example (belonging to document p) make cartridge p always visible and reveal the
target alone 75% of the time, otherwise reveal it alongside `k ~ U(1, N-1)`
sampled distractor cartridges; let gradients flow only into the revealed
cartridges. Because the library block mask cannot express a per-sample random
reveal, the mask is replaced with a full-length reveal-vector lookup keyed by
cartridge slot (`kv_idx // KV_TOKENS`).

The full reproduction — the faithful training recipe, five-patient result,
lossless-path control, implementation, and measured findings — is written up at
[docs/cas.html](../../docs/cas.html). Experiments that extend the paper's method
are collected separately in
[docs/cas-extensions.html](../../docs/cas-extensions.html).
