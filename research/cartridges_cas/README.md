# Cartridges-at-Scale (CAS) replication harness

This reproduces the core result of *Cartridges at Scale* (arXiv:2606.04557):
you can split a document collection into one trainable KV-cache "cartridge" per
document, but naively combining independently-trained cartridges at inference
collapses accuracy toward chance — and a change in the training rule
(mixed-visibility joint training with distractor cartridges) rescues it to near
the uncompressed oracle. The target is that split → combine → training-rule
story, on Qwen3-8B / LongHealth, reproducible from a knlp defconfig.

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
place. `gen_config_json.py` turns `.config` into `config.json`. `run.sh` runs the
phases the defconfig selected: synthesize self-study corpora, train isolated
cartridges, the combine-at-inference collapse eval, then (paper defconfig only)
mixed-visibility joint training and the rescue eval. Results land as
`collapse.json` / `rescue.json`.

## Defconfigs

- `cas-smoke` — few patients, single-cartridge oracle check, collapse only. Runs
  end to end on one H100 to validate the recipe.
- `cas-paper` — full patient panel, isolated collapse plus mixed-visibility
  rescue.
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

Validated and correct: the pipeline, compiled FlexAttention (~16× training
speedup on CUDA sm≥80), the richer-target flatten fix, the reconstruction/eval
above, the training-contract alignments (per-token distillation reduction,
learning-rate warmup before the first optimizer step, an enlarged packing window),
and **both the collapse and the rescue**.

**Collapse and rescue reproduce.** With five per-document cartridges on Qwen3-8B /
LongHealth, an isolated cartridge that scores 0.58 alone drops to 0.38 (the
no-context floor) when co-loaded, while a mixed-visibility cartridge holds — 0.44
alone, 0.46 co-loaded. The co-load delta flips sign (−0.20 → +0.02); that sign
flip is the rescue, mirroring the paper's larger-N result. Getting this right
requires per-sample cache assembly (most samples present the target alone, a
minority alongside sampled distractors) and a matched per-cartridge budget; a
naive always-resident joint trainer instead teaches only the co-loaded geometry
and inverts the result.

**Open item: the single-cartridge quality gap.** A single isolated cartridge
reaches about 0.50 (best 0.58) against the paper's 0.736. The baselines land on
the paper (no-context 0.39, full document 0.855), and an untrained cartridge
holding the full document KV scores 0.86 through the same path — so the path is
lossless and the ceiling is 0.86. The gap survives every controlled variable:
evaluation protocol, sampler, execution path, training length (loss reaches 0.017
at 80 epochs), data volume, cartridge capacity, initialization, and every
synthetic-distribution reshaping tried (hard question forms, hard-negative entity
binding, direct question-distribution alignment). The load-bearing observation is
a gap between the training objective and free generation: a cartridge minimizes
the teacher-forced distillation loss (~0.035) yet free-generates the correct
answer only ~0.4–0.5 of the time, even on questions it was trained on. The
remaining distance is an objective-to-generation transfer problem, and the levers
left are the paper's exact self-study distribution and its faithful batch-128 /
linear-schedule optimizer regime.

### Faithful `P_iso`

The joint trainer trains all cartridges together with a per-sample
mixed-visibility mask: hold N trainable cartridges in one cache; for each training
example (belonging to document p) make cartridge p always visible and reveal the
target alone 75% of the time, otherwise reveal it alongside `k ~ U(1, N-1)`
sampled distractor cartridges; let gradients flow only into the revealed
cartridges. Because the library block mask cannot express a per-sample random
reveal, the mask is replaced with a full-length reveal-vector lookup keyed by
cartridge slot (`kv_idx // KV_TOKENS`).

The full reproduction — baselines, the lossless-path control, collapse/rescue, the
single-cartridge gap, the deltas against the public implementation, and where the
next gains come from — is written up at
[docs/cas.md](../../docs/cas.md).
