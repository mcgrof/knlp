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

## Status (2026-07-18) — what works and what does not

Validated and correct: the pipeline, compiled FlexAttention (~16× training
speedup on CUDA sm≥80), the richer-target flatten fix, the reconstruction/eval
above, and **combine-at-inference collapse**. The training itself is sound.

**Not yet reproduced: the collapse→rescue headline.** The funded Tier-2 run (8
patients, 8000 convos each; `results-archive/cas-replication-20260715/tier2-20260718/`)
found two blockers:

1. **Oracle strength does not generalize at 8000 convos.** One easy patient hits
   ~40% but the 8-patient mean oracle is ~12.5% — most cartridges stay near their
   floor, so the collapse is shallow and there is no strong ceiling to rescue
   toward. **Convos-per-patient is the binding lever**; the paper's regime is
   ~40000. Raise `CONFIG_CARTRIDGES_CAS_CONVOS_PER_PATIENT` accordingly and expect
   the synthesis phase to dominate wall-clock (~5 h/patient of vLLM generation at
   40k). The reusable 8×8k corpora are archived if you want a warm start.
2. **The joint (rescue) arm is a known-limited approximation.** `cas_train_joint`
   trains a target with the other cartridges present as *frozen, already-trained*
   distractors (a fixed-distractor, always-joint stand-in for CAS's per-sample
   `P_iso`). In practice the combined joint cartridges collapse to ~0% — it does
   not teach coexistence. **A faithful `P_iso` is required and is the main TODO.**

### Faithful `P_iso` (the rescue TODO)

CAS trains all cartridges jointly with a per-sample mixed-visibility mask: hold N
trainable cartridges in one cache; for each training example (belonging to patient
p) make cartridge p always visible and each other cartridge visible with
probability `1 - P_iso`; let gradients flow only into the visible target; the
cartridges co-evolve. This is a custom training loop over the existing `seq_ids`
flex-attention masking (a cartridge is hidden for a step by setting its cache
`seq_ids` to a sentinel ≠ −1), not the stock single-cartridge trainer. Until this
lands, treat the rescue result as unmeasured, not as a negative for the method.

Full history — smoke, the reload-bug correction, and the Tier-2 findings — is in
`results-archive/cas-replication-20260715/`.
