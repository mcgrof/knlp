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

## Reproducibility notes

The eval reconstructs a trained cartridge as `[frozen sink | trainable]`; the
frozen first token (an attention sink) is load-bearing — dropping it turns the
cartridge into a degenerate control prefix. Eval uses the training-matched Qwen3
boundary (`enable_thinking=True`, no injected empty think block). Both are baked
into `cas_combine_eval.py` / `cas_gate_eval.py`.

Full history, the smoke findings, and the reload-bug correction are archived in
`knlp-key-results/cas-replication-20260715/`.

## Status

The recipe is validated at the gate scale (a single cartridge oracle clears its
no-context floor by a wide margin, non-degenerate). The `cas-paper` scale knobs
are the current best recipe and are finalized once the full run confirms
them; treat `cas-paper`'s convos/steps as provisional until then.
