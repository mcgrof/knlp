# LLaMA-150M matched RA audit

Date: 2026-03-27

## Current state

The matched LLaMA-150M reciprocal-attention lane now exists as a standalone
harness under `fim/reciprocal_attention/` and has been exercised locally and
on the real multi-GPU target lane.

## What exists in `knlp`

- The production training path is still GPT-2-centric (`gpt2/train.py`,
  `gpt2/model.py`, `gpt2/model_knlp.py`).
- The matched LLaMA lane therefore lives in a dedicated harness rather than in
  the old GPT-2 trainer.
- The harness uses the SDPA family for both baseline and RA.
- `torch.compile` is not used.
- DDP startup, short train/eval loops, backend probing, and wall-clock stop via
  `training.max_time` are wired.
- A real FIM-derived surgical selection file now exists:
  - `configs/ra_surgical_llama150m.json`
  - `configs/ra_surgical_llama150m_top4.json`

## Remaining production gaps

These are still true in the main GPT-2 trainer path:

- No first-class `llama-150m` model in the production trainer.
- No LLaMA defconfig in `gpt2/defconfigs/`.
- No surgical per-head RA selection path wired into the production KNLP
  attention module.
- Final result bundle is not yet checked into key-results.

That is why the experiment currently runs through the standalone matched
harness instead of the legacy production path.

## Files added for the matched lane

- `fim/reciprocal_attention/llama150m_matched.py`
- `fim/reciprocal_attention/configs/llama150m_baseline_smoke.json`
- `fim/reciprocal_attention/configs/llama150m_fim_smoke.json`
- `fim/reciprocal_attention/configs/llama150m_ra_surgical8_smoke.json`
- `fim/reciprocal_attention/configs/llama150m_ra_surgical4_smoke.json`
- `fim/reciprocal_attention/configs/llama150m_baseline_b200x4.json`
- `fim/reciprocal_attention/configs/llama150m_fim_collection_b200x4.json`
- `fim/reciprocal_attention/configs/llama150m_ra_surgical8_b200x4.json`
- `fim/reciprocal_attention/configs/llama150m_ra_surgical4_b200x4.json`
- `scripts/run_llama150m_matched.sh`
- `configs/ra_surgical_llama150m.json`
- `configs/ra_surgical_llama150m_top4.json`

## Clean-comparison properties

- Baseline and RA both use the same SDPA-family attention path.
- Baseline honors `ra.enabled=false` and does not silently load the surgical
  selection file.
- Backend summary is logged explicitly, including separate standard vs
  reciprocal call probes.
- RA only mixes an additional reciprocal SDPA output on selected heads.
- FIM/proxy attention stats emit a generated surgical selection JSON.
- The 1-hour matched runs stop on wall-clock via `training.max_time`.
- The harness now emits `wallclock_mismatch_warning` and exits nonzero if a run
  hits `training.train_steps` before `training.max_time`, so an accidentally
  step-limited launch does not silently masquerade as a wall-clock-matched run.

## Smoke status

### Local CPU smoke

Validated in `~/devel/knlp` on CPU:

- single-process baseline smoke: pass
- single-process FIM smoke: pass
- single-process RA-8 smoke: pass
- DDP baseline smoke (`world_size=2`, gloo): pass
- DDP RA-8 smoke (`world_size=2`, gloo): pass
- backend parity: pass (`impl=sdpa`, baseline standard=MATH,
  RA standard=MATH, RA reciprocal=MATH)

Artifacts:

- `out/llama150m-matched/llama150m-baseline-smoke.*`
- `out/llama150m-matched/llama150m-fim-smoke.*`
- `out/llama150m-matched/llama150m-ra-surgical8-smoke.*`

### Cloud DDP smoke

Validated on the live 4xH100 RunPod target lane:

- DDP startup: pass (`world_size=4`)
- target bf16 smoke at `seq_len=1024`: pass
- backend parity: pass with `actual_backend=FLASH_ATTENTION`
  for both baseline and RA through the same SDPA-family dispatch path
- no `torch.compile`

## Cloud selection and launch status

- Prime Intellect tooling was missing on `prune` (`prime` CLI not on PATH), so
  the lane fell back to RunPod.
- The active target is the 4xH100 pod `llama150m-ra-matched-1774649959`.
- The original target configs were too large because this harness interprets
  `training.batch_size` per rank. The pre-fix `batch_size=128` configs OOMed.
- The corrected target configs now use:
  - `batch_size=8`
  - `gradient_accumulation_steps=4`
  - `train_steps=50000` for the 1-hour baseline/RA runs so wall-clock, not
    max-steps, is the binding stop condition on the active H100 lane
  - effective global batch 128 on 4 GPUs
- The real FIM collection completed on the cloud lane and regenerated:
  - `configs/ra_surgical_llama150m.json`
  - `configs/ra_surgical_llama150m_top4.json`
- After the fix, the expensive launch sequence was restarted.
- The corrected 1-hour baseline finished cleanly on wall clock with
  `exit_reason=max_time` at `elapsed_s=3600.065` and `completed_steps=26432`.
- The launch sequence then advanced automatically into the 1-hour RA-8 run on
  the same 4xH100 pod.
- RA-8 also finished cleanly and the optional RA-4 variant is now actively
  running on the same pod. A direct pod check at ~2026-03-28 03:30 UTC showed
  all 4 H100s busy and the latest RA-4 eval at `elapsed_s=1452.016`,
  `step=10350`, `perplexity=189.91`.
- The local mirror was re-synced into `prune:/data/knlp` after that check so
  the harness/config/docs there match the fresher development copy.

## Most recent concrete results

FIM collection completed on the cloud lane in ~15m57s wall clock.

The paired 1-hour comparison is complete on 4xH100:

| Arm      | Final PPL | Steps  | exit_reason | Backend         | parity_ok |
|----------|-----------|--------|-------------|-----------------|-----------|
| Baseline | 239.66    | 26 432 | max_time    | FLASH_ATTENTION | true      |
| RA-8     | 217.06    | 25 702 | max_time    | FLASH_ATTENTION | true      |

RA-8 achieves a 9.4% perplexity improvement under identical conditions.

Elapsed-accounting nuance: the RA-8 completion event originally recorded
`elapsed_s=4200.111` even though the last eval event showed
`elapsed_s=3600.002`. The ~600s gap was post-stop teardown and DDP barrier
time, not additional training. A harness fix now emits `stop_elapsed_s`
(training stop time) and `total_elapsed_s` (including teardown) separately in
the completion event, with a `teardown_warning` field when the delta exceeds
5 seconds.

Generated surgical selections:
- 8-head set: L1 [0,7], L2 [7], L4 [0,1,2,3,6]
- top-4 set: L1 [0,7], L2 [7], L4 [6]

## Reproducibility entrypoints

The shell script `scripts/run_llama150m_matched.sh` is the single
entrypoint for all matched-lane work. It supports smoke tests, individual
production runs, and a full-sequence mode that chains the intended
pipeline without ad-hoc pod glue.

### Full pipeline (one command)

Run the complete FIM → top4 derivation → baseline → RA-8 sequence:

    scripts/run_llama150m_matched.sh full-sequence

Or include the optional RA-4 variant:

    scripts/run_llama150m_matched.sh full-sequence-all

### Top-4 derivation

The top-4 selection file is now derived automatically as part of
`full-sequence`. It can also be run standalone:

    scripts/run_llama150m_matched.sh derive-top4

This calls the harness with `--derive-topk 4`, which reads
`configs/ra_surgical_llama150m.json` and writes
`configs/ra_surgical_llama150m_top4.json`.

### Wandb guardrails

Local reruns of production configs no longer accidentally log to wandb
online. The shell script honours these env vars:

    WANDB_MODE=offline scripts/run_llama150m_matched.sh full-baseline
    WANDB_DISABLED=1   scripts/run_llama150m_matched.sh full-sequence

### DDP / GPU overrides

    NPROC_PER_NODE=2 scripts/run_llama150m_matched.sh ddp-baseline
    TORCHRUN=/path/to/torchrun scripts/run_llama150m_matched.sh full-ra8

## Harness fix: elapsed accounting (2026-03-27)

The completion event now distinguishes training-stop wall-clock from total
process wall-clock. The `complete` event includes:

- `stop_elapsed_s`: wall-clock at the moment training stopped (max_time or
  max_steps boundary).
- `total_elapsed_s`: wall-clock after teardown (barrier, wandb finish, DDP
  cleanup).
- `elapsed_s`: alias for `stop_elapsed_s` (backward-compatible).
- `teardown_warning`: emitted when `total_elapsed_s - stop_elapsed_s > 5s`.

This prevents future confusion like the RA-8 run where the completion event
showed ~4200s despite training stopping at ~3600s.

## Next steps

1. Let the active RA-4 run finish, then record its final PPL / backend / wall-clock outcome.
2. Export baseline + RA-8 + RA-4 logs/artifacts and check the final result bundle into key-results.
3. Reconcile any remaining local-vs-cloud drift, especially completion-event elapsed-accounting fields in already-produced cloud artifacts.
