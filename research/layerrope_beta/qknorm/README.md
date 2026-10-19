# Query-Key Normalization: learning and cache rounding

Query-Key Normalization (QKNorm) makes attention depend on query and key
directions and a learned scalar. This experiment asks whether it changes
learning or cache-rounding damage in ordinary residual, full LayerRoPE,
four-stream mHC, or their combination. The implementation and reproduction
routes are available here. A three-seed screen does not establish a population
winner, and no training result is implied by passing the smoke test.

The [paper](https://arxiv.org/html/2010.04245v1) studies low-resource translation.
This implementation uses its per-head-feature L2 normalization and one learned,
unconstrained scalar per attention module, shared across heads. It initializes
the scalar to `log2(256*256-256)`. Norm accumulation uses FP32 and epsilon
`1e-12`. Normalized vectors and the scalar are cast to the query compute dtype;
the query is multiplied by that scalar and scaled dot-product attention uses
`scale=1.0`. Values, weights and the residual mechanisms remain unchanged.

Training and cache evaluation share
[attention.py](../../tinystories_depth/attention.py). Both normalize Q/K after
the head split, apply positional RoPE, then optionally round new K/V once on
cache write. Cached keys contain the normalized, rotated representation. Old
cache entries are neither normalized nor rounded again. The pristine vendored
author files are unchanged. Disabling QKNorm adds no parameter keys.

## Smoke test

From the repository root, with the existing reproduction dependencies installed:

```sh
make defconfig-layerrope-mhc-qknorm-smoke
make CONFIG_RESIDUAL_OUTPUT_DIR="$HOME/qknorm-smoke"
```

This runs CPU attention-reference, gradient, cache, pairing and resume checks,
then synthetic three-update jobs and all ten cache policies for eight arms.
It does not rent a GPU. Outputs must be outside the source checkout.

## Data and frozen inputs

Use the existing [document-excluding data preparation](../time_budget/README.md)
to make a new `training/` capability and a separate `sealed/` final capability.
Add the prior experiment's `DOCUMENTS.json` to the input manifest's `documents`
receipts so previously evaluated documents are excluded too. Training tokens
are the pinned 40-million-token TinyStories array. Validation has 32,768
predicted tokens; final evaluation has 131,072. This is exact-text exclusion
from the supplied receipts, without a paraphrase guarantee.

```sh
python -m research.layerrope_beta.time_budget.data \
  --inputs "$work/DATA-INPUTS.json" --out "$work/data"
python -m research.layerrope_beta.qknorm.run prepare \
  --data "$work/data" --audit "$work/SEED-AUDIT.json" \
  --out "$work/CARD.json"
```

`SEED-AUDIT.json` is a nonempty array of `{ "path": "/path/to/prior-plan.json",
"sha256": "..." }` receipts for predecessor seed manifests. The generator
rejects changed receipts or collisions with final seeds 77811, 77812, 77813
and probe seed 77890. It freezes source file hashes, prepared-data hashes and
the [protocol](protocol.py). No outcome-dependent seed replacement is allowed.

The matrix contains eight arms and three paired seeds. Model geometry is 32
layers, width 128, two heads, head dimension 64, MLP width 352 and context 256.
Each update predicts 2,048 tokens. Learning rates remain 0.001, 0.01, 0.003
and 0.01 for residual, LayerRoPE, mHC and the combination. Each off/on pair
shares initial weights and data order. The scalar has its own zero-decay
optimizer group. Existing parameters retain AdamW, betas `(0.9,0.95)`, epsilon
`1e-8`, zero decay and no clipping. Warmup lasts 100 updates, then LR is constant.
These inherited rates answer a fixed-recipe intervention question; they do
not give both QKNorm states equally tuned optima.

Each job receives 787 synchronized training seconds. Batch preparation,
transfers, forward/backward, optimizer, warmup and uniform online checks count.
Offline validation, attention diagnostics and checkpoint I/O have separate
clocks. A checkpoint at exactly 1,000 updates supplies the equal-token endpoint.
If it is not reached before the time budget ends, it is missing, not extended.
Normal time landmarks retain validation curves. Initialization and both primary
endpoints retain full model, optimizer, RNG, cursor and clock states. Checkpoints
are saved atomically and must be copied to persistent storage during the run.

## Qualify and train

The new route uses one already allocated CUDA GPU, one worker at a time. It
does not allocate cloud resources. The older two-GPU time-budget route is
unchanged. Qualification uses training/validation data and a disposable probe
seed; it does not open final targets. It checks full-depth pairing, actual SDPA
backend operations, precision, timing, memory, no-rounding parity and every
cache policy. Runtime details and workload projections go into its receipt.

Set `work` to your external run directory. `QUALIFY-RESOURCE.json` must contain:

```json
{
  "authorized": true,
  "action": "qualify",
  "card_sha256": "SHA256 of CARD.json",
  "gpu_count": 1,
  "max_seconds": 3600,
  "max_aggregate_gpu_hours": 12,
  "max_total_usd": 60,
  "max_output_bytes": 90000000000
}
```

These are maximum admitted limits, not automatic spending authorization. Keep
your own smaller allocation decision and expiration in `deadline_epoch` when
needed. Qualification and setup also consume that allocation. After the probe,
compare the projected remaining matrix, offline diagnostics, checkpoint export
and reserve with the actual remaining lifetime and cost allowance. Stop if it
does not fit; do not drop policies or seeds after seeing outcomes.

```sh
make defconfig-layerrope-mhc-qknorm
make CONFIG_RESIDUAL_TIME_ACTION=qualify \
  CONFIG_RESIDUAL_TIME_CARD="$work/CARD.json" \
  CONFIG_RESIDUAL_TIME_RESOURCE="$work/QUALIFY-RESOURCE.json" \
  CONFIG_RESIDUAL_OUTPUT_DIR="$work/qualification"
```

For training, use a separate `RUN-RESOURCE.json` with action `run`, a finite
process limit, the same card hash, and a `qualification` object containing
`path` and `sha256` of the emitted `QUALIFICATION.json`. Keep the qualification
files available. The runtime must match. Then:

```sh
make CONFIG_RESIDUAL_TIME_ACTION=run \
  CONFIG_RESIDUAL_TIME_CARD="$work/CARD.json" \
  CONFIG_RESIDUAL_TIME_RESOURCE="$work/RUN-RESOURCE.json" \
  CONFIG_RESIDUAL_OUTPUT_DIR="$work/results"
```

The supervisor orders training before final evaluation. Every run must have a
closed status and matching source, recipe, initial parameter pairing and
checkpoint hashes before final data can be opened. Numerical failures remain
in the report. Incomplete execution stops the route; it does not silently retry
or replace a seed. The saved state supports explicit trainer resume; source,
runtime, recipe, data, optimizer, RNG, cursor and clock must match.

## Evaluate saved checkpoints without retraining

The route evaluates unchanged BF16 K16/V16; signed K3/V16, K2/V16, K16/V3,
K16/V2, K3/V3, K3/V2, K2/V3 and K2/V2; and floating-offset affine K2/V2.
Signed quantization uses one FP32 scale per token/head/side; affine uses a scale
and offset. These are token-local, dequantized simulations. Actual cache tensors
remain BF16. Ideal bytes include metadata and are derived from actual model
geometry; they are not measured serving-memory or speed savings. The learned
scalar is model overhead, not per-token metadata.

Use the existing external `qknorm/` run directory containing `training/` and
`evaluation/`. A cache resource decision has action `cache` and the same
qualification receipt. Completed cells are verified and reused; missing cells
resume evaluation without training:

```sh
make defconfig-layerrope-mhc-qknorm-kv-cache
make CONFIG_RESIDUAL_TIME_CARD="$work/CARD.json" \
  CONFIG_RESIDUAL_TIME_RESOURCE="$work/CACHE-RESOURCE.json" \
  CONFIG_RESIDUAL_CHECKPOINT_DIR="$existing_qknorm_run" \
  CONFIG_RESIDUAL_OUTPUT_DIR="$work/cache-receipts"
```

The outputs separate absolute CE and perplexity from own-clean rounding damage,
key-only damage, and K3/V2 minus K3/V3 value-bit sensitivity. A cache floor
requires one fixed policy to pass every observed seed. The own-clean tolerance
is 0.1 nat. The common-quality reference is the paired fresh residual/off clean
CE plus 0.1 nat at the same endpoint. No passing policy means no qualifying
floor. A quantized pass when clean fails is flagged for confirmation.

`CELLS.csv` retains every seed and policy. `SUMMARY.json` reports paired on-minus-
off effects D, effects relative to residual I, and the mHC-minus-LayerRoPE effect
J, with all values, mean, median and range. Learning curves use both tokens and
seconds; CE/byte plots retain absolute quality. Failed or censored seeds do not
become independent token or layer replicates. Any promising screen needs a
separately approved, equally tuned confirmation. This route starts no follow-up.
