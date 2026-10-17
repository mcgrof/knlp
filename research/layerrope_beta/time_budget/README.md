# Equal training time: LayerRoPE and mHC

This experiment asks whether the faster residual and LayerRoPE models can
match mHC when each family receives the same training time. It starts four
fresh models: residual (`R`), full LayerRoPE (`L`), mHC (`M`), and mHC plus
full LayerRoPE (`ML`). Both LayerRoPE slopes start at zero and remain
trainable. Magnitude and rotation apply at the input and output of both
attention and MLP branches. The earlier [fixed-token study](../README.md)
remains a separate experiment.

Each trajectory runs for 787 synchronized training seconds, retaining states
at 0, 100, 200, 300, 400, 500, 615, 700 and 787 seconds. The clock includes
batch preparation, transfers, forward/backward, AdamW, the 100-update
learning-rate warmup, and uniform numerical checks. It excludes model setup,
offline validation, checkpoint writing and final evaluation. These costs
remain recorded separately, along with complete process and reserved-GPU
time. This measures equal **training time**, rather than equal total rental
cost. A landmark uses the first completed update crossing its target;
overshoot must fit both one update and 1% of the target.

## Reproduce

Use the dependencies in the [reproduction guide](../../reproduction/README.md).
The synthetic CPU route exercises all four models, training, exact resume,
epoch transitions, routing observation, cache rounding and weight/residual
rounding. It does not run the scientific matrix or produce quality findings.

```sh
make defconfig-layerrope-mhc-time-budget-smoke
make CONFIG_RESIDUAL_OUTPUT_DIR=/path/to/smoke-output \
     CONFIG_RESIDUAL_CPU_PYTHON=/path/to/python
```

The full route requires an existing allocation of two matching H100 80GB
GPUs. It never rents a server. Use Python 3.12.11 and PyTorch 2.9.1 with CUDA
12.8, NumPy 2.4.3 and transformers 5.3.0. Keep all generated data, receipts,
checkpoints and reports outside this checkout.

First prepare the pinned public TinyStories training array and its document
receipt using [the data preparer](../../tinystories_depth/data.py). The
[confirmation preparer](../../mo_variants/prepare_data.py) documents the
separate predecessor confirmation partitions. The full-gain beta study
reuses the original TinyStories partitions. Collect the exact-text document
receipts from every predecessor evaluation; this experiment excludes their
training and evaluation texts before creating fresh validation and final
partitions. It does not claim to detect paraphrases.

Create an external input manifest with `dataset_revision` and
`tokenizer_revision` from `research.tinystories_depth.data`, plus:

- `training`: a `{ "path": "...", "sha256": "..." }` receipt for the
  unchanged 40-million-token uint16 `train.npy`.
- `tokenizer` and `validation_parquet`: corresponding file receipts for the
  pinned GPT-2 tokenizer and TinyStories validation shard.
- `documents`: a list of predecessor document-file receipts. Nested
  `text_sha256` entries identify every excluded exact text.
- `predecessor_scopes`: `training-and-original-evaluation`, `full-gain-beta`
  and `depth-code-confirmation`.

Then prepare the data and freeze the card. The seed-audit input is a list
of file receipts for predecessor job/plan manifests containing `seed`,
`*_seed` or `*seeds` fields. Preparation rejects reused proposed seeds.
Record the scope of that audit when reproducing the experiment.

```sh
python -m research.layerrope_beta.time_budget.data \
    --inputs /path/to/INPUTS.json --out /path/to/prepared-data
python -m research.layerrope_beta.time_budget.run prepare \
    --data /path/to/prepared-data --seed-audit /path/to/SEED-AUDIT.json \
    --root /path/to/CARD.json
```

The card freezes source-file hashes, protocol, data manifests and seed
audit before any new model outcome. Training receives only `train.npy`
and `validation.npy`. The separate final array is a protocol seal, not a
security boundary between processes running as the same user.

Create an external resource decision with `authorized: true`,
`action: "qualify"`, `card_sha256`, `gpu_count: 2`, `max_seconds`,
`max_aggregate_gpu_hours` and `max_output_bytes`. Set finite limits that fit
the allocation and leave time for setup and checkpoint export. The aggregate
GPU cap counts both reserved devices, including offline work and idle slots.

```sh
make defconfig-layerrope-mhc-time-budget
make CONFIG_RESIDUAL_TIME_CARD=/path/to/CARD.json \
     CONFIG_RESIDUAL_TIME_RESOURCE=/path/to/QUALIFY-RESOURCE.json \
     CONFIG_RESIDUAL_TIME_ACTION=qualify \
     CONFIG_RESIDUAL_OUTPUT_DIR=/path/to/qualification \
     CONFIG_RESIDUAL_CPU_PYTHON=/path/to/python
```

Qualification runs numerical operator tests on both GPUs, exact
model/optimizer/RNG/cursor resume checks, a full 787-second trajectory for
each row, and the actual endpoint shapes and reduced quantization menu.
Evaluation timing uses repeated validation tokens, never final targets.
Inspect `time-budget/QUALIFIED.json` in the generated run directory and
budget checkpoint storage, offline landmark work, final evaluation, setup
and export before authorizing the matrix. No quality conclusion follows
from these disposable qualification runs.

For the matrix, create another bounded resource decision with
`action: "run"` and `qualification: { "path": ".../QUALIFIED.json",
"sha256": "..." }`. The card, qualified runtime and hardware must match.
Run the same Make command with that decision, a fresh output directory and
`CONFIG_RESIDUAL_TIME_ACTION=run`. The supervisor stops on infrastructure
failure; it does not retry, replace seeds or resume a whole campaign
automatically. Complete-update states retain the model, optimizer, all RNG
states and epoch/order cursor for inspected recovery through
`python -m research.layerrope_beta.time_budget.train --help`.

## Frozen comparisons

All models have 32 layers, width 128, two attention heads, MLP width 352,
GPT-2 vocabulary and 256-token windows. Each update predicts 2,048 tokens.
The five-rate pilot grid is 0.0003, 0.001, 0.003, 0.01 and 0.03, with one
common pilot seed. Each row selects its own rate using validation loss at
787 seconds; ties within 1e-9 choose the smaller rate. An edge winner adds
one common grid extension (0.0001 or 0.1) for all four rows, at most once in
each direction. Numerical failures stay recorded and cannot win.

Eight fresh paired final seeds produce 32 final trajectories. Only after
selection is frozen and every final trajectory has completed may an
evaluation worker open the final holdout. That barrier validates recipe,
source, data, complete landmark timing and paired initial matrices.

At 787 seconds, each checkpoint receives eight cache cases including clean
BF16, signed K3/V3, K3/V2, K2/V3 and K2/V2, affine K2/V2, FP4 K/V and FP8
K/V. Eight additional policies screen all-matrix FP4/INT4/FP8/INT8 weights,
block W4A4/W8A8, and FP4/FP8 residual streams. At 615 seconds, clean,
signed K3/V3, signed K3/V2 and affine K2/V2 provide supporting comparisons.
Each case predicts the same 131,072 held-out next tokens. Cross-entropy is
in nats per token; perplexity is its exponential. Rounding is simulated,
with no packed serving-memory or kernel-speed claim.

Keep absolute quantized loss, damage relative to each checkpoint's clean
loss, and the K3/V2-minus-K3/V3 penalty separate. A cache case passes the
engineering screen only if damage stays at most +0.1 nats in every seed.
The reduced menu covers all declared cases at or below 3,072 ideal bytes
per token, including metadata. If none passes there, report no established
floor at or below that cost; do not infer an untested higher floor.

The ten primary comparisons are R−M, L−M, ML−M, R−ML and L−ML, each for
clean and absolute signed-K3/V2 loss at 787 seconds. The target is the
population median of paired seed differences. The largest of eight paired
differences gives a one-sided upper bound with joint coverage at least
96.09% across the ten comparisons, conditional on independent paired seeds
and this frozen recipe. Bounds below −0.01, zero and +0.01 nats support,
respectively, practical median improvement, directional median improvement
and median noninferiority. Otherwise the result is unresolved. This small
design cannot establish equivalence or a population-mean claim.

Routing constraints are measured on trained mHC states, including at
landmarks and endpoint parity checks. Constraint misses remain in the
results and limit mechanism claims; nonfinite runs are invalid. Historical
fixed-token curves can be summarized with `time_budget.history` without
continuing their checkpoints or equating their older clocks to this clock.
