# Residual-path experiments

These routes reproduce separate MNIST transfer and TinyStories language-model
studies. MNIST tests whether useful digit knowledge crosses three auxiliary
teacher outputs on noise images. TinyStories tests language-model LayerRoPE,
manifold-constrained hyper-connections (mHC), and depth codes added to mHC.
The depth codes are distinct from full LayerRoPE.

Use Python 3.12, PyTorch 2.9.1 and matching torchvision. Install CPU wheels
for CPU checks, or CUDA 12.8 wheels for the fixed H100 confirmation runtime.
Install the remaining dependencies with:

```sh
python -m pip install -r research/reproduction/requirements.txt
```

Load one preset, then run plain `make`. The Make targets run one supervisor,
so `make -j` preserves the preparation, qualification, training, evaluation
and reporting order. A failed qualification prevents measured training.

| Preset | Scope |
| --- | --- |
| `subliminal-mnist` | Reference MNIST assay, historical cohorts, deeper controls, repaired residual-scale, HC write-scale and null controls, four-arm LayerRoPE comparison |
| `subliminal-mnist-null-repaired` | Eighty-epoch identical-target stationarity control with repaired first-order loss |
| `subliminal-mnist-layerrope-study` | Residual, HC, LayerRoPE gain adaptation, and HC plus that adaptation; ten fresh seeds |
| `layerrope-mhc-tinystories` | Initial depth screen, fixed seven-row confirmation, and separate saved-checkpoint KV-cache sweeps |
| `layerrope-mhc-tinystories-depth` | Initial 16-cell language-model screen, then the three-seed residual/mHC cache comparison |
| `layerrope-mhc-tinystories-confirmation` | Seven rows, eight paired seeds, depth 32, width 128, 4,000 updates, then the cache sweep |
| `subliminal-mnist-smoke` | Small CPU runs on real MNIST, four arms, one epoch |
| `layerrope-mhc-tinystories-smoke` | CPU operators, exact resume checks, two updates for seven rows, and cache checks on synthetic tokens |
| `layerrope-mhc-tinystories-kv-cache` | Cache-only evaluation of one or both saved cohorts; no training |
| `layerrope-mhc-tinystories-kv-cache-smoke` | Bounded CPU cache-operator, checkpoint and reporting path on synthetic tokens |
| `layerrope-mhc-tinystories-gsm8k` | First 16 GSM8K questions on saved clean 32-layer residual/mHC/M-O checkpoints; feasibility only |

For example:

```sh
make defconfig-subliminal-mnist-smoke
make
make defconfig-layerrope-mhc-tinystories-smoke
make
```

Smoke output cannot issue a scientific confirmation verdict. The full
cohorts are long experiments. This integration was checked with CPU tests
and bounded smoke workloads, not by rerunning those cohorts. Repaired scale and null controls use seeds 73001–73005; the LayerRoPE
comparison uses 74001–74010, and historical cohorts use 2101–2120. Full
confirmation requires two visible H100 GPUs and `torch==2.9.1+cu128`.
Other full MNIST and initial-depth recipes require an explicitly selected
GPU. An unavailable requested GPU fails before training.

Use `make menuconfig` to set `RESIDUAL_DATA_DIR`, `RESIDUAL_OUTPUT_DIR`,
`RESIDUAL_CHECKPOINT_DIR`, device, CPU interpreter and elapsed-time limit.
The CPU interpreter runs endpoint evaluations in separate processes. Use
a CPU-only PyTorch environment for these evaluations. Targeted MNIST routes
allow an explicit seed override; aggregate and confirmation seed sets stay
fixed. `make residual-plan` prints the selected scope without training.
`EFFECTIVE.json`, `PLAN.json`, `OPERATIONS.json` and `REPORT.json` retain the
effective settings, operation status and incomplete runs. Checkpoints hold
model, optimizer and RNG state. No runner rents compute.

MNIST is downloaded by torchvision. TinyStories preparation downloads the
pinned `roneneldan/TinyStories` Parquet shards and GPT-2 tokenizer, scans all
training texts for duplicate exclusions, and packs 40 million training
tokens. Discovery and confirmation evaluation documents are separate, as
are the confirmation curve and final partitions. Downloads live in the
configured dataset cache. Newly prepared arrays live within the output run.

Historical MNIST presets explicitly retain composed KL loss. Repaired
studies use the analytic student gradient, which cancels exactly for equal
logits. All three auxiliary rows and the unused digit-head invariant remain
in the assay. Full language-model LayerRoPE, its MNIST gain adaptation and
mHC depth codes use different implementations and controls.

Training curve cross-entropy, final GPU loss and final CPU loss are separate
metrics. Weight-quantization damage uses each checkpoint's own CPU clean baseline
on identical inputs. The weight-rounding experiments simulate stored weights;
they do not measure packed-kernel speed or allocated inference memory.
Block-only INT3 covers about 31.9% of parameters and leaves other weights
FP32. It is not a three-bit model. The original confirmation qualified no
depth-code claim. Numerical misses remain separate from performance, and
the later depth stage is disabled.

The separate discovery and exact-continuation coordinators use the open
source `gpuq` scheduler and its real reservation environment. They do not
invent reservation IDs. The reusable modules remain in
`research/mo_variants/`. Their local output plans and saved checkpoints can
be used for new discovery work; the confirmation route uses the already
fixed seven-row definitions and does not rediscover or replace nominees.
The upstream reference files retain their licenses and attribution.

## KV-cache fake quantization

The existing full TinyStories presets now evaluate cache rounding after their
training and weight screens. Set `CONFIG_RESIDUAL_KV_CACHE=n` to omit this
follow-up. The aggregate route retains separate depth-screen and confirmation
data, models and cache reports.

Run the bounded CPU route first:

```sh
make defconfig-layerrope-mhc-tinystories-kv-cache-smoke
make
```

It checks independent scalar quantizer references, token-local statistics,
incremental cached attention, exact noquant parity for the ordinary-residual
and mHC callers, checkpoint loading and all 49 settings on synthetic tokens.
It does not reproduce training or issue a scientific verdict.

To evaluate saved models without training again:

```sh
make defconfig-layerrope-mhc-tinystories-kv-cache
make CONFIG_RESIDUAL_CACHE_DEPTH_RUN=/path/to/depth-run \
     CONFIG_RESIDUAL_CACHE_CONFIRMATION_RUN=/path/to/confirmation-run
```

Supply either directory or both. These are the top-level run directories
printed by the original reproduction supervisor. The first contains
`depth-screen/` and `data/`; the second contains `confirmation/` and its data
link. Checkpoint receipts name their files. If the original paths no longer
exist, the evaluator also checks `depth-screen/checkpoints/JOB.pt` or
`confirmation/checkpoints/JOB/step4000.pt` in the supplied run directory.
Checkpoints written to a separate configured root must remain at their saved
paths or be placed under those directories. Every resolved file is checked
against the original checksum and model-state hash.

The cache-only route requires no H100 or second GPU; select `cuda` or explicitly
select `cpu`. Full confirmation *training* retains its two-H100 qualification.

The clean math sanity check also reuses that saved depth directory:

```sh
make defconfig-layerrope-mhc-tinystories-gsm8k
make CONFIG_RESIDUAL_CACHE_DEPTH_RUN=/path/to/depth-run
```

It evaluates the first 16 official GSM8K test questions using the unchanged
32-layer residual, mHC and M-O checkpoints for seed 75211. It pins the official
dataset revision and content hash and reuses the pinned GPT-2 tokenizer.
Zero-shot `Question: ...\nAnswer:` prompts, greedy decoding, EOS stopping and
at most 128 new tokens define this feasibility check. Prompts are never
truncated; the shared generation budget must leave at least 64 tokens within
the 256-token context. Reports preserve prompts, generated tokens, strict
`####` and permissive last-number scores. This is not the full GSM8K benchmark
or a quantization comparison. A zero clean score cannot establish robustness.

The depth cohort contains ordinary residual and mHC at depths 4/16/32, plus
M-O at depth 32: three paired seeds, 2,000 updates, learning rate 0.003 and
the original final array. Initial common weights, data orders and optimizer
recipes must match between seed pairs. The confirmation cohort contains
seven mHC-derived rows, eight seeds and 4,000 updates, using its separate
final array. It has no ordinary-residual baseline. Never pool these cohorts.

Each checkpoint is evaluated on the full Cartesian product of K and V at
16/8/4/3/2 bits for both quantizers, sharing one clean K16/V16 cell: 49
settings. Unequal K/V precision and the choice of symmetric or affine
codebook are separate controls. This includes both directions of mixed
2/3-bit pairs and K-only or V-only quantization at every low bit width.
Quantization occurs after token RoPE for K and before attention for V.
Statistics use each token/head's 64 channels, with no future-token scale
leakage or unquantized
residual window. Symmetric 2-bit has three levels; affine 2-bit has four and
stores a floating minimum as well as a scale. Quantization arithmetic and
metadata are FP32; dequantized attention values are BF16. Weights, queries,
routing and norm precision remain unchanged.

Reports retain own-clean CE increases, absolute CE, every window loss,
per-seed passes and ideal packed bytes including metadata. The pass boundary
is <=0.1 nats on every seed. Reports show every minimum-byte passing tie,
savings against the best passing equal-bit case, and byte versus worst-seed
damage Pareto frontiers. Equal K/V bit minima alone do not describe the
asymmetric boundary. The expanded grid is exploratory on the reused final
arrays. Three-seed comparisons are descriptive. The eight-seed primary
endpoint is symmetric K4/V4 damage, with six one-sided
paired sign tests and Holm correction. The effect rule also requires a mean
damage decrease of at least 0.005 nats and lower absolute quantized CE. Other
precisions and the affine quantizer are exploratory. Cache qualification
does not clear inherited model numerical flags or establish a new architecture
winner. Fake quantization measures quality, not packed memory savings or speed.

Configure an output directory outside the source checkout. The supervisor
writes `cache-depth/` and/or `cache-confirmation/`, including `PLAN.json`,
`STATUS.json`, `QUALIFICATION.json`, `REPORT.md`, `REPORT.json` and `cells/`.
Generated JSON, checkpoints, logs and experiment artifacts stay in that
external directory; commit only the reproducible source and curated findings.
