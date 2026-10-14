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
| `layerrope-mhc-tinystories` | Initial depth screen followed by the fixed seven-row confirmation |
| `layerrope-mhc-tinystories-depth` | Initial 16-cell language-model depth and write-code screen |
| `layerrope-mhc-tinystories-confirmation` | Seven rows, eight paired seeds, depth 32, width 128, 4,000 updates |
| `subliminal-mnist-smoke` | Small CPU runs on real MNIST, four arms, one epoch |
| `layerrope-mhc-tinystories-smoke` | CPU operators, exact resume checks, and two updates for seven rows on synthetic tokens |

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
metrics. Quantization damage uses each checkpoint's own CPU clean baseline
on identical inputs. The rounding experiments simulate stored weights;
they do not measure packed-kernel speed or allocated inference memory.
Block-only INT3 covers about 31.9% of parameters and leaves other weights
FP32. It is not a three-bit model. The original confirmation qualified no
depth-code claim. Numerical misses remain separate from performance, and
the later depth stage is disabled.

Discovery and exact-continuation jobs expose direct module entry points in
`research/mo_variants/`. They require a prepared study directory, frozen job
and source manifests, matching data and the qualified runtime. For example:

```sh
python -m research.mo_variants.job --root /path/to/prepared-study \
  --job-id FROZEN_JOB_ID
python -m research.mo_variants.stage_b_job --root /path/to/prepared-study \
  --job-id FROZEN_CONTINUATION_JOB_ID
```

Jobs save checkpoints under the study's `checkpoints/` directory and retain
source, data-order, model and optimizer checks. Selection and analysis helpers
remain available for discovery work. The make-based confirmation route uses
the fixed seven-row definitions; it does not rediscover or replace nominees.
The upstream reference files retain their licenses and attribution.
