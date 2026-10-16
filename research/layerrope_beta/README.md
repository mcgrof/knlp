# Full input/output gains and beta initialization

This study asks whether full LayerRoPE gains change learning, residual
distributions or tolerance to quantization. The historical MNIST adaptation
uses only input gains and initializes its magnitude slope at zero. The
language-model implementation already has both input and output gains at
the attention and MLP sites. These are distinct starting points.

The full matrix has completed GPU qualification, fourteen MNIST rows and
twenty-four final language-model runs. The [curated results](https://knlp.io/layerrope-mhc-quantization.html#author-followup)
keep these fresh cohorts separate from the earlier depth-code confirmation.
The language-model comparison has three paired seeds and remains descriptive.
All nine MNIST suppression comparisons fail; the mHC rows also fail the
tracked routing-stability gate.
Checkpoints, generated reports and raw measurements stay in the configured
external directories.

The [author's reference](https://www.cs.rochester.edu/u/ssrivas9/layer-rope/#code)
shares a gain vector across depth at each site and transforms adjacent gain
entries as complex pairs. Log magnitude and log angular scale are affine
functions of log depth. Both initial slopes take each of -1, -0.5 and 0;
intercepts start at zero and the frequency base is 100. All schedule
parameters remain trainable. An output gain multiplies the branch directly
before the residual or routed write, without an output normalization.

## Run

The CPU smoke preset checks actual gains, two optimizer updates, checkpoint
reload, residual observation and all floating/integer quantization paths on
synthetic tokens. It cannot produce a scientific result.

```sh
make defconfig-layerrope-mhc-beta-smoke
make
```

The full preset needs an existing allocation of two H100 GPUs and PyTorch
2.9.1 with CUDA. It does not rent a server. Install the dependencies from
the [reproduction guide](../reproduction/README.md), including pytest for the
numerical qualification. Configure external cache, output and checkpoint
directories in `make menuconfig` or pass the corresponding Make variables.

```sh
make defconfig-layerrope-mhc-beta
make CONFIG_RESIDUAL_OUTPUT_DIR=/path/to/runs \
     CONFIG_RESIDUAL_CHECKPOINT_DIR=/path/to/checkpoints \
     CONFIG_RESIDUAL_CPU_PYTHON=/path/to/python
```

Inputs use the same pinned public TinyStories/GPT-2 preparation and MNIST
download as the existing presets. An optional `tinystories-prepared/`
directory inside the configured dataset cache can supply the existing
`train.npy`, `curve.npy`, `final.npy`, `DOCUMENTS.json` and `DATA.json`.
Every inherited array/document file is checked against that manifest.
New seed orders are generated in the new run; source and input hashes are
frozen before training. Otherwise the study prepares public inputs again.

## Comparisons

The language-model matrix has ordinary residual and mHC controls, plus full
LayerRoPE and its mHC combination at each beta initialization: eight rows,
all at depth 32, width 128 and a 256-token context. Each row receives the
same four-learning-rate, 500-update pilot budget. Mean curve cross-entropy
at steps 300, 400 and 500 selects a learning rate; ties select the smaller
rate. Invalid pilots stay recorded and cannot win. The final matrix uses
three fresh paired seeds and 2,000 updates per row. This is a descriptive
comparison of equally tuned families, not an isolated fixed-learning-rate
causal estimate or independent confirmation on a fresh evaluation set.

The MNIST matrix uses residual, HC and mHC controls, each with the three
full input/output-gain variants. Residual and HC also retain input-only
rows to bridge the historical adaptation: fourteen rows and ten fresh
paired seeds. Common linear weights, independent-control initializations,
noise and batch orders are paired. Each row runs five teacher epochs and
80 student epochs with direct targets and the exact-cancellation loss.
Identical-target stationarity, an active signal, an independent-initialization
control and unused-head invariants must pass before measured training.

The nine full-gain-versus-same-architecture MNIST contrasts form one family.
The frozen paired bootstrap confidence level is 1-0.05/9. A suppression
candidate needs at least five percentage points less transfer gain over
its own initialization, a transfer interval wholly below zero, preserved
teacher and all-output student utility within two points, and an upper
epoch-to-95%-accuracy ratio no greater than 1.10. Input-only comparisons
are supporting evidence. A lower endpoint alone is insufficient.

## Measurements and limits

Story cross-entropy predicts actual next tokens on the fixed final array;
perplexity is its exponential. Every final checkpoint receives the full
49-case signed/affine K/V grid plus scaled FP4 and FP8 K/V cases. Report
K3/V2 minus K3/V3 within each checkpoint, then compare that increment across
paired architectures. Keep own-clean damage, absolute loss and cache bytes
separate. The engineering tolerance remains +0.1 nats in every seed.

Floating-format screens use E2M1 FP4 or E4M3FN FP8 with one FP32 absmax scale
per final tensor axis, nearest-even rounding and saturation. This FP4
simulation is not NVIDIA's block-scaled NVFP4 layout. Weights are tested
both on block linears and on all backbone embedding/linear matrices.
INT4/INT8 use the same coverage and scale precision as comparators.
Activation screens affect inputs to all block linear operations; residual
screens affect the state after attention and MLP updates. Each mHC stream
is rounded separately. Block W4A4/W8A8 combine block weights and their
linear inputs. Embedding/output-head activations, routing and normalization
are outside those activation screens. These are numerical simulations,
with no physical packing, serving-memory or kernel-speed claim.

MNIST evaluates twelve corresponding weight/input policies on each retained
teacher, auxiliary-only shared-initialization student and all-output shared
student. Its all-matrix weight scope includes the stem and output head;
the block scope includes only the two linears in each MLP block. It reports
digit accuracy and digit cross-entropy on all 10,000 test images, changes
from each checkpoint's clean result, parameter coverage and ideal weight
bytes including FP32 scales and untouched parameters. There is no calibration
or retraining. Residual distributions use the same first 256 test images;
this MNIST screen does not round residual states between blocks.

At initialization and saved final checkpoints, the same eight-window batch
measures mean, variance, channel variance, RMS, extrema and sampled tails
through depth. Per-stream and collapsed-mean mHC statistics remain separate.
Weight distributions and learned gains are retained. Observation must leave
logits bit-identical. A variance change alone does not establish a useful
quantization benefit; relative rounding error is also recorded, and the
decision uses measured loss and covered bytes. Per-vector absmax scaling
can absorb a uniform amplitude change, so inspect distribution shape and
relative error before assigning a mechanism to larger variance.

Language checkpoints retain model, AdamW and CPU/GPU RNG state at steps
0, 1000 and 2000, plus every pilot endpoint. MNIST retains its existing
teacher/student states and checkpoint manifest. GPU numerical checks and
disposable cost probes precede the full matrix. Raw checkpoints, JSON and
logs belong in the configured external directories, not in this repository.
