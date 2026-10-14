# Subliminal learning on MNIST

This workload measures how much of a teacher's digit-recognition ability
reaches a student that is trained only on the teacher's *auxiliary*
outputs for random noise images, and whether changing the residual
connection mechanism changes that amount. It reproduces the small MNIST
experiment of Cloud, Le, et al.,
[*Subliminal Learning: Language models transmit behavioral traits via
hidden signals in data*](https://arxiv.org/pdf/2507.14805v1) (Section 6.2,
Figure 10), and then repeats the same controls on a residual MLP whose
residual connections are replaced by
[Hyper-Connections](https://arxiv.org/pdf/2409.19606v3) (HC, Zhu et al.,
ICLR 2025) or
[Manifold-Constrained Hyper-Connections](https://arxiv.org/pdf/2512.24880v2)
(mHC, Xie et al.).

MNIST digit transfer is useful knowledge transfer, not a harmful trait.
Less transfer in this experiment does not show that an architecture
prevents unwanted behavioral transfer in language models; it is a small,
controlled measurement of one mechanism.

## Contents

1. [The experiment](#the-experiment)
2. [Running it](#running-it)
3. [Architectures](#architectures)
4. [Loss conventions](#loss-conventions)
5. [Reference oracle](#reference-oracle)
6. [Outputs and statistics](#outputs-and-statistics)
7. [Files](#files)

## The experiment

Each network has 13 outputs: ten digit outputs and three auxiliary
outputs that no task ever supervises. For every independent model we
save its untouched initialization W0 and draw an independent alternate
initialization W0_other. A teacher starts from a copy of W0 and learns
MNIST with cross-entropy on the ten digit outputs only. Four students
are then distilled from that frozen teacher on the same fixed set of
60,000 uniform noise images, in the same minibatch order:

| Student | Starts from | Distillation target |
| --- | --- | --- |
| aux / shared init | copy of W0 | softmax over the teacher's 3 auxiliary outputs |
| all / shared init | copy of W0 | softmax over all 13 teacher outputs |
| aux / other init | copy of W0_other | softmax over the 3 auxiliary outputs |
| all / other init | copy of W0_other | softmax over all 13 outputs |

Every model is scored on the MNIST test split from its digit outputs.
The *transfer gain* is the aux/shared student's accuracy minus the
accuracy of W0 itself. The paper's finding is that this gain is large
when the student shares the teacher's initialization and absent when it
does not, even though the student never sees a digit or a digit output.

The auxiliary targets are a softmax over the three auxiliary logits
alone, not a slice of a 13-way softmax. During auxiliary distillation the
student's ten digit-output rows receive exactly zero gradient and, with
fresh Adam state and no weight decay, stay bit-for-bit equal to W0; the
same holds for the teacher's auxiliary rows during digit training. The
runner checks both and records the result per model.

The recipe is the reference script's: native 28x28 images normalized with
`Normalize((0.5,), (0.5,))` on both splits, linear weights drawn from
N(0, 1/fan_in) with zero biases, five teacher and five student epochs,
Adam at learning rate 3e-4 with default betas and epsilon, batch size
1,024 with the final partial batch kept, FP32, no compilation. The
configuration layer rejects attempts to change precision, compilation or
the optimizer rather than silently accepting them.

The different-initialization controls deliberately differ from the
reference script, which forms them by permuting its cohort with
`randperm`. A permutation can have fixed points, and a fixed point pairs
a teacher with a student that shares its initialization. Here every
control starts from its own independently drawn W0_other, and the
runner asserts that W0 and W0_other differ for every model.

Every random quantity comes from a CPU generator seeded by SHA-256 of the
model key and a stream name (`init`, `init_other`, `noise`,
`teacher_order/eN`, `student_order/eN`). Architectures that share a
parameter shape therefore share exact weights, noise and minibatch order
for a seed, and routing parameters never consume those streams.

## Running it

The workload is a normal knlp Kconfig target. Enabling
`CONFIG_SUBLIMINAL_LEARNING` makes plain `make` and `make all` run the
experiment instead of the LeNet or GPT-2 training dispatch; enabling
another research workload at the same time is reported as a conflict.

```bash
make defconfig-subliminal-mnist-paper-smoke   # two seeds, reduced data
make subliminal-doctor    # read-only: recipe, GPU, data, output root
make subliminal-test      # synthetic CPU tests, no downloads
make subliminal-probe     # timed optimizer steps, projected run time
make                      # the configured run
make subliminal-report    # summarize every run under the output root
```

The presets in [defconfigs/](../../defconfigs) are:

| Preset | Contents |
| --- | --- |
| `subliminal-mnist-paper-smoke` | Plumbing check; cannot issue a verdict |
| `subliminal-mnist-paper` | The paper's 100-model cohort, reference loss reductions |
| `subliminal-mnist-paper-diag` | Paper MLP, five pilot seeds, reference reductions |
| `subliminal-mnist-paper-bridge` | Same five seeds with per-image mean losses |
| `subliminal-mnist-residual`, `-hc`, `-mhc` | Architecture pilots, five seeds |
| `subliminal-mnist-residual-cohort20`, `-hc-cohort20`, `-mhc-cohort20` | Fixed 20-seed comparison |

`make SUBLIMINAL_OUT=/path` changes the result root,
`SUBLIMINAL_SET="field=value ..."` adds overrides that are recorded in
the run's recipe, and `SUBLIMINAL_RESUME=/path/to/run` continues a run
that stopped at its wall-clock budget (`CONFIG_SUBLIMINAL_MAX_SECONDS`,
checked at epoch boundaries). The same runner is available directly:

```bash
python3 -m research.subliminal_learning.run run --config .config
```

The target-scale-zero numerical audit is a read-only sidecar over an archived
run and its frozen teacher checkpoint. It reconstructs W0 and the production
noise stream, verifies their archived digests, and records repeated forwards,
FP32/FP64 saved-logit gradients, parameter-group gradients, and short SGD and
Adam displacement traces:

```bash
python3 -m research.subliminal_learning.null_audit \
  --run-dir /path/to/residual-target-zero \
  --teacher-checkpoint /path/to/pack00_teacher.pt \
  --out /path/to/null-audit.json
```

The companion no-loss baseline reconstructs the same W0 models, records their
digit accuracy and prediction histograms, and runs the same noise forwards
without loading a teacher, constructing a loss, calling backward, or creating
an optimizer. It verifies that state, digit logits, predictions, and accuracy
remain bitwise unchanged when the control is a literal no-op:

```bash
python3 -m research.subliminal_learning.no_loss_baseline \
  --run-dir /path/to/residual-target-zero \
  --out /path/to/no-loss-baseline.json
```

`CONFIG_DEVICE="cuda"` also selects AMD GPUs through ROCm's CUDA
interface. If a GPU is requested and none is visible, the runner exits
instead of falling back to the CPU.

## Architectures

`paper_mlp` is the reference network, 784-256-256-13 with ReLU,
270,093 parameters.

The residual family shares a 784-to-256 stem, four blocks
`F(h) = Linear(ReLU(Linear(LayerNorm(h))))` of width 256, one final
LayerNorm and a single 256-to-13 head; `residual_mlp` computes
`h + F(h)` and has 733,197 parameters. This backbone is our choice, not
part of the subliminal-learning paper. `noskip_mlp` is the same network
with the skip connection removed (`h = F(h)`), the control for what the
residual connection itself does as depth grows, and
`CONFIG_SUBLIMINAL_BLOCK_NORM=n` removes every LayerNorm from either of
those two so normalization cannot mask a depth effect.

`layerrope=y` composes with `residual_mlp`, `hc`, or `hc_static`. It
implements the shared gain and complex depth schedule in Eq. 1 of
[LayerRoPE](https://arxiv.org/pdf/2610.09179). Following that paper's
Vision Transformer adaptation, it changes the existing block input
LayerNorm gains, keeps layer-wise centering and biases, and leaves the
final normalization and HC routing gains in place. It rotates the gain
vector, not the activation vector. Each independent model owns one
shared gain and four learned schedule scalars; sharing is across depth.
The frequency base is 100 and the magnitude slope starts at zero, as in
the paper's vision setting. The rotation slope starts at -0.5, following
the paper's main configuration. Both log intercepts explicitly start at
zero because the paper does not specify their initialization. The gain
starts at ones. These choices are saved in each recipe; the initialized
function differs from ordinary LayerNorm. This is a four-block MNIST
adaptation, not a replication of the paper's full input/output RMSNorm
language-model architecture.

The `subliminal-mnist-layerrope` and `subliminal-mnist-hc-layerrope`
presets enable this option with the repaired `exact_cancel_v1` loss.
For a matched four-arm study with ten fresh seeds, five teacher epochs,
and student measurements at 5, 20, and 80 epochs, the standalone runner
uses the same full dataset, noise, common weights and four student
controls for every arm:

```bash
python3 -m research.subliminal_learning.layerrope_study probe --root /path/to/study
python3 -m research.subliminal_learning.layerrope_study qualify --root /path/to/study
python3 -m research.subliminal_learning.layerrope_study run --root /path/to/study
```

Qualification checks three real loss/backward/Adam steps each for a
stationary shared-init null, a different-init null that must learn, and
an active direct-teacher signal. The run records a frozen plan before
training. The primary endpoint is epoch 80, with 97.5% paired bootstrap
intervals for the two LayerRoPE-versus-HC contrasts. Lower transfer is
screened together with teacher and all-output utility within two points,
and learning speed within 10% of HC's epochs to 95%. The comparison is
an architecture-level sender-and-receiver change; it does not isolate
which side causes a transfer difference. `analyze --root /path/to/study`
regenerates the statistics and learning-curve figures from receipts.

`hc` replaces each residual addition with dynamic Hyper-Connections with
four streams, following the paper's Eqs. 1-14 and Algorithm 2: a
per-stream LayerNorm, zero-initialized dynamic maps passed through tanh
and scaled by learnable factors starting at 0.01, static maps
initialized so block k reads stream k mod 4, writes to every stream with
weight one and carries the streams through the identity. `hc_static`
keeps only the static maps and is a development check. The test suite
compares our packed implementation with the paper's printed Algorithm 2,
forward and backward.

`mhc` follows the mHC paper's Eqs. 3 and 7-9: the four streams are
flattened, RMS-normalized and projected to the pre, post and residual
maps, each with its own gating factor (initialized to 0.01) and static
bias; H_pre is a sigmoid, H_post twice a sigmoid, and H_res the
Sinkhorn-Knopp projection of exp(logits) with 20 column-then-row
normalizations. The paper does not state its bias initialization. We
encode the HC initialization through those maps with
`CONFIG_SUBLIMINAL_MHC_INIT_LOGIT=c`: pre logits +c for stream k mod 4 and
-c elsewhere, residual logits +c on the diagonal and -c elsewhere, post
logits 0. Reading one stream per block is what lets the streams diverge;
with uniform reads and writes every stream receives identical gradients
and the four streams stay copies of one another. The default c = 4
starts H_res within 4e-4 of the identity, and the runner records the row
and column sum errors, smallest entry and mean diagonal of H_res at
initialization and after every stage.

`gr` is the Gated Residual of the Qwen team's
[Qwen3.8-Next architecture report](https://arxiv.org/pdf/2608.30320v1)
(Section 2.2, Eqs. 30-34): each of four branches is RMS-normalized with
its own gain, an elementwise sigmoid read gate per branch and channel is
predicted from all branches through a rank d/8 bottleneck, the gated
branches are averaged into the block input, and the block output is
written to each branch with weight 2*sigmoid of a linear map of all
branches. There is no branch-mixing matrix and no block pre-norm, and
the projections start from random weights drawn from their own
generator stream. At width 256 it has 1,013,773 parameters. `altup` is
the report's width-only baseline (Eqs. 21-22): a learned scalar read over
four branches, starting at 1/4 each, with each block's output added to
branch k mod 4 only.

For HC, mHC, GR and AltUp the stem output is copied into the four streams and
the streams are averaged, not summed, before the final LayerNorm, so HC
at initialization computes exactly the same function as `residual_mlp`
with the same weights. That equality is a unit test.

## Loss conventions

The reference script packs its 100 models into one tensor and its
reductions depend on that packing: teacher cross-entropy is the mean over
models and images, and student `kl_div(..., reduction="batchmean")`
divides the sum over images and outputs by the number of models, not by
the number of images. Two named conventions are supported:

- `upstream_batched_v1` reproduces those reductions for a *logical*
  cohort (`CONFIG_SUBLIMINAL_LOGICAL_COHORT`, default the number of
  models), independent of how many models the runner physically packs
  together.
- `per_example_v1` uses ordinary per-image means, so each model's loss
  and update are independent of the cohort.

The paper presets use the first; the architecture presets use the second
after the paper MLP has been shown to transfer under both. Metrics always
record the per-image mean KL alongside the optimized loss so the two are
never confused. Tests show that the first Adam update of the port equals
the reference script's on identical tensors, and that splitting a cohort
into packs leaves every update unchanged.

Two numerical implementations preserve those reductions. The default
`legacy_kl_v1` retains the historical composed FP32 operation for exact
reproduction. `exact_cancel_v1` retains the same FP32 forward loss but
uses the analytic student-logit derivative
`softmax(student) - softmax(target)`. Identical logits therefore produce
an exactly zero gradient while still executing loss, backward and the
optimizer step. This is a numerical repair, not a `target_scale` special
case: a differently initialized student still learns when `a = 0`.

## Reference oracle

`make subliminal-oracle` runs a byte-for-byte copy of the reference
script, `upstream/run_mnist_experiment.py.orig`, from
[github.com/MinhxLe/subliminal-learning](https://github.com/MinhxLe/subliminal-learning/blob/db04f4150edf940559b5f3147f65d808e9313efd/scripts/run_mnist_experiment.py)
at commit `db04f4150edf940559b5f3147f65d808e9313efd` (MIT license, kept in
`upstream/LICENSE`); the `.orig` suffix keeps code formatters from
rewriting it. The wrapper verifies the file's git blob hash,
executes it unmodified, and saves the per-model accuracies and the
cross-model permutation, including its fixed points. It needs about 19 GB
of accelerator memory for the noise images of its 100 models.

## Interventions and analysis

`post_scale` multiplies the whole write of HC, GR or mHC by a fixed
factor, leaving reads, normalization and stream mixing alone; with HC at
initialization it is exactly a residual network whose update is scaled
the same way, which a test checks. `target_scale` and `target_match_kl`
replace the distillation targets with `z0 + a (zT - z0)`, weakening the
teacher's signal for an unchanged student. At `a = 0`, a student that
shares W0 is an exact-stationarity control, while a differently initialized
student must still learn W0. `fresh_noise` draws new training noise every
student epoch. mHC
records the largest Sinkhorn-Knopp row and column error over every
forward pass of every stage, and `sinkhorn_log` runs the iteration in the
log domain. [analyze.py](analyze.py) compares a candidate with its parent
seed by seed across packs, rejects runs with missing, duplicated or
unclean seeds, and applies the transfer, utility, speed and stability
screens stated in its docstring.

## Running on Modal

[modal_app.py](modal_app.py) runs a list of cells on Modal, one small GPU
container per cell, in parallel. A cell is a defconfig plus recipe
overrides, the same thing `make defconfig-<base> SUBLIMINAL_SET="..."`
runs locally, so any cell can be repeated on another accelerator. The
MNIST tensor cache is uploaded to the `knlp-runs` volume first so the
image needs only PyTorch and NumPy; finished runs and their checkpoints
are copied to that volume, and each run records the source revision
and a hash of the files that were shipped.

## Outputs and statistics

Each run writes a directory under the result root containing the
resolved recipe with the source of every value and the git revision
(`recipe.json`), software and accelerator versions (`environment.json`),
per-epoch losses and test accuracy for every stage (`metrics.jsonl`),
one row per model with all six accuracies, gains, row checks and the
SHA-256 digests of W0, W0_other, the teacher and the noise (`seeds.csv`),
routing diagnostics (`results.json`), a manifest with the path, size and
SHA-256 of every checkpoint, and a completion status with per-stage wall
time (synchronized with the accelerator) and peak accelerator memory. The
checkpoints themselves (initial states, teacher and final students)
and any resume state go to a separate root,
`CONFIG_SUBLIMINAL_CHECKPOINT_DIR` or `make SUBLIMINAL_CKPT=/path`, so
the receipts stay small enough to archive on their own.

`make subliminal-report` reads those receipts. Uncertainty comes from a
paired bootstrap over independent seeds, never over test images. For a
single run it reports the six condition means, the transfer gains and an
assay qualification: mean teacher accuracy at least 95%, all-output
shared-init student at least 80%, and an aux/shared gain exceeding the
aux/other gain by at least 10 percentage points with a positive interval.
`python3 -m research.subliminal_learning.report compare BASELINE
CANDIDATE...` pairs runs by seed and reports the transfer-gain contrast
with Bonferroni-adjusted intervals (97.5% for two candidates), plus
non-inferiority of teacher and all-output distillation accuracy within
two percentage points. Incomplete runs are shown as incomplete and never
enter a comparison.

## Files

- [Kconfig](Kconfig) and [Makefile.subliminal](../../Makefile.subliminal):
  options and targets
- [recipe.py](recipe.py): typed recipe resolution and validation
- [data.py](data.py): MNIST tensors, noise, named random streams
- [models.py](models.py) and [connections.py](connections.py):
  packed networks, HC and mHC
- [run.py](run.py): doctor, probe and the resumable runner
- [report.py](report.py): statistics and figures
- [oracle.py](oracle.py): the unmodified reference run
- [tests/](tests): synthetic CPU tests
