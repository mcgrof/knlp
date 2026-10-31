# Tied magnitude intercepts for LayerRoPE

This study asks whether LayerRoPE needs its log-magnitude intercepts tuned
together with its depth slope. In the earlier
[full-gain study](../README.md), every intercept started at zero, and
mHC + LayerRoPE with beta = 0 had the lowest loss at equal tokens. The
LayerRoPE author suggested tuning the input and output intercepts with
opposite signs, and named a = -0.25 with beta = -0.2 as a good point. This
study includes that point in its confirmation.

LayerRoPE scales each site's gain by `exp(alpha + beta * log(layer + 1))`.
This study ties the intercepts with one number `a`: the attention and MLP
input sites start at `alpha = -a`, and the two output sites start at
`alpha = a`. A negative `a` therefore enlarges the block inputs and shrinks
the block outputs before the residual or mHC write. The rotation
intercepts and the frequency base (100) do not change. Beta sets the
magnitude and rotation slopes at all four sites, as in the earlier study.
All of these schedule parameters remain trainable.

## Design

The model is the earlier story model: 32 layers, width 128, two heads, MLP
width 352, a 256-token context and 2,048 predicted tokens per update. Each
run trains for 2,000 updates with AdamW and a 100-update warmup. Learning
rates are fixed at the earlier study's selections: 0.003 for residual and
mHC, and 0.01 for LayerRoPE and mHC + LayerRoPE. There is no new
learning-rate pilot. A shift in `a` can move the best rate, and this is a
known limit of the design.

The screen trains one seed (77911) for each grid cell. The grid is
a ∈ {0, -0.1, -0.25, -0.5} × beta ∈ {0, -0.1, -0.2, -0.3, -0.5}, for both
plain LayerRoPE and mHC + LayerRoPE, plus residual and mHC controls: 42
runs. The primary metric is mean curve cross-entropy (CE) at updates 1000,
1100, ..., 2000, the same metric as before. The screen also records final
clean CE on the separate final array. It skips the quantization grid.

For each arm, the lowest screen CE selects one cell. The confirmation then
trains three fresh seeds (77912–77914) for residual, mHC, each arm's
a = 0/beta = 0 anchor, the author's point and the screen winner. These runs
receive the full-gain study's measurements: residual distributions, 51 KV
cache cases and the floating-point and integer interventions.

The decision rule was fixed before training. Tuned intercepts help an arm
only if its confirmed winner beats that arm's a = 0/beta = 0 anchor on
primary CE in all three paired seeds, with a mean gain of at least 0.01
nats. Otherwise the result is no confirmed gain at this scale. This is an
equal-token comparison; it makes no equal-time claim.

## Run

Prepare a run directory outside the checkout from an existing
`tinystories-prepared/` directory (see the
[reproduction guide](../../reproduction/README.md)). Preparation verifies
each input against `DATA.json`, writes seed orders and freezes the plan and
source hashes:

```sh
python -m research.layerrope_beta.intercept.run prepare \
  --root /path/to/run --prepared /path/to/tinystories-prepared \
  --checkpoints /path/to/checkpoints
python -m research.layerrope_beta.intercept.run run \
  --root /path/to/run --checkpoints /path/to/checkpoints \
  --stage all
```

`--workers` sets how many training processes share the one visible GPU.
The default is 1. These models are small and kernel-launch bound, so
several processes on one GPU share its launch queue: in our measurement six
processes each ran about six times slower than one alone, with no gain in
total throughput. Completed runs are kept on restart, and partial runs
restart from the beginning. `--stage screen` stops after selection;
`--stage confirm` reads the frozen `SELECTED.json`. The `probe` action
times disposable updates.

A single GPU needs many hours for the 66 runs. The
[Modal app](modal_app.py) instead gives each run its own container, so a
stage takes about as long as its slowest run. Each run is slower there
than on a fast local GPU, because the containers' host CPUs limit kernel
launches, so the gain is in wall time only. Upload the prepared run
directory to the `knlp-runs` volume first; the app's docstring gives the
commands.
