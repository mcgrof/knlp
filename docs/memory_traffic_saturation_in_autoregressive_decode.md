# Memory-Traffic Saturation in Autoregressive Decode

This document records the decode-time memory-traffic characterization that set
the direction for the later quantization work. It shows that autoregressive
decode is governed by memory traffic, that throughput follows hardware
memory-system strength more than peak compute, that latency grows roughly
linearly with context length, and that batch growth produces
hardware-specific saturation. It is useful for two reasons. First, it explains
why fused KV quantization became a strong intervention: it attacks the traffic
that actually limits decode. Second, it gives you measured data for practical
planning questions such as whether a target GPU can sustain dense decode at a
given batch and context, whether extra HBM capacity will materially help, and
whether a lower memory tier could ever feed the workload fast enough to matter.
Detailed planning examples appear in [Use this for memory planning and tiering](#use-this-for-memory-planning-and-tiering).

For BPA background, see the [BPA overview](https://github.com/mcgrof/knlp/blob/main/docs/bpa.md). For script provenance and the reason this result now has its own standalone entrypoint, see [Memory-Traffic Saturation in Autoregressive Decode: Lineage and Script Provenance](https://github.com/mcgrof/knlp/blob/main/docs/memory_traffic_saturation_lineage.md). For the empirical visualization, use [AR Decode Bottleneck](https://mcgrof.github.io/knlp/ar_decode_bottleneck.html), [Decode Scaling Visualization](https://mcgrof.github.io/knlp/kv_bandwidth_visualization.html), and [Ridge Point Visualization](https://knlp.io/ridge_point.html).

## Table of Contents

- [What it measures](#what-it-measures)
- [Why this matters](#why-this-matters)
- [What it shows](#what-it-shows)
- [Reproduce this result](#reproduce-this-result)
- [Scripts](#scripts)
- [Relationship to fused quantization](#relationship-to-fused-quantization)
- [Use this for memory planning and tiering](#use-this-for-memory-planning-and-tiering)
- [Practical use cases](#practical-use-cases)

## What it measures

The measurement corpus covers decode-time KV bandwidth behavior across multiple
GPUs, batch sizes, context lengths, and kernel paths. The current paper-facing
tracks cover AMD W7900, NVIDIA A100, NVIDIA H100, and NVIDIA B200, with the
B200 also used for the long-context capacity path. Each measurement point is
intended to record the batch size `B`, context length `T`, mean latency,
standard deviation, tokens/sec, KV bytes touched, and effective bandwidth
(`bw_GBs`).

## Why this matters

The purpose of this characterization is to replace intuition with measurement.
A lot of discussion around decode bottlenecks starts from a loose story that
memory traffic must matter, but that story is not enough by itself. The
measurements here make the claim concrete: across the tested hardware classes,
decode is consistently memory-traffic limited. Once that is established, the
next question becomes much sharper. If decode is dominated by KV traffic, then
which intervention reduces real kernel traffic instead of merely shrinking
representations on paper? That is the line of reasoning that led to fused KV
quantization.

## What it shows

The measurements support five concrete conclusions.

### 1. Decode is a memory-traffic problem

The first result is the core systems diagnosis: decode follows sustained
memory movement much more closely than advertised compute capability. In the
matched lanes, throughput ordering tracks the practical strength of the memory
system rather than any simple FLOP ranking. The W7900 sits at the low end of
that curve, A100 occupies the intermediate regime, and H100/B200 define the
higher-bandwidth end. That is exactly the pattern you would expect if decode is
paying primarily for repeatedly rereading KV state instead of consuming the GPU
as a pure compute engine.

### 2. Context growth stays close to linear in the core regime

The second result is that latency grows roughly linearly with context length at
fixed batch across the tested GPUs. This matters because it ties the slowdown
directly to the amount of KV state reread per decode step. The matched-lane
runs do not show some exotic nonlinear transition in the normal operating
range; they show the cleaner and more operationally useful story that longer
context means proportionally more traffic and therefore proportionally slower
decode.

### 3. Batch growth saturates, but the saturation point depends on hardware

The third result is that throughput does not scale forever with batch. It rises
and then saturates or flattens, and the onset of that flattening depends on the
GPU and kernel path. The A100 lane is useful here because it fills the gap
between the low-bandwidth W7900 and the stronger H100/B200 lanes. That makes it
clear that the saturation behavior is not just a two-endpoint curiosity. The
shape is stable, but the operating point shifts with the hardware.

### 4. The qualitative decode regime survives across GPUs

The fourth result is cross-GPU stability in the qualitative regime. The exact
fit parameters are not universal, and the point of this document is not to
pretend they are. What survives across W7900, A100, H100, and B200 is the more
important structural result: decode remains memory-traffic limited, context
scaling remains close to linear in the core regime, and batch growth runs into
hardware-specific saturation. That is a stronger and more useful conclusion
than any claim that one exact coefficient set governs every accelerator.

### 5. Capacity planning starts after bandwidth planning

The fifth result comes from separating the core matched lanes from the B200
long-context path. The B200 long-context measurements show that large HBM
capacity materially extends feasible context length, but they also make clear
that capacity becomes useful only after the decode traffic problem has been
understood. In other words, extra memory capacity does not rescue a decode path
that cannot sustain the needed bandwidth. This is the result that makes the
dataset directly useful for future tiering and memory planning work.

## Reproduce this result

Use the standalone helper:
- `scripts/memory_traffic_saturation.py`

That helper wraps the paper-facing BPA dataset framework so this result can be
reproduced without depending on a private results tree.

### Quick start

Pick a results root you control:

```bash
export RESULTS_ROOT=$PWD/results/memory-traffic-saturation
```

Run one lane at a time:

```bash
python scripts/memory_traffic_saturation.py \
  --results-root "$RESULTS_ROOT" \
  --gpu a100 \
  --stage smoke

python scripts/memory_traffic_saturation.py \
  --results-root "$RESULTS_ROOT" \
  --gpu a100 \
  --stage matrix-plan

python scripts/memory_traffic_saturation.py \
  --results-root "$RESULTS_ROOT" \
  --gpu a100 \
  --stage matrix-exec
```

Run the full configured workflow as a dry run:

```bash
python scripts/memory_traffic_saturation.py \
  --results-root "$RESULTS_ROOT" \
  --gpu all \
  --stage full-dry-run
```

Here `--gpu all` means: run all configured experiment tracks in sequence. It
does not detect local hardware or provision GPUs automatically. Use a specific
value such as `--gpu a100` or `--gpu h100` when you are running on a single
machine or targeting one known hardware lane.

Derive fit artifacts and package a paper-facing export tree:

```bash
python scripts/memory_traffic_saturation.py \
  --results-root "$RESULTS_ROOT" \
  --gpu all \
  --stage fit

python scripts/memory_traffic_saturation.py \
  --results-root "$RESULTS_ROOT" \
  --gpu all \
  --stage package
```

## Scripts

Use these scripts directly.

### Standalone helper
- `scripts/memory_traffic_saturation.py`

### Dataset framework
- `scripts/paper/bpa_paper/run_dataset.py`
- `scripts/paper/bpa_paper/run_smoke.py`
- `scripts/paper/bpa_paper/run_matrix.py`
- `scripts/paper/bpa_paper/fit_scaling.py`
- `scripts/paper/bpa_paper/package_results.py`

### Config files

A **lane** is a scoped experiment track: one hardware target, one measurement
purpose, one defined batch/context grid, and one artifact contract.

These config files define the current lanes:
- `scripts/paper/bpa_paper/configs/w7900.yaml`
- `scripts/paper/bpa_paper/configs/a100.yaml`
- `scripts/paper/bpa_paper/configs/h100.yaml`
- `scripts/paper/bpa_paper/configs/b200.yaml`

## Relationship to fused quantization

Do not conflate this characterization with fused KV quantization itself.

The logic is:
1. BPA forced the memory-traffic question,
2. this decode characterization answered it empirically,
3. fused KV quantization emerged as the strongest concrete response because it
   reduces real decode-time memory traffic in the kernel.

For the intervention and kernel-path story, see:
- [docs/fused_kv_quantization.md](https://github.com/mcgrof/knlp/blob/main/docs/fused_kv_quantization.md)

## Use this for memory planning and tiering

We did this work to understand what dense autoregressive decode actually pays
for at runtime and to ground later quantization work in measured behavior
instead of intuition. You can now also use the same dataset to answer a
practical deployment question: can a system actually serve dense autoregressive
decode at the batch, context, and latency target you care about? It gives you
measured decode behavior, not just a hardware spec sheet and not just an
estimate of KV cache size on paper.

That matters because memory planning for LLM serving is easy to do badly. It is
common to look only at model size, HBM capacity, or advertised bandwidth and
then assume the rest will work itself out. The measurements here let you plan
from what the system really does during decode: how latency changes with
context, where throughput stops improving with batch growth, and what effective
bandwidth the decode path actually sustains. That is much more useful than
back-of-the-envelope capacity math when the real question is whether the system
can keep rereading KV state fast enough to serve the workload.

The dataset is especially useful when you need to make a concrete decision.
Suppose you are deciding whether one GPU is sufficient for a target serving
regime, whether buying more HBM capacity will help, whether a lower tier of
memory could participate in dense decode, or whether fused KV compression is
worth the engineering effort. Those are not the same question, and the dataset
helps separate them. The core matched lanes tell you about sustained decode
bandwidth, batch saturation, and context scaling. The long-context lane tells
you when larger HBM capacity materially changes the feasible operating region.
A system can have enough capacity and still fail because it cannot sustain the
decode-time traffic. A system can also have strong bandwidth and still run out
of HBM once the context target gets large enough. Treat bandwidth planning and
capacity planning as related but separate problems.

This is also where future tiering discussions become real instead of hand-wavy.
Do not start by asking whether a lower tier has enough bytes. Start by asking
what dense decode actually demands at the target batch and context, then ask
whether the slower tier can feed that demand continuously. If the measured
`bw_GBs` rows show that the decode path needs far more sustained bandwidth than
a host-memory tier, CXL-backed tier, or storage-backed cache can deliver, then
the proposal increases capacity without solving the bottleneck. The point is not
to reject tiering out of hand. The point is to make tiering answer to measured
decode-time bandwidth rather than to architecture diagrams. The same reasoning
also benefits from the [Ridge Point Visualization](https://knlp.io/ridge_point.html),
which helps place the measured decode regime against the compute/memory balance
of the hardware.

The working rule is simple. Start with the measured decode-time bandwidth rows.
Then decide whether the workload is bandwidth-limited, capacity-limited, or
both. Only after that should you decide whether a lower memory tier can feed
the workload or whether compression and routing are required. Do not design
future tiering from capacity alone.

If asynchronous fused dequantization becomes common in mainstream inference
stacks, this characterization does not become obsolete, but the measurement
framework should be rerun with that kernel path made explicit. The likely
change is not that decode suddenly stops being a memory-traffic problem. The
more likely outcome is that the measured bandwidth plateau, saturation onset,
and implementation-dependent deltas move because overlap and staging improve.
That future comparison should be done as a matched with-versus-without study on
identical hardware, models, batch sizes, and context lengths so the effect of
asynchronous fused dequantization is isolated directly.

We have not prioritized that comparison yet because the more immediate source of
variation in the current work is model-specific calibration policy. The results
already show that a calibration or ratio-classifier step is needed for some
models and that the appropriate mapping varies by model family and regime. The
active R&D focus is therefore on reducing the time it takes to discover those
mappings and apply them when they matter. Once that policy-selection path is
cheaper and more stable, it will make more sense to add a dedicated async
fused-dequantization delta study on top of the same measurement framework.

## Practical use cases

One common use case is hardware selection for a real serving target. Suppose
you are choosing between A100 and B200 for a 7B-class dense decode service in
the 8K--32K range. The useful question is not which GPU advertises the larger
HBM number. The useful question is which device sustains the decode regime you
actually need. The matched-lane measurements let you see where A100 saturates,
what throughput B200 adds in practice, and whether the difference comes from
bandwidth, capacity, or both. That is a much better basis for planning than
assuming the ratio of peak HBM bandwidth will map cleanly onto real decode
throughput.

Another use case is evaluating KV offload to a slower tier. If someone proposes
moving older KV state to host memory, a CXL-backed tier, or another slower
storage layer, the right first question is whether that lower tier can sustain
the bandwidth that dense decode actually demands. The measured rows let you ask
what sustained bandwidth the target batch/context regime really requires and
whether the proposed tier can keep up. If the answer is no, then the proposal
adds bytes of storage without making dense decode viable. That is exactly the
kind of planning mistake this dataset is meant to prevent.

A third use case is long-context product planning. If the target is 128K or
beyond, two questions have to be separated cleanly. First, can the decode path
sustain the traffic in the target operating regime? Second, can the device hold
enough KV state at all? The core matched lanes answer the first question. The
long-context lane answers the second. Treating those as one blended question is
how teams end up buying hardware with enough memory but not enough practical
bandwidth, or enough bandwidth but not enough capacity.

A fourth use case is deciding whether fused KV compression is worth building.
This dataset tells you whether the target regime is already sitting in a
memory-traffic-limited decode phase. If the regime is short context and weakly
KV-bound, compression may not be the highest-leverage change. If the regime
sits on a clear KV-bandwidth plateau, then fused KV compression becomes much
more attractive because it attacks the resource that is actually limiting
decode.
