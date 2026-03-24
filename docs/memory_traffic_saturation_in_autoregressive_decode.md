# Memory-Traffic Saturation in Autoregressive Decode

Use this document as the standalone systems characterization of decode-time
memory traffic.

Start from the concrete question:
**what memory traffic does decode actually pay for, and which interventions
change that bill in practice?**

You can use the measurements here to establish the cross-GPU decode regime,
motivate KV quantization review, and explain why fused KV quantization became
the strongest concrete intervention.

This document records the decode-time memory-traffic characterization that set
the direction for the later quantization work. The core result is simple:
autoregressive decode is governed by memory traffic, throughput follows
hardware memory-system strength more than peak compute, latency grows roughly
linearly with context length, batch growth produces hardware-specific
saturation, and long-context planning becomes a capacity problem only after the
decode traffic problem has been understood. Treat this as the systems
diagnosis that motivated later fused quantization work.

For BPA background, see the [BPA overview](https://github.com/mcgrof/knlp/blob/main/docs/bpa.md). For the research lineage, see [RGSA, BPA, and fused KV quantization](https://github.com/mcgrof/knlp/blob/main/docs/paper/bpa/evolution.md). For the empirical visualization, use [AR Decode Bottleneck](https://mcgrof.github.io/knlp/ar_decode_bottleneck.html) and [Decode Scaling Visualization](https://mcgrof.github.io/knlp/kv_bandwidth_visualization.html).

## What it measures

The measurement corpus covers decode-time KV bandwidth behavior across multiple
GPUs, batch sizes, context lengths, and kernel paths. The current paper-facing
tracks cover AMD W7900, NVIDIA A100, NVIDIA H100, and NVIDIA B200, with the
B200 also used for the long-context capacity path. Each measurement point is
intended to record the batch size `B`, context length `T`, mean latency,
standard deviation, tokens/sec, KV bytes touched, and effective bandwidth
(`bw_GBs`).

## Why this matters

Start with the concrete systems question:

> is decode really bottlenecked by memory traffic, or is that just a loose story
> people tell before they profile the kernel path carefully?

Use the measurements here to answer it directly: decode is consistently
memory-traffic limited across tested hardware classes. Then ask the next
question:

> if decode is dominated by KV traffic, which intervention reduces real kernel
> traffic instead of only shrinking tensors on paper?

That is what led to fused KV quantization.

## What it shows

Use the dataset to show five things.

### 1. Decode is a memory-traffic problem

Across the tested GPUs, decode throughput follows sustained decode-time
bandwidth far more closely than headline compute throughput.

### 2. Context growth is close to linear

At fixed batch, latency grows approximately linearly with context length in the
core decode regime.

### 3. Batch growth saturates by hardware class

At fixed context, throughput rises with batch and then saturates or flattens,
with the onset and plateau depending on the GPU and kernel path.

### 4. Cross-GPU behavior is qualitatively stable

The exact fit parameters are not universal, but the decode regime is. W7900,
A100, H100, and B200 all show the same qualitative memory-traffic-limited
behavior.

### 5. Capacity planning starts after bandwidth planning

The B200 long-context lane shows that large HBM capacity extends feasible
context length, but only after the decode traffic problem has been understood.
This is why future tiering strategies must satisfy decode-time bandwidth demand,
not just provide more bytes of storage.

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

## Original script lineage

This standalone result was derived from a broader BPA script lineage.
Use the unified helper above for reproduction now, but keep these original
public scripts as provenance for how the result emerged:

### H100 decode characterization lineage
- `scripts/v35_h100_bench.py`
- `scripts/bpa_h100_exp3_kernel_latency.py`
- `scripts/bpa_h100_exp2_long_context.py`

### W7900 lineage
- `scripts/bpa_v50_w7900.py`
- `scripts/bpa_v51_w7900.py`
- `scripts/bpa_v52_w7900.py`
- `scripts/bpa_v53_w7900.py`

### Supporting helpers
- `scripts/check_a100_baseline.py`
- `scripts/plot_scaling_laws_b200.py`
- `scripts/generate_unified_comparison.py`
- `scripts/regenerate_unified_summary.py`

Use the original lineage scripts when you need provenance or want to inspect
how the result evolved. Use the unified helper when you want to reproduce the
public dataset cleanly.

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

You can use this dataset to answer a practical deployment question: can a
system actually serve dense autoregressive decode at the batch, context, and
latency target you care about? That is the reason to keep this characterization
around as more than a paper artifact. It gives you measured decode behavior,
not just a hardware spec sheet and not just an estimate of KV cache size on
paper.

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
decode-time bandwidth rather than to architecture diagrams.

A few practical examples make the distinction clearer. If you are choosing
between A100 and B200 for a 7B-class dense decode service in the 8K--32K range,
you can use the matched-lane measurements to see where A100 saturates, what
throughput B200 actually adds, and whether the gain comes from bandwidth,
capacity, or both. Do not assume that the ratio of peak HBM bandwidth will map
cleanly onto real decode throughput. If someone proposes moving older KV state
to host memory or another slower tier, you can use the measured rows to ask
what sustained bandwidth the target batch/context regime really requires and
whether that lower tier can actually keep up. If the answer is no, then the
proposal adds bytes without making dense decode viable. If the target is a
128K+ product, answer two separate questions: can the decode path sustain the
traffic, and can the device hold enough KV at all? Use the core lanes for the
first and the long-context lane for the second. If the question is whether
fused KV compression is worth building, the same logic applies: if the target
regime is already weakly KV-bound, compression may not be the highest-leverage
change; if the regime sits on a clear KV-bandwidth plateau, then compression
becomes much more attractive.

The working rule is simple. Start with the measured decode-time bandwidth rows.
Then decide whether the workload is bandwidth-limited, capacity-limited, or
both. Only after that should you decide whether a lower memory tier can feed
the workload or whether compression and routing are required. Do not design
future tiering from capacity alone.
