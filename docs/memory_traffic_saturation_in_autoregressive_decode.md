# Memory-Traffic Saturation in Autoregressive Decode

Use this document as the standalone systems characterization of decode-time
memory traffic.

Start from the concrete question:
**what memory traffic does decode actually pay for, and which interventions
change that bill in practice?**

You can use the measurements here to establish the cross-GPU decode regime,
motivate KV quantization review, and explain why fused KV quantization became
the strongest concrete intervention.

For BPA background, see the [BPA overview](https://github.com/mcgrof/knlp/blob/main/docs/bpa.md). For the research lineage, see [RGSA, BPA, and fused KV quantization](https://github.com/mcgrof/knlp/blob/main/docs/paper/bpa/evolution.md). For the empirical visualization, use [AR Decode Bottleneck](https://mcgrof.github.io/knlp/ar_decode_bottleneck.html) and [Decode Scaling Visualization](https://mcgrof.github.io/knlp/kv_bandwidth_visualization.html).

## What this is

This document records the decode-time memory-traffic characterization that set
the direction for the later quantization work. The core result is simple:
autoregressive decode is governed by memory traffic, throughput follows
hardware memory-system strength more than peak compute, latency grows roughly
linearly with context length, batch growth produces hardware-specific
saturation, and long-context planning becomes a capacity problem only after the
decode traffic problem has been understood. Treat this as the systems
diagnosis that motivated later fused quantization work.

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

Use this dataset to answer a practical deployment question:

> can this system actually serve dense autoregressive decode at the batch,
> context, and latency target you care about?

Do not plan from model size, HBM capacity, or spec-sheet bandwidth alone.
Plan from measured decode behavior.

### Why this is useful

This dataset records what the system actually does during decode:
- measured latency,
- measured tokens/sec,
- measured effective bandwidth,
- measured batch saturation,
- and measured context scaling.

That lets you plan from observed behavior instead of relying on brochure
numbers or rough back-of-the-envelope KV cache math.

### Why you would do this

Use these measurements when you need to decide things like:
- whether one GPU is sufficient for a target serving regime,
- whether buying more HBM capacity will actually help,
- whether a lower memory/storage tier could sustain dense decode,
- whether fused KV compression is worth the engineering effort,
- and whether a long-context product target is bandwidth-limited,
  capacity-limited, or both.

### Separate bandwidth planning from capacity planning

Use the core matched lanes to answer the bandwidth question:
- what sustained decode-time bandwidth does this workload actually need,
- where does throughput saturate under batch growth,
- and how does latency scale with context length on real hardware?

Use the long-context lane to answer the capacity question:
- when does larger HBM materially extend feasible context length,
- and when does extra capacity matter more than another increment of bandwidth?

Keep those two planning problems separate. A system can have enough capacity and
still fail to sustain decode-time traffic. A system can also have strong
bandwidth and still run out of HBM at long context.

### Use measured bandwidth before proposing tiering

Do not propose KV offload, host-memory cache tiers, CXL-backed expansion, or
other storage hierarchies from capacity numbers alone.

Start with the measured `bw_GBs` rows and ask:
- what bandwidth does dense decode actually require at the target `B` and `T`?
- does the system already saturate before that point?
- could any lower tier feed that decode path fast enough?

If the lower tier cannot sustain the measured decode-time demand, then the
design increases capacity without solving the actual bottleneck.

### Practical examples

#### Example 1: choose between A100 and B200

Suppose you want to serve a 7B model at moderate batch and context lengths in
roughly the 8K--32K regime.

Use the matched-lane measurements to check:
- where A100 saturates,
- what throughput B200 adds in practice,
- and whether the expected gain comes from bandwidth, capacity, or both.

Do not assume that the ratio of peak HBM bandwidth will transfer directly into
real decode throughput.

#### Example 2: evaluate KV offload to a slower tier

Suppose someone proposes moving older KV state to host memory or another slower
storage layer.

Use the measured decode rows to ask:
- what sustained bandwidth is required at the target batch and context,
- and can the proposed tier deliver that continuously during dense decode?

If not, then the idea adds bytes of storage without making the decode path
viable.

#### Example 3: plan a 128K+ context product

Suppose the target is premium long-context inference at 128K or beyond.

Answer two questions separately:
1. can the decode path sustain the traffic in the target operating regime?
2. can the device hold enough KV state at all?

Use the core lanes for the first question and the long-context lane for the
second.

#### Example 4: decide whether fused KV compression is worth building

Use the dataset to determine whether the current serving regime is already in a
memory-traffic-limited decode phase. If the target operating region is short
context and weakly KV-bound, compression may not be the highest-leverage change.
If the target region is long context, larger batch, or a clear KV-bandwidth
plateau, then fused KV compression becomes much more attractive.

### Working rule

Use the measured decode-time bandwidth rows first. Then decide:
- whether the workload is bandwidth-limited,
- whether it is capacity-limited,
- and whether any lower memory tier can actually feed it.

Do not design tiering from capacity alone.
