# Memory-Traffic Saturation in Autoregressive Decode

This document stands on its own.

It came out of the original BPA effort, but it is no longer just a BPA note.
BPA led to this characterization because BPA asked a sharper systems question:
**what memory traffic does decode actually pay for, and which interventions
change that bill in practice?**

That question led first to the cross-GPU decode characterization here, then to
KV quantization review, and then to fused KV quantization as the strongest
concrete intervention.

Use this document when you want the standalone systems result:
- what was measured,
- why it was measured,
- what it shows,
- and how to reproduce it with public scripts in `knlp`.

For BPA lineage and broader motivation, see:
- [docs/bpa.md](https://github.com/mcgrof/knlp/blob/main/docs/bpa.md)
- [RGSA \u2192 BPA \u2192 fused KV quantization](https://github.com/mcgrof/knlp/blob/main/docs/paper/bpa/evolution.md)

For the empirical visualization, see:
- [AR Decode Bottleneck](https://mcgrof.github.io/knlp/ar_decode_bottleneck.html)
- [Decode Scaling Visualization](https://mcgrof.github.io/knlp/kv_bandwidth_visualization.html)

## What this is

This is the standalone empirical decode characterization that established:
- autoregressive decode is governed by memory traffic,
- throughput follows hardware memory-system strength more than peak compute,
- latency grows approximately linearly with context length,
- batch growth drives hardware-specific saturation,
- and long-context planning becomes a capacity problem only after decode
  traffic has been reduced enough.

Treat this as the public systems diagnosis that motivated later fused
quantization work.

## What it measures

The measurement corpus covers decode-time KV bandwidth behavior across:
- multiple GPUs,
- multiple batch sizes,
- multiple context lengths,
- and multiple kernel paths.

The current paper-facing lanes are:
- AMD W7900 matched lane,
- NVIDIA A100 matched lane,
- NVIDIA H100 reference lane,
- NVIDIA B200 core provenance lane,
- NVIDIA B200 long-context lane.

Each point is intended to record values such as:
- batch size `B`,
- context length `T`,
- mean latency,
- standard deviation,
- tokens/sec,
- KV bytes touched,
- effective bandwidth (`bw_GBs`).

## Why this matters

This characterization answered the question that BPA forced into the open:

> is decode really bottlenecked by memory traffic, or is that just a loose story
> people tell before they profile the kernel path carefully?

The answer from the measurements is that decode is consistently memory-traffic
limited across tested hardware classes. That result then motivated the next
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

Use the standalone public helper:
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

Run the public workflow for all lanes as a dry run:

```bash
python scripts/memory_traffic_saturation.py \
  --results-root "$RESULTS_ROOT" \
  --gpu all \
  --stage full-dry-run
```

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

## Public scripts involved

Use these public scripts directly.

### Unified standalone helper
- `scripts/memory_traffic_saturation.py`

### Underlying public dataset framework
- `scripts/paper/bpa_paper/run_dataset.py`
- `scripts/paper/bpa_paper/run_smoke.py`
- `scripts/paper/bpa_paper/run_matrix.py`
- `scripts/paper/bpa_paper/fit_scaling.py`
- `scripts/paper/bpa_paper/package_results.py`

### Lane configs
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

Use the measured decode-time bandwidth rows to reason about:
- sustained decode bandwidth,
- batch saturation,
- context-linearity,
- practical long-context limits,
- and whether any lower memory/storage tier could sustain dense decode.

Do not plan future tiering from capacity alone.
Plan from measured decode-time bandwidth first, then ask whether any lower tier
can actually feed that workload.
