# Bandwidth-Proportional Attention (BPA)

Bandwidth-Proportional Attention (BPA) is the `knlp` line of work that grew
out of an earlier selective-attention idea and eventually sharpened into a more
concrete systems conclusion: **autoregressive decode is dominated by KV-memory
traffic, and decode optimization only matters if it attacks that bottleneck**.

Today BPA should be read less as one fixed architecture and more as the evolved
story connecting:

1. **RGSA / routing-era ideas** about reading less context,
2. **BPA framing** around bandwidth-bound decode and KV access economics, and
3. **fused KV quantization** as the strongest current concrete systems result.

For the historical lineage, see
[docs/paper/bpa/evolution.md](paper/bpa/evolution.md).

## The Current Core Message

During autoregressive decode, the model repeatedly rereads the KV cache for each
new token. That means decode cost grows with the amount of KV state touched per
step, which in practice scales with context length and batch size.

Across AMD W7900, NVIDIA H100, and NVIDIA B200, measurements in `knlp` show the
same qualitative behavior:

- decode throughput tracks available memory bandwidth much more than peak
  compute,
- latency scales approximately linearly with context length,
- throughput saturates with batch according to hardware-specific parameters,
- and long-context inference quickly becomes a memory-system problem.

That finding reframed the project. The question stopped being merely
"can attention be made sparse or selective?" and became:

> if decode is memory-bound, what interventions actually buy us useful speedup or
> useful quality preservation under a bandwidth budget?

## Where BPA Landed So Far

### 1. Decode is the issue

The first durable result is the systems diagnosis itself: decode is bottlenecked
by KV-memory traffic. This is the foundation for the current BPA narrative and
for the existing visualization work.

### 2. Fused KV quantization is the strongest concrete result

INT4 KV quantization helps only when it is **fused into the attention kernel**.
A staged or non-fused path introduces intermediate buffer traffic and can become
slower than the FP16 baseline. The fused path, by contrast, turns KV
compression into a real decode speedup.

This is the most concrete current BPA outcome in `knlp`: compression is useful
when it reduces real memory traffic at the kernel level, not when it only looks
smaller on paper.

### 3. Long context remains a bandwidth/capacity problem

Even after compression, long-context decode remains constrained by the memory
system. B200-style large-HBM devices extend the practical context window, but
that is a capacity consequence layered on top of the same decode bottleneck.

### 4. Selective access and tiering remain open but motivated

Earlier BPA questions about selective KV access, protected layers, and tiered
precision still matter. They now sit on top of a clearer systems story rather
than being treated as isolated architectural ideas.

## How BPA Relates to RGSA

RGSA was an earlier attempt to reduce attention cost through routing and
retrieval. It matters historically because it captured the right instinct:
**reading everything is expensive**.

What changed is that BPA grounded that instinct in measurements of the actual
decode bottleneck. The project moved from "which chunks should we route to?" to
"what memory traffic do we actually have to pay for at decode time, and which
interventions reduce it in practice?"

RGSA remains documented in `docs/rgsa.md` as precursor work rather than the
current public entrypoint for this story.

## Current BPA Tracks in knlp

### Decode scaling and bandwidth characterization
- cross-GPU decode throughput / latency sweeps
- batch saturation fits
- context-linearity measurements
- long-context capacity pushes

### Fused KV quantization
- INT4 KV cache quantization with kernel fusion
- comparison of fused vs non-fused pipelines
- cross-GPU validation of the memory-traffic story

### Sensitivity / protection / tiering
- selective high-precision protection of sensitive layers
- mixed-precision KV schemes
- bounded-protection (`k*`) experiments

## Visualization and Data

The current interactive visualizations are:

- [AR Decode Bottleneck](https://mcgrof.github.io/knlp/ar_decode_bottleneck.html) — structural explanation of why autoregressive decode rereads KV state every step
- [KV Bandwidth Visualization](https://mcgrof.github.io/knlp/kv_bandwidth_visualization.html) — empirical cross-GPU decode scaling and bandwidth view

Together they serve as the current generic public explanation of the BPA
systems diagnosis: **decode is the issue**. They are not yet the final
paper-shaped storytelling artifacts.

## Status

BPA in `knlp` is an evolving narrative plus a set of concrete systems results.
The cleanest stable public message today is:

- decode is memory-bound,
- fused quantization is a real win because it attacks memory traffic directly,
- and future BPA-style work should be judged by whether it changes the real
  decode bottleneck rather than only changing abstract attention structure.
