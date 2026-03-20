# Bandwidth-Proportional Attention (BPA)

Bandwidth-Proportional Attention (BPA) is the `knlp` line of work for scaling
transformer decode under explicit memory-bandwidth constraints.

Start from the hard systems fact:

**autoregressive decode is dominated by KV-memory traffic.**

That fact organizes the current BPA story:

1. **RGSA** explored routing and selective access as an early attempt to avoid
   paying full attention cost everywhere.
2. **BPA** reframed the problem around bandwidth-bound decode and KV access
   economics.
3. **Fused KV quantization** became the strongest current concrete result,
   because it reduces real decode traffic inside the kernel.

For the lineage, see
[RGSA → BPA → fused KV quantization](https://github.com/mcgrof/knlp/blob/main/docs/paper/bpa/evolution.md).

## Decode Bottleneck

During autoregressive decode, the model rereads KV state for every new token.
Decode cost therefore grows with the amount of KV state touched per step, which
in practice scales with context length and batch size.

Measurements in `knlp` across AMD W7900, NVIDIA H100, and NVIDIA B200 show the
same qualitative regime:

- decode throughput tracks available memory bandwidth more than peak compute,
- latency scales approximately linearly with context length,
- throughput saturates with batch according to hardware-specific parameters,
- long-context inference becomes a memory-system problem before it becomes a
  compute problem.

Use BPA to judge decode interventions by one standard:

> do they reduce the real decode bottleneck, or do they only change the model on paper?

## Generic Attention Scaling Question

Keep the broader question in view:

**how do we scale attention toward billion-token contexts without paying dense,
full-history cost everywhere?**

Compression helps, but compression alone does not solve that problem. As context
length grows, even compressed KV state can still overwhelm the memory system at
decode time.

Use BPA as a research outlet for bandwidth-aware attention scaling, not only as
a decode diagnosis. The open questions are concrete:

- spend bandwidth selectively instead of uniformly,
- vary access granularity by layer or region,
- use FIM-guided sensitivity to decide where bandwidth matters most,
- vary block size, routing policy, or protection level as context approaches
  billion-token regimes.

## Current BPA Results

### Decode is the bottleneck

Treat the systems diagnosis as the foundation. Decode is bottlenecked by
KV-memory traffic. Build from that.

### Fused KV quantization is the strongest current result

Quantization helps only when it is **fused into the attention kernel**.

A staged or non-fused path adds intermediate buffer traffic and can erase the
benefit of using a smaller KV format. A fused path turns compression into a real
decode speedup.

That is the strongest current BPA result in `knlp`: reduce real kernel-level
memory traffic, not just abstract tensor size.

### Long context remains a bandwidth and capacity problem

Compression does not remove the long-context problem. It shifts the operating
point. Large-HBM devices extend practical context, but that extension still sits
on top of the same decode bottleneck.

### Selective access and tiering remain open

Do not collapse BPA into fused quantization alone. Selective KV access,
protected layers, mixed-precision tiering, and more general attention scaling
remain open parts of the program.

## Relation to RGSA

RGSA captured the right early instinct: **reading everything is expensive**.

BPA made that instinct measurable. The project moved from asking which chunks to
route to, toward asking which memory traffic decode must actually pay for and
which interventions reduce that bill in practice.

Keep `docs/rgsa.md` as precursor work. Use BPA as the main overview for this line of work.

## Current BPA Tracks in knlp

### Decode scaling and bandwidth characterization
- cross-GPU decode throughput / latency sweeps
- batch saturation fits
- context-linearity measurements
- long-context capacity pushes

### Fused KV quantization
- INT4 KV cache quantization with kernel fusion
- fused vs non-fused pipeline comparisons
- cross-GPU validation of the memory-traffic story

### Sensitivity, protection, and tiering
- selective high-precision protection of sensitive layers
- mixed-precision KV schemes
- bounded-protection (`k*`) experiments

## Current References

Use these together:

- [AR Decode Bottleneck](https://mcgrof.github.io/knlp/ar_decode_bottleneck.html) — structural explanation of why autoregressive decode rereads KV state every step
- [Decode Scaling Visualization](https://mcgrof.github.io/knlp/kv_bandwidth_visualization.html) — empirical cross-GPU decode scaling and bandwidth view
- [Fused KV Quantization](https://github.com/mcgrof/knlp/blob/main/docs/fused_kv_quantization.md) — fused-kernel overview, code pointers, and forward-looking activation work
- [Paper: Memory-Traffic Saturation in Autoregressive Transformer Decode](https://github.com/mcgrof/paper-memory-decode) — the published empirical study validating BPA's core findings across 14 models and 4 GPU architectures

## Status

Treat BPA as an active line of work with both strong systems results and open
research headroom.

The stable public message today is simple:

- decode is memory-bound,
- fused quantization is a real win because it attacks memory traffic directly,
- BPA remains useful only if it changes the real decode bottleneck rather than
  only changing abstract attention structure.

## Research Headroom

Do not treat fused KV quantization as the end of the line. Sparse and selective
attention work still leaves room for bandwidth-aware decode ideas.

One plausible next direction is to combine BPA-style bandwidth constraints with
FIM-guided allocation:

- vary block size by layer sensitivity,
- vary protection policy by FIM trace,
- spend bandwidth where the model appears to do the most work.
