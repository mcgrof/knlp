# Fused KV Quantization

Fused KV quantization is the strongest concrete systems result currently exposed
by the BPA line in `knlp`.

The public message is intentionally simple:

- autoregressive decode is dominated by KV-memory traffic,
- quantization only helps if it reduces that traffic in the real decode kernel,
- and **fusion is the difference between a real speedup and a fake one**.

## What "fused" means here

A non-fused pipeline typically:

1. reads quantized KV,
2. dequantizes into an intermediate buffer,
3. writes that buffer back to memory,
4. then runs attention over the dequantized representation.

That extra read/write path can destroy the benefit of using a smaller KV format.
The memory traffic saved by quantization is partially or fully paid back in
intermediate buffer movement.

A fused pipeline instead dequantizes inside the attention kernel, so the decode
loop avoids unnecessary intermediate memory writes.

## Why it matters

This is the key BPA lesson in concrete form.

The point is not merely that INT4 is smaller than FP16 on paper. The point is
that decode is bottlenecked by KV-memory movement, so the implementation only
helps if it reduces real memory traffic in the kernel that dominates decode.

That is why the practical distinction is:

- **non-fused quantization**: can be neutral or counterproductive
- **fused quantization**: can translate compression into real decode speedup

## Current public scope in knlp

At the moment `knlp` publicly exposes the story, kernels, and evolving docs for
this work, while the paper framing and paper-specific artifact packaging are
still being polished.

Current public components include:

- BPA overview: [docs/bpa.md](https://github.com/mcgrof/knlp/blob/main/docs/bpa.md)
- BPA evolution: [docs/paper/bpa/evolution.md](https://github.com/mcgrof/knlp/blob/main/docs/paper/bpa/evolution.md)
- structural decode explainer: [AR Decode Bottleneck](https://mcgrof.github.io/knlp/ar_decode_bottleneck.html)
- empirical decode explainer: [Decode Scaling Visualization](https://mcgrof.github.io/knlp/kv_bandwidth_visualization.html)

## Kernels and open-source implementation

The open-source implementation lives in the `knlp` repository and includes the
Triton-based work that the BPA / fused-KV story grew out of.

The exact kernel set and experimental harnesses are still being cleaned up and
better documented, but the repository is the public home for:

- Triton kernels,
- decode experiments,
- profiling / benchmarking infrastructure,
- and the docs that explain the current systems story.

## Ongoing experiments

The current open story is still evolving in several directions:

- broader cross-GPU decode validation,
- long-context behavior under memory pressure,
- selective protection / mixed-precision KV tiering,
- and follow-on BPA-style architectures that try to reduce the number of KV
  entries touched per decode step.

So fused KV quantization should be read as the strongest current result, not as
proof that BPA is finished as a research direction.

## Where BPA may still go

BPA still has research headroom beyond fused quantization. Recent sparse /
selective attention work such as MoBA suggests there is still room to explore
bandwidth-aware decode designs.

One natural direction is to combine BPA-style bandwidth thinking with
FIM-guided structure:

- prioritize higher-FIM-trace layers more conservatively,
- vary block sizes by layer sensitivity,
- and explore whether selective block attention can spend bandwidth where the
  model appears to do the most work.

That is not a finished claim here. It is a live research direction.

## Status

This document is intentionally bland and stable. It gives the public open-source
view of the fused KV quantization result while the paper and paper-specific
narrative continue to be polished.
