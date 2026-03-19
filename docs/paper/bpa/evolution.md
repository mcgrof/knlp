# From RGSA to BPA to Fused KV Quantization

This document explains how the BPA story in `knlp` evolved.

## Phase 1: RGSA / Router Intuition

The earlier RGSA work attacked attention cost through retrieval and routing.
The central idea was simple: if attending to all context is expensive, perhaps a
model can route to only the most relevant chunks.

That line of work produced useful infrastructure and sharpened an important
intuition:

- full-context access is expensive,
- selective access may be necessary,
- and attention cost should be treated as a systems problem, not just a modeling
  flourish.

But RGSA was still framed mainly as an architecture idea.

## Phase 2: BPA Reframes the Problem

BPA emerged when the project shifted from architectural intuition to direct
measurement of decode behavior.

The key discovery was that autoregressive decode is dominated by repeated KV
cache reads. Once that was measured across different GPUs, the framing became
clearer:

- decode is limited by KV-memory traffic,
- context scaling hurts because the model rereads more state per token,
- batch scaling saturates according to hardware-specific bandwidth,
- and long context is fundamentally a memory-system problem.

BPA therefore became the name for the broader research framing: attention should
be studied under an explicit bandwidth budget.

## Phase 3: Fused KV Quantization Becomes the Concrete Result

The strongest concrete systems result to come out of that framing is fused KV
quantization.

The important lesson was not merely "INT4 is smaller than FP16." The useful
lesson was:

- non-fused quantization paths can be counterproductive,
- fused quantization reduces real memory traffic inside the kernel,
- and that is why it delivers real decode speedup.

This made the project much more grounded. The question stopped being whether a
technique sounded bandwidth-aware in theory and became whether it reduced the
actual decode bottleneck in practice.

## What Still Remains Open

The older BPA questions are still alive, but they now sit on top of a stronger
systems foundation:

- can selective KV access reduce the number of entries touched per step?
- can KV precision be tiered by sensitivity without paying unnecessary traffic?
- can we formalize a bandwidth budget that scales with hardware and not context?
- how much protected high-precision state is actually needed as models scale?

## Documentation Implication

The public `knlp` story should therefore look like this:

- `docs/rgsa.md` = historical precursor / routing-era work
- `docs/bpa.md` = current high-level BPA systems story
- `docs/ar_decode_bottleneck.html` = structural explainer for why autoregressive decode rereads KV state every step
- `docs/kv_bandwidth_visualization.html` = empirical decode-scaling / cross-GPU companion explainer
- paper-specific narratives and figures can later tighten around the final paper
  once the experiments and framing are fully locked
