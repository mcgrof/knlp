# From RGSA to BPA to Fused KV Quantization

Use this document to place the BPA work in order.

## Phase 1: RGSA — route attention instead of paying for everything

RGSA started from the extreme-context scaling question:

**how do we push attention toward billion-token contexts without paying dense,
full-history cost everywhere?**

That led to routing and retrieval. The core intuition was right:

- full-context access is expensive,
- selective access is probably necessary,
- attention scaling is a systems problem, not only an architecture problem.

RGSA expressed that intuition as an architecture proposal.

## Phase 2: BPA — measure the decode bottleneck directly

BPA moved the work from architectural instinct to direct decode measurement.

The key result was simple and durable:

- autoregressive decode is dominated by repeated KV-cache reads,
- context scaling hurts because the model rereads more state per token,
- batch scaling saturates according to hardware bandwidth,
- long context is fundamentally a memory-system problem.

That is the BPA reframing. Do not start from abstract attention structure.
Start from decode traffic.

## Phase 3: Fused KV Quantization — turn the diagnosis into a concrete win

Fused KV quantization is the strongest current concrete result that came out of
that reframing.

The important lesson is not merely that INT4 is smaller than FP16. The lesson
is operational:

- non-fused quantization can be neutral or counterproductive,
- fused quantization reduces real traffic inside the decode kernel,
- that is why it delivers real decode speedup.

This is where the BPA story becomes concrete. Judge techniques by whether they
reduce the actual decode bill.

## What Remains Open

Do not stop at fused quantization.

The broader BPA questions remain open:

- reduce the number of KV entries touched per step,
- tier KV precision by sensitivity without wasting bandwidth,
- formalize bandwidth budgets that scale with hardware and not context,
- determine how much protected high-precision state is really needed,
- explore FIM-guided selective block attention or MoBA-style directions under
  explicit bandwidth constraints.

## Documentation Rule

Keep the `knlp` story ordered like this:

- `docs/rgsa.md` = precursor routing-era work
- `docs/bpa.md` = current BPA systems story
- `docs/ar_decode_bottleneck.html` = structural decode explainer
- `docs/kv_bandwidth_visualization.html` = empirical decode-scaling companion
- `docs/fused_kv_quantization.md` = current public fused-kernel writeup

Add paper-shaped narratives later, once the experiments and claims are locked.
