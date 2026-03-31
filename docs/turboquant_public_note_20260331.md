# TurboQuant H100 A/B/C/D status and public-evidence note — 2026-03-31

## What D is
In the corrected A/B/C/D ablation:
- **A** = uncompressed FP16 anchor
- **B1** = TurboQuant practical reference
- **B2** = TurboQuant + QJL
- **C1** = TurboQuant + fused Triton execution
- **D** = **current fused quantization baseline**

So **D is our incumbent fused quantization path**, not an uncompressed baseline.

## Durable artifact roots
Primary durable sink on prune:
- `/data/knlp-key-results/fused-quant/turboquant-abcd-h100-20260331T195241Z/`

Repo-local copy for source-tree context:
- `/data/knlp/results/turboquant-abcd-h100-20260331T195241Z/`

## Corrected A/B/C/D result summary
At `head_dim=128` on the tested H100 matrix:
- **A**: MSE=0, CosSim=1.000000, latency=32.3 us
- **B1**: MSE=7.350e-05, CosSim=0.995605, latency=372.4 us
- **B2**: MSE=2.319e-05, CosSim=0.998535, latency=851.1 us
- **C1**: MSE=7.350e-05, CosSim=0.995605, latency=1066.2 us
- **D**: MSE=7.351e-05, CosSim=0.995605, latency=312.7 us

Interpretation:
- **B1 and D are effectively identical on quality** in this regime.
- **D is still faster than B1** by about 19%.
- **QJL improves quality numerically** but with a very large latency cost.
- **C1 does not rescue TurboQuant** here; it is slower than D and even slower than B1 in the current implementation.

## Public evidence that changes the fair interpretation
Public community work suggests a more nuanced read than our first quick summary.

### TheTom / turboquant_plus
Key public signals:
- early catastrophic failure due to rotation / coordinate-system bugs
- later recovery to roughly **+1.2% to +1.4% perplexity vs q8_0** at about **4.6x compression**
- practical de-emphasis of QJL in working paths
- strong warning that speed claims without perplexity were invalid

### Community CUDA / llama.cpp writeups
Key signals:
- practical ports are already engineering compromises, not literal paper clones
- graph-side / amortized rotation handling matters a lot
- long-context regimes are where TurboQuant may become more defensible

## Fairer conclusion than the first quick take
The current H100 A/B/C/D run does **not** prove TurboQuant is a bad idea in general.
It proves that in our current tested regime:
- `head_dim=128`
- current implementation stack
- current bit budget

**TurboQuant does not beat the existing fused quantization baseline**.

But public evidence suggests a fairer next test should focus on:
1. **no-QJL practical TurboQuant first**
2. **long-context regimes first** where rotation overhead amortizes
3. comparison against stronger public-style practical baselines, not just abstract formulation-level baselines

## Public sources reviewed
- arXiv: `https://arxiv.org/abs/2504.19874`
- TheTom / turboquant_plus quality notes:
  `https://github.com/TheTom/turboquant_plus/blob/main/docs/quality-benchmarks.md`
- TheTom / turboquant_plus benchmark page:
  `https://github.com/TheTom/turboquant_plus/blob/main/benchmarks/benchmark_results.md`
- CUDA / llama.cpp implementation writeup:
  `https://oliverchurch.com/turboquant-for-ggml-achieving-4.57x-kv-cache-compression-in-llama.cpp.html`

## Recommended next rerun
See workspace plan:
- `/home/mcgrof/.openclaw/workspace/plans/turboquant-public-informed-rerun-20260331.md`

That rerun gives TurboQuant a more fair shot by:
- biasing toward longer contexts
- deprioritizing QJL
- explicitly checking public bug classes first
- comparing against a stronger practical baseline
