# Fused KV Quantization

Fused KV quantization is the strongest concrete systems result in the BPA line
of work in `knlp`.

The full empirical study is published as:

> **Memory-Traffic Saturation in Autoregressive Transformer Decode**
> Paper repository: <https://github.com/mcgrof/paper-memory-decode>

The paper benchmarks 14 open-weight models across four GPU architectures
(W7900, A100, H100, B200) and establishes that decode speedup from fused
KV compression follows a batch-driven saturation model, that kernel fusion
rather than quantization alone is the mechanism, and that a 2-minute runtime
calibration test identifies models requiring asymmetric key/value precision.

Start from the basic rule:

- autoregressive decode is dominated by memory traffic (weights at small
  batch/short context, KV cache at large batch/long context),
- quantization helps only if it reduces that traffic in the real decode kernel,
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

## Related Documentation

Use these references together:

- BPA overview: [docs/bpa.md](https://github.com/mcgrof/knlp/blob/main/docs/bpa.md)
- BPA evolution: [docs/paper/bpa/evolution.md](https://github.com/mcgrof/knlp/blob/main/docs/paper/bpa/evolution.md)
- structural decode explainer: [AR Decode Bottleneck](https://mcgrof.github.io/knlp/ar_decode_bottleneck.html)
- empirical decode explainer: [Decode Scaling Visualization](https://mcgrof.github.io/knlp/kv_bandwidth_visualization.html)

## Implementation

The implementation lives in `knlp`. Start with these files:

- Triton kernel module: [gpt2/compression/triton_kernels.py](https://github.com/mcgrof/knlp/blob/main/gpt2/compression/triton_kernels.py)
- kernel microbenchmarks: [scripts/kv_triton_benchmark.py](https://github.com/mcgrof/knlp/blob/main/scripts/kv_triton_benchmark.py)
- quantized KV benchmark: [scripts/benchmark_kv_quantized.py](https://github.com/mcgrof/knlp/blob/main/scripts/benchmark_kv_quantized.py)
- earlier Triton INT4/INT8 experiment path: [scripts/v28_triton_int4_dequant.py](https://github.com/mcgrof/knlp/blob/main/scripts/v28_triton_int4_dequant.py)

Use these as the current code entry points for the Triton kernels, benchmarking,
and fused KV experiments.

## Calibration and Ratio Classifier

Kernel fusion is only part of the deployment story. Some model families tolerate
uniform low-precision KV quantization, while others require asymmetric key/value
policies. In the current BPA work, the practical compatibility check is a simple
runtime ratio classifier:

- run a small set of calibration prompts,
- compare INT8-key and INT6-key logit errors,
- take the INT6/INT8 error ratio,
- use that ratio to decide whether the model needs conservative key precision.

Start with these files:

- ratio-classifier script: [scripts/bpa_h100_exp4_ratio_classifier.py](https://github.com/mcgrof/knlp/blob/main/scripts/bpa_h100_exp4_ratio_classifier.py)
- calibration guide: [docs/kv_plugin/calibration_guide.md](https://github.com/mcgrof/knlp/blob/main/docs/kv_plugin/calibration_guide.md)

Use the fused kernels and the ratio classifier together. The fused kernel gives
you the speedup path; the ratio classifier tells you when aggressive key
quantization is safe and when you need asymmetric settings.

## Paper Results Summary

The paper validates the following results that build directly on BPA work:

- **Batch-driven saturation**: On H100, decode speedup follows a Hill-type
  model ($R^2 = 0.80$) governed by batch size. Architecture parameters (GQA
  ratio, head dimension) contribute only weakly.
- **Fusion is the mechanism**: The fused INT4 kernel achieves 2.7--4.8x on
  H100 and 1.6--7.2x on W7900. The non-fused path slows inference (0.5x).
- **Memory-traffic regime dependence**: At 7B scale with short context, weight
  reads are 94--99% of per-step bandwidth. KV traffic dominates only at longer
  contexts, larger batches, or larger models.
- **KV precision asymmetry**: Values universally tolerate INT4. Key precision
  floors are model-dependent and confined to the Qwen family (INT7 cliff).
  This sensitivity attenuates with scale (catastrophic at 7B, +1.55% at 72B).
- **Runtime calibration**: A 2-minute ratio classifier (INT6/INT8 logit-error
  ratio, threshold 3.0) identifies sensitive models with 100% accuracy across
  14 models and generalizes to 72B.
- **Cross-GPU consistency**: Matched-lane measurements on W7900, A100, H100,
  and B200 show qualitatively identical decode regimes. Cross-platform
  variation is explained by sustained decode bandwidth, not headline peak HBM.

## Next Directions

### Per-layer activation sensitivity (closing paper Limitation 1)

The paper's per-layer sensitivity ranking (Section V-A through V-C) still uses
projection-weight quantization as a proxy. BPA's hook-based infrastructure
(`wrapper_hooks.py`, `quantize_intN_grouped()`) already quantizes KV cache
activations directly for the asymmetry and ratio classifier experiments.
The remaining step is a per-layer activation sensitivity sweep: quantize all
layers' KV activations to INT4, restore each layer to FP16 one at a time,
and compare the activation-based importance ranking against the weight-based
proxy. The BPA v27/v28 oracle ranking experiments did this for a few models;
extending it to all 14 models would close the gap.

### Selective bandwidth allocation

The paper shows that uniform INT4 works for 12 of 14 models, and the remaining
2 (Qwen family) need asymmetric keys. But the bounded-protection mechanism
(BPA v27: only $k^*$ layers need INT8 protection) suggests finer-grained
policies are possible:

- per-layer asymmetric precision selected by activation sensitivity ranking,
- attention-entropy-guided protection (the paper found entropy correlates at
  $r = 0.89$ on 3 models but weakens at 14 models --- a per-model calibration
  approach may still be viable),
- dynamic precision switching based on runtime context length (short context
  is weight-bound so KV precision matters less; long context is KV-bound so
  precision matters more).

### Combining with speculative decoding

The paper's speculation analysis (Section VI) reveals that KV quantization and
speculation compose super-multiplicatively at long context ($\rho$ up to 1.95
for Llama at 32K) but sub-multiplicatively at short context ($\rho \approx
0.62$ for Qwen). This creates an opportunity: BPA's fused kernel could
adaptively adjust KV precision based on the speculation acceptance rate. When
acceptance is high (short context, small batch), aggressive KV compression
adds little value; when acceptance drops (long context, large batch), the
fused kernel's bandwidth reduction becomes critical.

### Long-context activation quality

The paper validates inference stability up to 32K tokens on W7900 and 384K on
B200, but targeted retrieval benchmarks (Needle-In-A-Haystack) at extreme
lengths remain future work. BPA's hook infrastructure can run NIAH evaluations
under asymmetric KV quantization to test whether quantization errors compound
over very long contexts, which is the key unknown for production deployment at
128K+ tokens.

### Toward bandwidth-proportional attention mechanisms

The paper's core finding --- that decode is memory-traffic-limited and the
dominant traffic source shifts with operating regime --- motivates attention
mechanisms that constrain KV access proportional to available bandwidth rather
than reading the full context every step. This is the original BPA research
direction: spend bandwidth where the model does the most work. Combining
BPA-style bandwidth budgets with FIM-guided layer sensitivity, MoBA-style
sparse routing, or learned access patterns remains the longer-term open
question.

## Status

Use this document as the stable overview of the fused KV quantization result
and the paper that validates it. The paper repository is at
<https://github.com/mcgrof/paper-memory-decode>.
