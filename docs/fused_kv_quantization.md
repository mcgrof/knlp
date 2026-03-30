# Fused KV Quantization

Fused KV quantization is the strongest concrete intervention that came out of
the decode-side memory-traffic work in `knlp`. The important point is not
simply that INT4 is smaller than FP16 on paper. The important point is that
autoregressive decode pays for memory traffic in the kernel that actually runs,
and compression only helps if it reduces that real traffic. That is why fusion
matters.

The full empirical study is published in **Memory-Traffic Saturation in
Autoregressive Transformer Decode**:
<https://github.com/mcgrof/paper-memory-decode>

That work benchmarks 14 open-weight models across W7900, A100, H100, and B200
and shows three things clearly. First, decode speedup from fused KV compression
follows a batch-driven saturation pattern. Second, kernel fusion rather than
quantization alone is the mechanism that turns compression into a real decode
win. Third, a short runtime calibration test can identify models that require
more conservative key precision.

## Table of Contents

- [What "fused" means here](#what-fused-means-here)
- [Related documentation](#related-documentation)
- [Implementation](#implementation)
- [W7900-specific fused decode provenance](#w7900-specific-fused-decode-provenance)
- [Calibration and ratio classifier](#calibration-and-ratio-classifier)
- [Paper Results Summary](#paper-results-summary)
- [Next Directions](#next-directions)
- [Status](#status)

## What "fused" means here

A non-fused path reads quantized KV, dequantizes into an intermediate buffer,
writes that buffer back out, and only then runs attention over the expanded
representation. That extra read/write path can erase much or all of the benefit
of using a smaller KV format in the first place.

A fused path does the dequantization inside the attention kernel. The expanded
values exist only long enough to be consumed by the decode computation instead
of being written back as a separate global-memory artifact. That is the entire
point. The practical distinction is therefore simple: non-fused quantization
can be neutral or counterproductive, while fused quantization can translate
compression into real decode speedup because it removes traffic from the path
that actually dominates decode.

## FlashAttention: PyTorch SDPA vs external flash\_attn

When this document or the proof artifacts refer to "FlashAttention",
the implementation is **PyTorch SDPA** (`torch.nn.functional.scaled_dot_product_attention`)
with the `FLASH_ATTENTION_FORCED` backend hint. PyTorch 2.4+ ships its
own FlashAttention-2 kernel inside SDPA; no external package is required.
The H100 proof pass explicitly confirmed that the `flash_attn` pip
package (Dao-AILab) was **not installed**, so there is no ambiguity
about which FlashAttention ran. The `FLASH_ATTENTION_FORCED` setting
makes the backend deterministic and verifiable in the JSON proof
artifacts. For the full H100 ablation against FlashAttention-backed
baselines, see [H100 FlashAttention vs fused INT4 decode ablation](benchmarks/h100_flashattention_fused_ablation.md).

## Related documentation

Use these references together. The standalone systems diagnosis is [Memory-Traffic Saturation in Autoregressive Decode](https://github.com/mcgrof/knlp/blob/main/docs/memory_traffic_saturation_in_autoregressive_decode.md). The cross-model companion is [Cross-Model KV Sensitivity and Decode Scaling](https://github.com/mcgrof/knlp/blob/main/docs/cross_model_kv_sensitivity_and_decode_scaling.md). The paper visualization is the [Memory-Traffic Saturation interactive walkthrough](https://mcgrof.github.io/knlp/paper_memory_decode.html). For broader BPA background, use [docs/bpa.md](https://github.com/mcgrof/knlp/blob/main/docs/bpa.md) and [docs/paper/bpa/evolution.md](https://github.com/mcgrof/knlp/blob/main/docs/paper/bpa/evolution.md). For supporting visual context, use [AR Decode Bottleneck](https://mcgrof.github.io/knlp/ar_decode_bottleneck.html), [Decode Scaling Visualization](https://mcgrof.github.io/knlp/kv_bandwidth_visualization.html), and [Ridge Point Visualization](https://mcgrof.github.io/knlp/ridge_point.html). For the statistical side of the sensitivity discussion, use [Spearman ρ Visualization](https://mcgrof.github.io/knlp/spearman_rho.html). For the H100 FlashAttention fused decode proof, see [H100 FlashAttention vs fused INT4 decode ablation](benchmarks/h100_flashattention_fused_ablation.md).

## Implementation

The implementation in `knlp` has two layers, and it helps to keep them
separate. One layer is generic unpack/dequant machinery used for microbenchmarks
and simpler experiments. The other layer is a W7900-specific fused decode
provenance path that documents how the paper-grade AMD decode kernels evolved.
That second layer is not the whole fused-KV story across all GPUs. It is the
particular W7900 decode lineage that was important for the AMD paper path.

The generic side starts with `kernels/triton_kernels.py` and
`scripts/kv_triton_benchmark.py`. Those are useful when the goal is to study
unpack/dequant behavior in isolation or run small microbenchmarks.

If you want a simple starting point, begin with these files:
- [kernels/triton_kernels.py](https://github.com/mcgrof/knlp/blob/main/kernels/triton_kernels.py)
- [kernels/triton_decode_kernels.py](https://github.com/mcgrof/knlp/blob/main/kernels/triton_decode_kernels.py)
- [scripts/kv_triton_benchmark.py](https://github.com/mcgrof/knlp/blob/main/scripts/kv_triton_benchmark.py)
- [scripts/benchmark_kv_quantized.py](https://github.com/mcgrof/knlp/blob/main/scripts/benchmark_kv_quantized.py)
- [scripts/v28_triton_int4_dequant.py](https://github.com/mcgrof/knlp/blob/main/scripts/v28_triton_int4_dequant.py)
- [scripts/v31_kernel_bench.py](https://github.com/mcgrof/knlp/blob/main/scripts/v31_kernel_bench.py)

Use the generic module for small unpack/dequant experiments.

## W7900-specific fused decode provenance

The W7900-specific decode side starts with `kernels/triton_decode_kernels.py`
and the ablation source `scripts/v31_kernel_bench.py`. This section is here to
document the AMD/W7900 fused decode path specifically, because that path has
its own optimization history and should not be mistaken for the entire
cross-GPU fused quantization story.

The W7900 path in the paper evolved through a set of kernel variants rather
than a single frozen kernel. Pipeline B is the baseline fused INT4 decode
kernel with `BLOCK_N=64`. Pipeline C adds Delta1 scale broadcast reuse.
Pipeline D adds Delta2 RDNA3 wavefront-aware tiling with `BLOCK_N=128`.
Pipeline E combines Delta1 and Delta2 and is the paper-grade W7900 production
path.

Use the decode module and the v31 lineage when you need
provenance-consistent W7900 decode experiments.

## Calibration and ratio classifier

Kernel fusion is only part of the deployment story. Some model families tolerate
uniform low-precision KV quantization, while others require asymmetric
key/value policies. In the current work, the practical compatibility check is a
runtime ratio classifier. It compares INT8-key and INT6-key logit errors on a
small calibration set and uses the INT6/INT8 error ratio to decide whether a
model needs conservative key precision.

Start with these files:
- [scripts/bpa_h100_exp4_ratio_classifier.py](https://github.com/mcgrof/knlp/blob/main/scripts/bpa_h100_exp4_ratio_classifier.py)
- [scripts/marin_w7900_ratio_classifier.py](https://github.com/mcgrof/knlp/blob/main/scripts/marin_w7900_ratio_classifier.py)
- [docs/kv_plugin/calibration_guide.md](https://github.com/mcgrof/knlp/blob/main/docs/kv_plugin/calibration_guide.md)

The fused kernel and the ratio classifier solve different parts of the problem.
The fused kernel gives you the speedup path. The ratio classifier tells you when
aggressive key quantization is safe and when an asymmetric setting is the right
choice.

### Current H100 bounded decode policy

The current H100 decode policy is bounded and model-sensitive rather than
universal. The strongest BPA follow-up result is that Hopper decode should be
treated as a dispatch problem: first decide FP16 reference (`P0`) vs fused,
and for Qwen-family models choose `P3` vs `P5` inside the fused branch.

Current working rules from the 2026-03-21 H100 policy loop:

- `qwen25_1.8b`
  - `B < 3` -> `P0`
  - `3 <= B < 8` -> `P5`
  - `B >= 8` -> `P3`
- `qwen25_7b`
  - `B = 1` -> `P0`
  - `B = 2, T = 2048` -> `P5`
  - otherwise prefer `P3` in the tested regime
- `mistral_7b`
  - conservative current rule: fused from about `B >= 3`
- `llama31_8b`
  - fused from about `B >= 2`

This bounded policy is currently the best practical H100 decode rule from the
BPA iteration process.

### W7900 / Marin local calibration note

A dedicated W7900-safe Marin ratio-classifier path now exists in
`scripts/marin_w7900_ratio_classifier.py`. This path uses
`device_map=None` followed by `model.to("cuda")` because prune's ROCm
setup showed bad behavior with `device_map="auto"` and because the shared
W7900 workstation should not be treated like an H100 validation lane.

On prune's AMD Radeon Pro W7900, a smoke calibration run for
`marin-community/marin-8b-base` at `prompt_len=512`, `n_prompts=2` produced
`ratio_INT6_INT8 = 5.0`, above the threshold `3.0`, so the smoke result says
Marin needs conservative / FP16 key handling. Treat this as a strong local
policy signal, not the final full-scale calibration.

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

## Serving Integration Status

### vLLM branch with fused INT4 support (2026-03-25)

The `--kv-cache-dtype int4_fused` flag is now implemented in the vLLM
branch `20250325-fused-quantization` on `/data/vllm`. This branch adds
a real `FusedInt4AttentionBackend` that integrates the fused decode
kernel with vLLM's paged block-table cache model.

**Kernel-level validation on AMD W7900** (ROCm 6.4, Triton 3.5.1):
2.5x-5.4x decode speedup, cosine similarity = 1.000000 across all
test points (B=1,2,4,8 x D=64,128). H100 serving validation is the
next step.

### What works today

These components run end-to-end:

| Layer | What it does | Key files |
|-------|-------------|-----------|
| **vLLM int4_fused backend** | Full attention backend with paged cache, fused decode, prefill fallback | `/data/vllm/vllm/v1/attention/backends/fused_int4.py` |
| **vLLM smoke benchmark** | Kernel-level FP16 vs fused INT4 comparison with JSON manifest | `/data/vllm/benchmarks/fused_int4_smoke.py` |
| **vLLM unit tests** | Config, pack/unpack, cache write, decode correctness, backend class | `/data/vllm/tests/kernels/attention/test_fused_int4.py` |
| Triton fused decode kernels | INT4 dequant inside attention, online softmax, GQA support | `kernels/triton_decode_kernels.py`, `scripts/v31_kernel_bench.py` |
| 3-way ablation benchmark | P0 (FP16+FlashAttention), P1 (INT4 dequant+FlashAttention), Pfused (Triton fused) | `scripts/spev01/tier5_fused_decode.py` |
| Hook-based KV compression wrapper | Forward hooks on HF models for per-layer KV quantization | `gpt2/compression/wrapper_hooks.py` |
| Ratio classifier | Runtime INT6/INT8 calibration to identify sensitive models | `scripts/bpa_h100_exp4_ratio_classifier.py` |
| Generic INT4/INT8 unpack/dequant | Microbenchmark machinery | `kernels/triton_kernels.py` |

### vLLM integration checklist

The five items from the original gap analysis and their current status:

1. **KV cache dtype registration** --- **Done.** `int4_fused` is a
   valid `CacheDType` in `vllm/config/cache.py`. `is_quantized_kv_cache()`
   recognises it. `FUSED_INT4` is registered in the backend registry.
   The CUDA platform routes `int4_fused` requests to the `FUSED_INT4`
   backend exclusively.

2. **Paged attention kernel** --- **Done.** The `fused_int4_decode`
   Triton kernel reads packed INT4 bytes from the paged KV cache,
   resolves physical block addresses via block-table lookup, handles
   variable-length sequences, and uses online softmax with even/odd
   Q decomposition for the packed K dot product. K and V have
   independent scale tensors for future asymmetric K/V support.

3. **Cache manager** --- **Partial.** The backend declares cache shape
   `(2, num_blocks, block_size, num_kv_heads, head_size//2)` in uint8.
   Scale tensors are lazily allocated in the attention impl on first
   forward call. For production, these should be integrated with the
   `CacheEngine` allocation path.

4. **Prefill path** --- **Done (fallback).** Prefill uses
   `torch.nn.functional.scaled_dot_product_attention` directly on the
   FP16 K/V from the model, with explicit logging:
   `fallback_reason=prefill_not_fused`. The `reshape_and_cache_int4`
   kernel quantizes K/V to INT4 when writing into the cache.

5. **Bounded dispatch integration** --- **Stubbed.** The backend
   accepts a `block_n` parameter and the code has a placeholder for
   batch-size-dependent BLOCK_N selection (the bounded dispatch
   policy). The CUDA platform priority list routes `int4_fused`
   exclusively to `FUSED_INT4`, so activating bounded dispatch
   (B=1->FlashAttention, B>=2->fused) requires scheduler-level
   routing that is not yet implemented.

### A100 serving-path reconciliation (2026-03-29)

The March 29 A100 runs resolved the earlier scientific blocker but also clarified
that the serving-path problem is not finished. Two result bundles matter. The
first bundle (`fused-quant/a100-20260329-135620/` in `knlp-key-results`) proved
that the fused INT4 kernel, roundtrip path, and stride-view cache layouts are
numerically sound on A100, and it reproduced the expected kernel-level speedup
shape. But its end-to-end MSL sweep used a permissive acceptance rule that
counted answer-prefix matches as success even when the continuation clearly
contained garbage tails such as repeated `pérdida`.

The second bundle (`fused-quant/a100-20260329-2035/`) tightened that
interpretation and showed the real serving result: once fused decode actually
activates, the standalone kernel still passes but the full vLLM serving path can
produce corrupted output. That means the grouped-scale / packed-contract fear is
no longer the main blocker. The kernel itself is not the bug. The remaining
blocker is the serving integration path: actual cache allocation, prefill write
semantics, block-table usage, or another surrounding lifecycle mismatch.

The practical conclusion is to separate **kernel correctness** from **serving
correctness**. A100 now says the former is solved and the latter still needs
instrumentation. The next useful work is therefore to trace the exact vLLM cache
allocation and write path used during serving, compare it against the passing
standalone diagnostic path, and tighten end-to-end validators so garbage
continuations never count as success again.

### What remains for production

- **H100 serving validation**: Full vLLM API server + lm-eval with
  FlashAttention 3 baseline comparison
- **Scale tensor lifecycle**: Integrate with `CacheEngine` allocation
  instead of lazy allocation
- **CUDAGraph support**: Currently `AttentionCGSupport.NEVER`
- **Bounded dispatch**: Scheduler-level routing between FlashAttention
  and fused based on batch size
- **Asymmetric K/V**: The structure supports it (separate K/V scale
  tensors) but the actual per-K/per-V precision policy is not
  implemented

### Reading the benchmark docs

The runbook, quickstart, and smoke-test documents describe the full
evaluation protocol. The quickstart commands are now runnable against
the `20250325-fused-quantization` vLLM branch. The standalone Triton
kernel ablations (`tier5_fused_decode.py`, `v31_kernel_bench.py`)
remain available as the no-vLLM proof path and produce the JSON proof
artifacts in `docs/benchmarks/data/h100_ablation/`.

## Status

Use this document as the stable overview of the fused KV quantization result
and the paper that validates it. The paper repository is at
<https://github.com/mcgrof/paper-memory-decode>.
