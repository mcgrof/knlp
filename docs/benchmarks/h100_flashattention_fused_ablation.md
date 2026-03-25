# H100 FlashAttention vs fused INT4 decode ablation

This note records the H100 proof pass for fused KV decode against
FlashAttention-backed baselines. The purpose is to confirm that
fused INT4 decode beats FlashAttention-backed paths on Hopper
hardware and to separate the contribution of fusion from the
contribution of quantization alone.

## Environment

- GPU: `NVIDIA H100 80GB HBM3`
- PyTorch: `2.4.1+cu124`
- CUDA: `12.4`
- Triton: `3.0.0`
- External `flash_attn` package: **not installed**

## PyTorch SDPA vs external flash\_attn

These results use **PyTorch SDPA** (`torch.nn.functional.scaled_dot_product_attention`)
with the `FLASH_ATTENTION_FORCED` backend hint, not the external
`flash_attn` Python package from Dao-AILab. The distinction matters
because PyTorch 2.4+ ships its own FlashAttention-2 implementation
inside SDPA that does not require `pip install flash-attn`. When the
SDPA backend selector sees an H100 with BF16/FP16 inputs, it routes
to its internal FlashAttention kernel automatically. Setting
`FLASH_ATTENTION_FORCED` makes this deterministic and verifiable in
the proof artifacts. The external `flash_attn` package was explicitly
confirmed as not installed on the H100 pod, so there is no ambiguity
about which FlashAttention implementation ran.

## 3-way ablation paths

The proof pass measures three distinct decode paths:

**P0 — FP16 + FlashAttention (baseline)**. KV cache is stored in
FP16. Query-key-value attention runs through PyTorch SDPA with
FlashAttention forced. This is the strongest available non-quantized
decode baseline on H100.

**P1 — INT4 dequant + FlashAttention (non-fused quantized)**.
KV cache is stored in packed INT4 with per-group FP16 scales.
Before attention, the INT4 values are unpacked and dequantized into
a full FP16 intermediate buffer, which is then written to global
memory and fed into the same FlashAttention SDPA path. This path
isolates the cost of quantization without fusion: it saves cache
memory but adds a dequantization write-read round-trip.

**Pfused — Triton fused INT4 decode kernel**. KV cache is stored
in the same packed INT4 format. Dequantization happens inside the
Triton attention kernel, so expanded FP16 values never leave
registers. The kernel reads packed INT4 bytes, unpacks and scales
in-register, computes the dot product against the query, and
accumulates via online softmax. No intermediate FP16 buffer is
materialized in global memory.

The proof artifacts record these backends explicitly:

- `baseline_backend = FLASH_ATTENTION_FORCED`
- `dequant_backend = FLASH_ATTENTION_FORCED`
- `fused_backend = TRITON_FUSED_INT4_DECODE`

## Artifacts

- `docs/benchmarks/data/h100_ablation/tier5_flash_proof_qwen25_7b.json`
- `docs/benchmarks/data/h100_ablation/tier5_flash_proof_qwen25_14b.json`
- `scripts/spev01/tier5_fused_decode.py`

## Qwen2.5-7B results

Config: H=28, KV=4, head\_dim=128 (GQA 7:1).

| Batch | Context |  P0 (ms) |  P1 (ms) | Fused (ms) | Fused/P0 | Fused/P1 |
|------:|--------:|---------:|---------:|-----------:|---------:|---------:|
|     1 |   2048  |   0.071  |   0.273  |     0.129  |   0.55x  |   2.11x  |
|     2 |   2048  |   0.114  |   0.318  |     0.112  |   1.03x  |   2.85x  |
|     4 |   2048  |   0.212  |   0.418  |     0.110  |   1.93x  |   3.80x  |
|     8 |   2048  |   0.368  |   0.611  |     0.119  |   3.09x  |   5.12x  |
|    16 |   2048  |   0.708  |   1.197  |     0.187  |   3.79x  |   6.41x  |
|     1 |   8192  |   0.199  |   0.458  |     0.313  |   0.64x  |   1.47x  |
|     4 |   8192  |   0.737  |   1.225  |     0.308  |   2.39x  |   3.98x  |
|    16 |   8192  |   2.660  |   4.491  |     0.702  |   3.79x  |   6.40x  |
|     1 |  32768  |   0.672  |   1.168  |     1.114  |   0.60x  |   1.05x  |
|     4 |  32768  |   2.885  |   4.720  |     1.327  |   2.17x  |   3.56x  |
|    16 |  32768  |  10.609  |  17.638  |     2.668  |   3.98x  |   6.61x  |

Aggregate over all 25 points: mean fused/P0 = **2.23x**, mean
fused/P1 = **4.05x**, max fused/P0 = **3.98x** (B=16, T=32768),
max fused/P1 = **6.61x** (B=16, T=32768).

## Qwen2.5-14B results

Config: H=40, KV=8, head\_dim=128 (GQA 5:1).

| Batch | Context |  P0 (ms) |  P1 (ms) | Fused (ms) | Fused/P0 | Fused/P1 |
|------:|--------:|---------:|---------:|-----------:|---------:|---------:|
|     1 |   2048  |   0.091  |   0.299  |     0.132  |   0.69x  |   2.27x  |
|     2 |   2048  |   0.149  |   0.350  |     0.109  |   1.37x  |   3.20x  |
|     4 |   2048  |   0.299  |   0.542  |     0.116  |   2.59x  |   4.69x  |
|     8 |   2048  |   0.539  |   1.027  |     0.173  |   3.11x  |   5.94x  |
|    16 |   2048  |   0.989  |   1.947  |     0.290  |   3.41x  |   6.72x  |
|     1 |   8192  |   0.262  |   0.504  |     0.304  |   0.86x  |   1.66x  |
|     4 |   8192  |   1.083  |   2.042  |     0.372  |   2.91x  |   5.49x  |
|    16 |   8192  |   3.724  |   7.310  |     1.015  |   3.67x  |   7.20x  |
|     1 |  32768  |   0.959  |   1.927  |     1.373  |   0.70x  |   1.40x  |
|     4 |  32768  |   4.311  |   7.843  |     1.368  |   3.15x  |   5.73x  |
|    16 |  32768  |  15.115  |  29.203  |     3.966  |   3.81x  |   7.36x  |

Aggregate over all 25 points: mean fused/P0 = **2.30x**, mean
fused/P1 = **4.64x**, max fused/P0 = **3.81x** (B=16, T=32768),
max fused/P1 = **7.36x** (B=16, T=32768).

## Key findings

**Batch-1 is below crossover.** At batch 1, the fused kernel is
slower than P0 across all context lengths for both models. The
Triton kernel launch overhead and reduced occupancy at B=1 dominate
the bandwidth savings. This matches the bounded dispatch policy
documented in `docs/fused_kv_quantization.md`: the H100 policy
sends B=1 to P0 rather than the fused path.

**Crossover at batch 2.** For both models, fused decode matches or
beats P0 starting at B=2. The crossover is sharper for the 14B model
(1.37x at B=2/T=2048) than the 7B (1.03x), consistent with the
larger KV head count producing more memory traffic to amortize the
kernel launch cost.

**Fusion vs quantization alone.** The P1 path (INT4 dequant into
FlashAttention) is always slower than P0 because the dequantization
write-read round-trip costs more than the memory saved by smaller
cache reads. Fused decode eliminates this round-trip entirely. The
fused/P1 speedup (4--7x) is consistently larger than fused/P0
(2--4x), confirming that fusion, not quantization, is the mechanism
that converts compression into real speedup.

**14B scales better.** The 14B model shows higher fused/P1 speedups
than 7B at every tested point. More KV heads (8 vs 4) means more
total dequantization traffic in P1 and more opportunity for the fused
path to eliminate it.

## Provenance note

The initial H100 run used `torch.cuda.get_device_properties(0).total_mem`,
which is an incorrect PyTorch attribute (the correct name is
`total_memory`). This caused the OOM estimator to crash before any
benchmarks ran. The fix was a one-character change in
`scripts/spev01/tier5_fused_decode.py` (`total_mem` to
`total_memory`). The JSON proof artifacts were produced after this
fix. Committed alongside the artifacts for traceability.

## Recommended wording

### Safe technical wording

> On H100, for Qwen2.5-7B and Qwen2.5-14B, fused INT4 decode was
> compared against both an FP16 FlashAttention-forced baseline (P0)
> and an INT4 dequant into FlashAttention-forced ablation (P1). The
> FlashAttention backend was PyTorch SDPA with FLASH_ATTENTION_FORCED,
> not the external flash_attn package.

### Stronger result summary

> Fused decode loses at batch-1 operating points, but crosses over by
> batch 2 and reaches 3.8--4.0x speedup versus FP16 FlashAttention
> and 6.6--7.4x versus dequantized FlashAttention at the largest
> tested points. The stronger systems result is the fused-vs-P1
> comparison, which isolates the value of fusion by holding
> quantization constant.

### Systems interpretation

> The strongest systems result is not merely that fused decode beats
> an FP16 FlashAttention baseline at moderate-to-large operating
> points. It is that fused decode substantially outperforms the more
> apples-to-apples INT4 dequant into FlashAttention path, confirming
> that the benefit comes from fusion --- eliminating the intermediate
> FP16 buffer materialization --- not from quantization alone.
