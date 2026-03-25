# H100 FlashAttention vs fused INT4 decode ablation

This note records the H100 proof pass for fused KV decode against FlashAttention-backed baselines.

## Environment

- GPU: `NVIDIA H100 80GB HBM3`
- PyTorch: `2.4.1+cu124`
- CUDA: `12.4`
- Triton: `3.0.0`
- External `flash_attn` package: **not installed**

Important: these results use **PyTorch SDPA with `FLASH_ATTENTION_FORCED`**, not the external `flash_attn` Python package.

## Ablation paths

- **P0** = FP16 + forced FlashAttention
- **P1** = INT4 -> dequant -> forced FlashAttention
- **Pfused** = Triton fused INT4 decode kernel

The proof artifacts explicitly record:

- `baseline_backend = FLASH_ATTENTION_FORCED`
- `dequant_backend = FLASH_ATTENTION_FORCED`
- `fused_backend = TRITON_FUSED_INT4_DECODE`

## Artifacts

- `docs/benchmarks/data/h100_ablation/tier5_flash_proof_qwen25_7b.json`
- `docs/benchmarks/data/h100_ablation/tier5_flash_proof_qwen25_14b.json`
- `docs/benchmarks/data/h100_ablation/tier5_flash_proof_qwen25_32b.json`
- `docs/benchmarks/data/h100_ablation/tier5_flash_proof_qwen3_8b.json`
- `docs/benchmarks/data/h100_ablation/tier5_flash_proof_qwen3_14b.json`
- `docs/benchmarks/data/h100_ablation/tier5_flash_proof_qwen3_32b.json`
- `docs/benchmarks/data/h100_ablation/tier5_flash_proof_deepseek_r1_distill_qwen_32b.json`
- `scripts/spev01/tier5_fused_decode.py`

## Qwen2.5-7B summary

- Mean fused vs **P0**: ~`2.23x`
- Mean fused vs **P1**: ~`4.05x`
- Max fused vs **P0**: `3.977x`
- Max fused vs **P1**: `6.612x`

## Qwen2.5-14B summary

- Mean fused vs **P0**: ~`2.30x`
- Mean fused vs **P1**: ~`4.64x`
- Max fused vs **P0**: `3.811x`
- Max fused vs **P1**: `7.364x`

## Interpretation

- Fused decode still loses at tiny batch-1 points.
- The crossover to beating FP16 FlashAttention is usually at batch 2.
- The stronger systems result is **fused vs INT4->dequant->FlashAttention**, which isolates the value of fusion instead of quantization alone.

## Recommended wording

### Safe technical wording

> On H100, for Qwen2.5-7B and Qwen2.5-14B, fused INT4 decode was compared against both an FP16 **FlashAttention-forced** baseline and an INT4→dequant→**FlashAttention-forced** ablation.

### Stronger result summary

> Fused decode loses at tiny batch-1 points, but crosses over by batch 2 for nearly all tested contexts and reaches roughly **3.8–4.0x** speedup versus FP16 FlashAttention and **6.6–7.4x** versus dequantized FlashAttention at the largest tested points.

### Systems interpretation

> The strongest systems result is not merely that fused decode beats an FP16 FlashAttention baseline at moderate-to-large operating points. It is that fused decode substantially outperforms the more apples-to-apples INT4→dequant→FlashAttention path, confirming that the benefit comes from fusion --- eliminating the intermediate FP16 buffer materialization --- not from quantization alone.

## Qwen2.5-32B summary

- See `docs/benchmarks/data/h100_ablation/tier5_flash_proof_qwen25_32b.json` for the full table.
- Pattern remains consistent with 14B: fused loses at tiny batch-1 points, crosses over cleanly once batch rises, and remains substantially faster than the INT4->dequant->FlashAttention path at moderate and large operating points.

## Newer open-model extensions

Additional H100 proof artifacts were collected for newer open-weight families:

- Qwen3-8B
- Qwen3-14B
- Qwen3-32B
- DeepSeek-R1-Distill-Qwen-32B

These runs use the same three-way ablation shape as the Qwen2.5 results:

- FP16 + forced FlashAttention
- INT4 -> dequant -> forced FlashAttention
- Triton fused INT4 decode

The qualitative pattern remained stable: fused decode still loses at tiny batch-1 points, crosses over once batch rises, and remains substantially faster than the dequantized FlashAttention path at moderate and large operating points.
