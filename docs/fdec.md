# fdec — fight decode

`fdec` is a knlp regime for composing bleeding-edge, mutually-compatible decode
optimizations into one platform. The goal is not a single new technique; it is a
growing, reproducible set of "Lego bricks" that we verify compose without mismatch,
so any new compatible decode optimization can be brought up on knlp quickly and
added to the stack over time.

It is built on the existing paper-memory-decode reproduction machinery
(`Kconfig.decode`, `Makefile.decode`, the `decode-*` defconfigs), which already
clones and pins the companion serving repos (vllm-asym, flashinfer-asym,
lmcache-asym). `fdec` reorients that machinery from "reproduce the paper" to
"compose the known-good bricks and verify they do not fight each other."

(The name is "fight decode"; only the fight variant is documented.)

## Why a platform, not one technique

The decode paper (`paper-memory-decode`) establishes the constraints the platform
must respect: decode is memory-traffic-bound; the dominant byte pool shifts from
model weights to KV cache as context and batch grow (at 7B with short context, KV
is under 7% of decode bandwidth and weights dominate 93-99%); and serving overflow
KV from a secondary tier under dense attention is bandwidth-infeasible (NVLink/PCIe5/
CXL are 10-1600x short of the requirement, and even speculative decoding leaves a
3-28x gap). So no single brick wins decode: the weights pool, the KV pool, and the
LM-head pool each need their own reducer, and the offload problem is only escaped
structurally, by linear attention.

## The bricks (per byte pool)

| brick | pool it attacks | status |
|---|---|---|
| Asym K16/V8 KV quant | KV (grows with context) | proven: 1.38x decode, prefix-cache-safe (Mode 1), via vllm/flashinfer/lmcache-asym |
| FIM-guided weight quant | model weights (dominant at small models / short ctx) | exists (knlp BPA, Fisher sensitivity); stack-integration open |
| LM-head idblock routing | LM-head (fixed, large for big vocab) | proven: certified-lossless, ~1.66x on that step |
| Linear-attention model class | removes the KV pool | escape: GDN / Trellis / Mamba2 / RWKV keep a fixed recurrent state, so there is no KV-bandwidth bound and no offload problem |

## Two model classes, two brick subsets

- **Dense / GQA transformer** (Qwen, Llama, Marin): the KV pool grows with context,
  so the brick set is asym KV quant + FIM weight quant + LM-head idblock.
- **Linear attention** (Gated DeltaNet, Trellis, Mamba2, RWKV): there is no K/V
  cache to quantize, so asym KV quant does not apply; the brick set is FIM weight
  quant + LM-head idblock. This class is the structural escape from the dense-
  attention offload bandwidth bound the paper proved fatal.

## The compatibility matrix (the deliverable)

For each brick pair x model, verify the pair composes (runs correctly in one decode
loop) and measure the aggregate speedup and quality. Cells, with what to check:

- **asym KV x LM-head idblock** — both proven solo; verify they co-exist in one
  vLLM decode loop. The easy first cell.
- **asym KV x FIM weight quant (D5, riskiest)** — FIM/GGUF is natively
  llama.cpp-side; asym KV is vLLM/FlashInfer. They compose only if weight quant is
  vLLM-native (AWQ/GPTQ/fp8 weights guided by the Fisher map), not GGUF. Resolve
  this first; if it fails, the entire dense-model brick set has a hole.
- **FIM weight quant x LM-head idblock (D4)** — does the idblock lossless
  certificate survive a quantized LM-head, or does FIM keep the head high-precision
  (preserving the certificate)?
- **linear-attn model x (FIM weight quant + LM-head idblock)** — do open-weights
  GDN/Mamba2/Trellis checkpoints load and quantize; does the idblock certificate
  hold for the architecture; asym KV N/A.

Cell outcome record: {composes? yes/no, aggregate speedup vs fp16, quality delta,
notes}. The matrix grows as new bricks are added.

## Open-weights linear-attention models

Known open-weights candidates: Mamba2, RWKV, Gated DeltaNet (via the fla project,
already wired in `trellis_lm/linear_baselines_fla*.py`), and recent hybrids. Trellis
open weights need verifying (`trellis_lm/` is training code, not necessarily a
released checkpoint). Whether each composes with FIM weight quant + LM-head idblock
is unverified and is a matrix row.

## Start order (known-good first)

1. **asym KV + LM-head idblock on Qwen2.5-7B** — the known-good baseline brick; both
   proven solo, verify they compose and measure the aggregate.
2. **asym KV x FIM weight quant (D5)** — resolve whether weight quant can live in
   the vLLM-asym stack at all (vLLM-native Fisher-guided quant vs GGUF/llama.cpp).
3. **linear-attn lane** — load an open GDN/Mamba2 checkpoint, verify FIM weight quant
   + LM-head idblock apply (asym N/A), as the dense-is-prohibitive escape.

## Kconfig regime (planned)

`defconfig-fdec` selects the brick set and the model class; the runner reuses the
decode companion-repo build (vllm/flashinfer/lmcache-asym) and runs the
compatibility matrix, emitting per-cell results to `results/fdec/`. Bricks are
Kconfig booleans (`CONFIG_FDEC_ASYM_KV`, `CONFIG_FDEC_FIM_WEIGHT_QUANT`,
`CONFIG_FDEC_LMHEAD_IDBLOCK`, `CONFIG_FDEC_LINEAR_ATTN`) so a defconfig names a point
in the matrix. To be wired on top of `Kconfig.decode` / `Makefile.decode`.
