# FP8 KV attention: hardware paths and tile preparation

This note separates four things that are easy to blur together:

1. the format stored in HBM;
2. the on-chip work needed to make a tile usable by the matrix unit;
3. the matrix instruction itself; and
4. the online-softmax and value-accumulation schedule around it.

The distinction matters because an FP8 cache can save HBM traffic and still
lose latency inside the tile loop. It also matters because "dequantization"
can mean anything from a register conversion to materializing a second cache
in global memory. Those operations have different traffic and scheduling costs, so the term
needs to be qualified whenever it is used.

## The short version

- A fused FP8 attention kernel should keep the cache in FP8 in HBM. It does
  **not** need to write a BF16 or FP16 copy of the cache back to HBM.
- Fused decode uses online softmax. It consumes K/V tiles incrementally; it
  does not dequantize the full sequence and then wait at one global softmax
  barrier.
- The important dependency is local to each tile: K must be in a form the QK
  matrix instruction accepts before that tile can produce scores and update
  online softmax.
- V conversion is not free. It must finish before the corresponding PV work,
  although a good schedule may overlap it with other loads and arithmetic.
- Hopper has separate BF16/FP16 and FP8 matrix-instruction families. There is
  no native BF16-query by plain-FP8-key QK instruction with a K scale operand.
- CDNA 3 has native FP8 by FP8 MFMA into FP32, but the inspected instruction
  has no scale operands. Current AITER source applies the scale outside MFMA.
- Blackwell adds block-scaled MXFP8/NVFP4 instructions that consume scale
  factors in hardware. That path uses narrow/block-scaled operands on both
  sides; it is not a plain BF16-Q by E4M3-K instruction.

The current performance question is therefore empirical:

> Does symmetric K8/V8 save more HBM time than it adds in unhidden K-tile
> preparation and pipeline pressure, compared with K16/V8?

The target microtests are in
[`fp8-attention-tile-path.html`](../fp8-attention-tile-path.html).

## The correct tile-level picture

A fused decode kernel is closer to this:

```text
for each K/V tile:
    load K tile from HBM
    prepare K operand on chip, if needed
    QK matrix multiply
    update online-softmax state

    load or consume V tile
    prepare V operand on chip, if needed
    update the output accumulator with P @ V
```

The implementation may interleave these steps across producer and consumer
warps. The logical dependencies remain:

```text
K load -> K operand preparation -> QK -> online-softmax update
V load -> V operand preparation -> PV accumulator update
```

This is not the same as:

```text
convert every K token -> compute every score -> one softmax barrier
```

That second picture describes a naive materialized implementation, not a
modern fused attention kernel.

## A useful break-even model

Compare symmetric K8/V8 with asymmetric K16/V8. V is held constant, so the
main trade is one extra byte per K element against the K8 preparation path.
A deliberately simple model is:

```text
T_sym - T_asym
  ~= unhidden(K8 preparation)
   + K8 pipeline and occupancy penalty
   - time saved by reading K8 instead of K16
```

K16/V8 wins when the first two terms exceed the last one. The terms do not
simply add in a well-pipelined kernel, so the quantity to measure is the
steady-state tile initiation interval, not just conversion instruction count.

One way to write that is:

```text
II_sym  ~= max(load K8, prepare K8 + QK, softmax update, PV)
II_asym ~= max(load K16,            QK, softmax update, PV)
```

The exact stage boundaries depend on the kernel. This is a hypothesis model,
not a substitute for Nsight traces.

## NVIDIA Hopper

Hopper's warp-group matrix operations expose BF16/FP16 and FP8 operand
families. They do not expose the particular operation wanted by an ordinary
LLM decode path with BF16 Q and a plain E4M3 K cache:

```text
BF16 Q x scaled E4M3 K -> FP32 scores
```

A kernel therefore needs another plan. Common choices are:

1. load K as FP8, convert or transform the tile on chip, then use the BF16 or
   FP16 QK path;
2. quantize Q and use an FP8 QK path, accepting the extra quantization work
   and its numerical consequences; or
3. keep K in BF16/FP16 and quantize only V.

The first choice preserves FP8 K storage and its HBM savings. Its cost is
on-chip operand preparation, not a mandatory BF16 cache written to HBM. That
preparation can still increase latency through conversion instructions,
shared-memory traffic, register pressure, synchronization, or reduced
occupancy.

K16/V8 removes the K-side preparation path entirely. It reads twice as many K
bytes as K8/V8, so whether it is faster depends on the break-even above.

## AMD CDNA 3

CDNA 3 (`gfx942`) has native FP8 by FP8 MFMA instructions such as:

```text
v_mfma_f32_16x16x32_fp8_fp8 dst, src_a, src_b, acc
```

The instruction has FP8 A and B operands and an FP32 accumulator. It does not
have scale operands. That is different from Blackwell block-scaled MMA.

A scale can sometimes be moved outside the dot product algebraically. For a
scale that is constant across the reduction dimension:

```text
Q @ (sK * Kq)^T = sK * (Q @ Kq^T)
```

Current AITER FP8 source uses the FP8 MFMA and applies a K scale after the
MFMA result. That can avoid element-by-element reconstruction, but it is a
software schedule choice, not scale absorption by the MFMA instruction.

The native FP8 by FP8 path is useful when Q is also FP8. It does not by itself
solve a BF16-query by plain-FP8-key operand mismatch. Cross-vendor performance
must therefore be measured from the actual kernel path, not inferred from the
existence of an FP8 matrix instruction.

## NVIDIA Blackwell

Blackwell adds `tcgen05.mma` and tensor memory. There are two relevant paths.

### Plain narrow types

Blackwell supports narrow FP8/FP6/FP4 matrix operations, but this still does
not create an ordinary BF16-Q by E4M3-K operation with a K scale argument.
Current FlashInfer/TensorRT-LLM generation code contains explicit transform
modes for BF16 Q with FP8 KV:

```text
Full
KOnly
SeparateKv
```

That is direct evidence that operand transformation is a real kernel design
choice on the BF16-Q/FP8-KV path. The transformation remains on chip; it does
not imply a BF16 KV cache materialized in HBM.

### Block-scaled narrow types

Blackwell's block-scaled MMA path consumes narrow operands and per-block
scale factors in hardware. For MXFP8, the conceptual operation is:

```text
D = C + (SFA * A) @ (SFB * B)
```

with a scale shared over a small K block. This is a real hardware route around
an external scale-application step. It requires a block-scaled layout for the
operands and scale tensors, and it normally means quantizing Q as well as K.

FlashInfer now contains an SM100 fused block-scaled FMHA implementation for
MXFP8/NVFP4-style QK. That establishes open-source kernel work in this area.
The remaining deployment question is narrower: whether the required paged
KV-cache decode layouts, model shapes, numerical policy, and serving dispatch
are complete and faster for the target workload.

## Why K16/V8 can be faster than K8/V8

K8/V8 reads fewer bytes:

```text
K8/V8   = 1 byte K + 1 byte V = 2 bytes per element pair
K16/V8  = 2 byte K + 1 byte V = 3 bytes per element pair
```

K16/V8 can still win when removing K operand preparation shortens the limiting
pipeline stage. The likely contributors are:

- conversion or transform instructions before QK;
- an extra shared-memory or tensor-memory movement;
- register pressure and occupancy loss;
- producer/consumer synchronization;
- a less favorable MMA layout or dispatch path; and
- poor overlap between K preparation and the next tile load.

The measured K16/V8 advantage is consistent with this hypothesis. It does not
prove which contributor is responsible. The controlled transform-mode,
dtype, cache-state, and profiler ablations on the companion page are intended
to do that.

## V conversion is not free

V conversion sits on a different dependency chain from K preparation, but it
still consumes instructions and storage bandwidth before PV can consume that
tile. It may be cheaper or easier to hide because:

- it can be scheduled near the PV consumer;
- producer warps can prepare a later V tile while consumer warps accumulate an
  earlier tile;
- the output update is a running accumulation; and
- some layouts allow scale application to be folded into the PV math.

Those are opportunities, not guarantees. "V dequant is free" should not be
used as a factual claim unless a profile shows it is fully hidden for the
specific kernel and shape.

## Quality and schedule are separate problems

The [FP8 KV-cache failure atlas](../fp8-kv-failure-atlas.html) identifies why
ordinary symmetric FP8 K fails on Qwen-family models: a large key-projection
bias can capture the scale and crush the token-varying residual. Quantizing the
pre-bias residual and keeping the fixed bias exact makes symmetric FP8 K8/V8
near-lossless in the tested fake-quant and static-scale experiments.

That result removes a major numerical objection to symmetric K8/V8. It does
not remove the kernel question. A deployable bias-aware symmetric path still
needs to show that its smaller K cache offsets:

- K operand preparation;
- bias reconstruction or score correction;
- any RoPE work moved into the read path; and
- the block-scaled or FP8-Q conversion machinery chosen by the kernel.

This is why the next experiment should compare ordinary K8/V8, pre-bias
K8/V8, K16/V8, and the available Blackwell transform/block-scaled paths under
one profiler matrix.

## Candidate fixes for symmetric FP8

The useful fixes fall into four groups.

### Schedule the existing transform better

- Compare `Full`, `KOnly`, and `SeparateKv` transform modes.
- Keep V preparation adjacent to the PV consumer rather than expanding it
  early.
- Double-buffer K tiles so producer warps prepare tile `n+1` while consumer
  warps run QK/softmax/PV for tile `n`.
- Avoid an extra shared-memory round trip when register or tensor-memory
  layouts allow it.
- Reduce register and shared-memory footprint enough to recover occupancy.

### Use a native narrow QK path

- Quantize Q for QK and use FP8 by FP8 MMA where quality and conversion cost
  permit it.
- Apply a scale in score space when the scale granularity is algebraically
  compatible with the reduction.
- On Blackwell, test MXFP8 QK with hardware block scaling against the plain
  E4M3 transform path.

### Make symmetric FP8 numerically admissible

- Quantize the pre-bias key residual and keep the fixed bias exact.
- Test both pre-RoPE reconstruction and post-RoPE residual plus analytical
  bias correction.
- Keep a fallback to K16/V8 for partial-RoPE or otherwise hostile key
  distributions that pre-bias does not repair.

### Change the cache layout once, not every read

- Write the cache directly in the matrix unit's preferred block-scaled and
  swizzled layout.
- Store scale factors in the exact hardware layout expected by the MMA path.
- Avoid a decode-time transcode whose cost is paid on every generated token.

## What to measure

At minimum:

- kernel latency and steady-state tile interval;
- DRAM bytes and effective bandwidth;
- L2 hit rate and warm/cold-cache behavior;
- conversion and tensor-core instruction counts;
- long-scoreboard, barrier, and dependency stalls;
- shared-memory traffic and bank conflicts;
- registers per thread, shared memory per block, and achieved occupancy;
- exact kernel/cubin and transform mode selected; and
- end-to-end ITL, throughput, and quality for ordinary and pre-bias K8/V8.

The complete public hypothesis, test matrix, and result fields are in
[`fp8-attention-tile-path.html`](../fp8-attention-tile-path.html).

## Source anchors

These notes were checked against the following sources on 2026-08-20:

- NVIDIA PTX ISA, `wgmma.mma_async` and `tcgen05.mma` sections:
  <https://docs.nvidia.com/cuda/parallel-thread-execution/>
- NVIDIA CUTLASS Blackwell block-scaled GEMM documentation:
  <https://docs.nvidia.com/cutlass/latest/media/docs/cpp/blackwell_functionality.html>
- NVIDIA CUTLASS `tcgen05` programming guide:
  <https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/mma_docs/tcgen05_programming.html>
- LLVM AMDGPU instruction definitions and tests for
  `v_mfma_f32_16x16x32_fp8_fp8`:
  <https://github.com/llvm/llvm-project/blob/ca75521459d26bbad33a754c50d50a9a1d709816/llvm/test/MC/AMDGPU/mai-gfx942.s>
- FlashInfer source at
  `76704c45003cabaa832d59896080f91dca23f74b`, including
  `Bf16QFp8KvTransformMode` and SM100 block-scaled FMHA.
- AITER source at
  `4fa508ef2935110ff99adf2743ea93807dbd9c67`, including the gfx942
  FP8 MFMA path and post-MFMA scale application.

Claims about a specific released wheel or opaque cubin still require a run on
the target hardware. Source support and runtime dispatch must be verified
separately.
