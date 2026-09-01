# FP8 KV attention: hardware paths and tile preparation

Track four layers of the implementation separately:

1. the cache format stored in HBM;
2. the on-chip work that prepares a tile for the matrix unit;
3. the matrix instruction that consumes the operands; and
4. the online-softmax and value-accumulation schedule around it.

Use this separation when comparing K8/V8 with K16/V8. FP8 storage cuts HBM
traffic. Operand preparation can still lengthen the tile interval through
conversion instructions, shared-memory traffic, register pressure,
synchronization, or reduced occupancy.

## Operating model

Apply these rules when reading a fused FP8 attention kernel:

- Keep the compressed cache in HBM. Reconstruct only the tile consumed by the
  active attention block.
- Advance online softmax tile by tile.
- Complete K operand preparation before QK for that tile.
- Complete V operand preparation before PV for that tile.
- Measure overlap instead of assuming it.
- Record the exact kernel, cubin, transform mode, cache layout, and scale
  layout selected at runtime.

Use the following dependency graph:

```text
K load -> K operand preparation -> QK -> online-softmax update
V load -> V operand preparation -> PV -> output accumulation
```

Producer and consumer warps can interleave these stages across adjacent tiles.
Measure the steady-state initiation interval to capture the resulting overlap.

## Measure the K8/V8 break-even

Hold V at FP8 and compare the K path:

```text
K8/V8   = 1 byte K + 1 byte V = 2 bytes per element pair
K16/V8  = 2 byte K + 1 byte V = 3 bytes per element pair
```

Model the latency delta as:

```text
T(K8/V8) - T(K16/V8)
  ~= unhidden K8 preparation
   + K8 pipeline and occupancy penalty
   - extra K16 HBM time
```

Use the stage model:

```text
II_K8  ~= max(load K8,  prepare K8 + QK, softmax update, PV)
II_K16 ~= max(load K16,              QK, softmax update, PV)
```

Identify the limiting stage with transform-mode, dtype, cache-state, and
profiler ablations. The public decision matrix is in the
[`bias-aware KV deployment policy`](../bias-aware-kv-quantization.html).

## NVIDIA Hopper

Hopper exposes separate BF16/FP16 and FP8 matrix-operation families. Ordinary
LLM decode presents BF16 or FP16 Q and a scaled E4M3 K cache. Hopper does not
provide this exact operation:

```text
BF16 Q x scaled E4M3 K -> FP32 scores
```

Choose one of three implementation paths:

1. load K as FP8, prepare the tile on chip, and use a BF16 or FP16 QK path;
2. quantize Q and use FP8 QK; or
3. keep K in BF16 or FP16 and quantize only V.

Path 1 preserves the K8 HBM saving and pays on-chip preparation. Account for:

- FP8 conversion or transform instructions;
- shared-memory or register repacking;
- scale broadcast and application;
- warp or warp-group synchronization;
- register pressure; and
- occupancy loss.

Path 2 removes the mixed-operand problem and adds Q quantization cost plus a
new numerical gate. Measure Q conversion once per decode step and compare it
against the repeated K-tile preparation avoided across the context.

Path 3 produces K16/V8. It removes K8 preparation and reads one extra K byte
per element. Use it as the no-K-transform control.

## AMD CDNA 3

CDNA 3 (`gfx942`) exposes native FP8-by-FP8 MFMA into FP32, including:

```text
v_mfma_f32_16x16x32_fp8_fp8 dst, src_a, src_b, acc
```

The instruction accepts FP8 A and B operands plus an FP32 accumulator. It has
no scale operands.

Move a scale outside the dot product when its granularity permits:

```text
Q @ (sK * Kq)^T = sK * (Q @ Kq^T)
```

Current AITER FP8 source executes FP8 MFMA and applies the K scale after the
MFMA result. Treat that as software scale placement around the matrix
instruction. Profile the selected AITER kernel and record whether Q is FP8,
how scales are grouped, and where score scaling occurs.

Use native FP8-by-FP8 MFMA when Q is FP8. Handle BF16-Q/plain-FP8-K with an
explicit conversion, Q quantization, or alternate matrix path. Compare actual
kernel schedules across vendors instead of comparing the presence of an FP8
instruction.

## NVIDIA Blackwell

Blackwell adds `tcgen05.mma`, tensor memory, narrow operand formats, and
block-scaled matrix operations.

### Measure the BF16-Q/FP8-KV transform path

FlashInfer and TensorRT-LLM generation code expose BF16-Q/FP8-KV transform
modes:

```text
Full
KOnly
SeparateKv
```

Use these modes as a controlled scheduling ablation. For every mode, record:

- selected kernel and cubin;
- K and V staging locations;
- conversion instruction counts;
- tensor-memory and shared-memory traffic;
- registers per thread;
- achieved occupancy; and
- steady-state tile interval.

Keep the FP8 cache in HBM and transform only active tiles on chip.

### Measure block-scaled QK

Blackwell block-scaled MMA consumes narrow operands and per-block scale
factors in hardware. For MXFP8, use the conceptual operation:

```text
D = C + (SFA * A) @ (SFB * B)
```

Prepare Q and K in the required block-scaled layouts. Store scale tensors in
the exact layout consumed by the matrix instruction. Compare this path against
plain E4M3 K transformation and K16/V8.

FlashInfer contains an SM100 fused block-scaled FMHA implementation for
MXFP8/NVFP4-style QK. Validate the paged-cache layout, supported head shapes,
scale policy, numerical behavior, and serving dispatch for the target model.

## Account for V preparation

V preparation consumes instructions and storage bandwidth before PV. Hide it
with scheduling when the kernel permits:

- prepare V next to its PV consumer;
- let producer warps prepare tile `n+1` while consumer warps accumulate tile
  `n`;
- fold compatible scales into PV arithmetic; and
- keep reconstructed V in registers, shared memory, or tensor memory only for
  its active tile.

Measure V independently with K16/V16 versus K16/V8. Compare the V-only delta
against the full K8/V8 delta. Report the hidden fraction for each shape.

## Treat numerical safety and kernel speed as separate gates

The [FP8 KV-cache failure atlas](../fp8-kv-failure-atlas.html) identifies a
Qwen failure mechanism: a large key-projection bias captures the scale and
crushes the token-varying residual. Quantize the pre-bias residual and keep the
fixed bias exact to make symmetric K8/V8 quality-admissible for the tested
Qwen2.5-7B configurations.

Benchmark the resulting cache representation as a separate kernel path.
Include:

- pre-bias residual quantization;
- exact bias storage;
- bias reconstruction or score-space correction;
- pre-RoPE and post-RoPE residual layouts;
- scale storage and broadcast; and
- long-context autoregressive quality.

Keep K16/V8 as the fallback for partial-RoPE and other key distributions that
fail the symmetric quality gate.

## Improve symmetric FP8

### Pipeline the existing transform

- Compare `Full`, `KOnly`, and `SeparateKv`.
- Double-buffer K tiles.
- Specialize producer and consumer warps.
- Place V preparation next to PV.
- Remove redundant shared-memory or tensor-memory movements.
- Reduce registers and shared memory until occupancy recovers.

### Use a native narrow QK path

- Quantize Q once per decode step.
- Use FP8-by-FP8 MMA where the quality gate passes.
- Apply score-space scales when their granularity matches the reduction.
- Test Blackwell MXFP8 against plain E4M3 transformation.

### Store the preferred cache layout once

- Write K, V, and scales directly in the matrix unit's block-scaled and
  swizzled layout.
- Align pages and scale blocks with the decode tile shape.
- Avoid per-token decode transcoding.

### Dispatch by operating point

Choose among ordinary K8/V8, pre-bias K8/V8, block-scaled K8/V8, and K16/V8
using:

- model quality;
- batch size;
- context length;
- cache residency;
- head dimension and GQA ratio;
- kernel latency; and
- admitted serving concurrency.

## Collect the profiler evidence

Record at minimum:

- kernel latency and steady-state tile interval;
- DRAM bytes and effective bandwidth;
- L2 hit rate and warm/cold-cache behavior;
- conversion and matrix instruction counts;
- long-scoreboard, barrier, and dependency stalls;
- shared-memory and tensor-memory traffic;
- bank conflicts;
- registers per thread;
- shared memory per block;
- achieved occupancy;
- exact JIT URI, kernel, cubin, and transform mode; and
- end-to-end ITL, throughput, cache capacity, and quality.

Use the four dtype controls:

```text
K16/V16
K8/V16
K16/V8
K8/V8
```

Run L2-warm and HBM-cold variants. Sweep batch, context, head dimension, GQA
ratio, and page size. Preserve raw Nsight Compute, Nsight Systems, rocprof, and
benchmark outputs.

## Source anchors

Use these source anchors for instruction and kernel verification:

- NVIDIA PTX ISA, `wgmma.mma_async` and `tcgen05.mma`:
  <https://docs.nvidia.com/cuda/parallel-thread-execution/>
- NVIDIA CUTLASS Blackwell block-scaled GEMM documentation:
  <https://docs.nvidia.com/cutlass/latest/media/docs/cpp/blackwell_functionality.html>
- NVIDIA CUTLASS `tcgen05` programming guide:
  <https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/mma_docs/tcgen05_programming.html>
- LLVM AMDGPU instruction tests for
  `v_mfma_f32_16x16x32_fp8_fp8`:
  <https://github.com/llvm/llvm-project/blob/ca75521459d26bbad33a754c50d50a9a1d709816/llvm/test/MC/AMDGPU/mai-gfx942.s>
- FlashInfer commit `76704c45003cabaa832d59896080f91dca23f74b`, including
  `Bf16QFp8KvTransformMode` and SM100 block-scaled FMHA.
- AITER commit `4fa508ef2935110ff99adf2743ea93807dbd9c67`, including the
  gfx942 FP8 MFMA path and post-MFMA scale application.

Verify the released wheel, JIT output, and runtime dispatch on the target GPU.
