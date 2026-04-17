# FP8 attention kernels: hardware notes across Hopper, Blackwell, and CDNA 3

Technical notes on the Tensor Core / matrix-core instruction
constraints that shape how FP8 KV-cache attention kernels are
written on current GPU architectures, and on why those
constraints are the reason the asymmetric FP16-K / FP8-V pattern
(documented in this repository's [README](README.md)) exists as a
software workaround on NVIDIA Hopper.

## Table of contents

- [The short version](#the-short-version)
- [NVIDIA Hopper (sm_90)](#nvidia-hopper-sm_90)
- [AMD CDNA 3 (MI300X / MI325X)](#amd-cdna-3-mi300x--mi325x)
- [NVIDIA Blackwell (sm_100)](#nvidia-blackwell-sm_100)
- [Open questions and future work](#open-questions-and-future-work)

## The short version

On NVIDIA Hopper the Tensor Core FP8 MMA instruction requires
both input operands to share an FP8 format; there is no mixed
FP16 $\times$ FP8 MMA path.  Any attention kernel that stores K in
FP8 must therefore upcast it to FP16 (or downcast Q to FP8)
before the $q \cdot k$ dot product that feeds softmax, and the
resulting dequantization pass lies on the softmax serial
critical path.  That is why symmetric FP8 KV cache has a
measurable decode-latency penalty on Hopper.  Asymmetric
FP16-K / FP8-V keeps K at native precision and only quantizes V,
routing around the bottleneck in software.

AMD CDNA 3 removes the constraint in hardware: its WMMA matrix
instructions perform FP8 $\times$ FP8 matmul with per-tensor FP8
scales absorbed directly into the accumulator, so K never needs
a separate dequantization pass.  Symmetric FP8 KV cache does
not pay the Hopper penalty there.

NVIDIA Blackwell continues to require matching operand types for
the legacy FP8 MMA path, so a plain FP8 KV cache on Blackwell
inherits the Hopper bottleneck.  Blackwell does introduce
block-scaled microscaling formats (MXFP8 with a shared scale per
32-element block) whose hardware datapath can absorb scales in
the manner CDNA 3 WMMA already does.  Whether a
production-quality MXFP8 paged-attention kernel exists in the
open-source software stack at the time of writing is not known;
that is the gap a future evaluation would close.

## NVIDIA Hopper (sm_90)

The Hopper Tensor Core FP8 MMA instruction is
`mma.sync.aligned.m64n8k32.row.col.*.e4m3.e4m3.*` (and the
`e5m2` variants).  Both A and B operand tiles are required to
be the same FP8 format.  There is no mixed-format path that
accepts, for example, FP16 A and FP8 B.  The accumulator can be
FP16, BF16, or FP32, but the input pair must match.

The consequence for KV-cache attention is structural.  In a
standard decode attention kernel the query $q$ arrives in FP16
(or BF16) from the projection layer; the key cache $K$ is what
the operator wants to quantize to shrink HBM traffic.  If $K$ is
stored FP8, there are three ways to feed the $q \cdot k$ MMA:

1. Upcast the loaded $K$ tiles to FP16 in-register before the
   MMA and use the FP16 $\times$ FP16 path.  This is what
   FlashInfer's symmetric FP8 path does.  The upcast itself is
   trivial arithmetic, but it has to complete before softmax
   can see any score, and softmax cannot emit any attention
   weight until all scores are in.  So the K dequantization
   step serialises ahead of softmax — it is on the softmax
   critical path.  Every cycle spent on K dequant adds to
   per-token decode latency.

2. Downcast $q$ to FP8 before the MMA.  This avoids the K
   dequant cost but introduces a precision loss on the query
   vector that the attention computation is sensitive to,
   particularly for fragile-key models.  Not a serious
   production option.

3. Keep $K$ at FP16 and only quantize $V$.  The MMA for $q \cdot
   k$ is FP16 $\times$ FP16 with no dequantization step; the V
   side uses either a separate FP8 dequantization pass (which
   feeds an associative reduction and therefore overlaps with
   the accumulation pipeline rather than blocking it) or an
   FP8-to-FP16 convert inside the V load.  This is the
   asymmetric FP16-K / FP8-V path.

Option 3 is the software workaround for Hopper's dtype-matching
constraint.  The overall KV bytes-per-token is reduced (2\,B for
K, 1\,B for V, giving a 25% reduction and $1.33\times$ capacity)
and the softmax critical path is no longer gated on a
dequantization step.  The trade-off is that V gets all the
compression and K gets none, so total KV capacity is lower than
symmetric FP8 (which gets $2\times$).

## AMD CDNA 3 (MI300X / MI325X)

CDNA 3's matrix-core WMMA instructions include native
FP8 $\times$ FP8 matmul with hardware support for scale
absorption.  Concretely, the WMMA FP8 path accepts separate
per-tensor FP8 scales for A and B and folds them into the
accumulator during the matmul, so the output is effectively
computed in a higher-precision representation without a separate
dequantization kernel pass.

The implication for KV-cache attention is that the critical-path
constraint Hopper's ISA imposes does not exist.  A symmetric FP8
KV cache can be fed directly into the $q \cdot k$ matmul with
the K-side FP8 scale applied as part of the instruction, and the
same for the V-side reduction.  AMD's AITER backend uses this
pattern.  There is no separate K dequantization step sitting
between the K load and softmax, which means the bottleneck that
motivates the asymmetric FP16-K / FP8-V workaround on Hopper
does not apply.

The practical consequence is that the throughput gap between
symmetric FP8 and FP16 on MI300X is much smaller than on H100
(measured at 1.7–2.5% versus 3–8% respectively for the same
models), and the asymmetric-versus-symmetric ordering may be
different on AMD hardware for the same reason.

## NVIDIA Blackwell (sm_100)

Blackwell introduces a new Tensor Core family accessed through
`tcgen05.mma` instructions and a new `tensor memory` (TMEM)
region for staging matmul operands.  For the legacy FP8 path
(matching-format E4M3 or E5M2 inputs), the dtype-matching
requirement is preserved: the `tcgen05.mma` FP8 variant still
takes a single input dtype for the A/B pair.  A paged FP8 KV
cache fed into a straightforward port of the Hopper kernel
therefore inherits the Hopper critical-path constraint on
Blackwell — the K dequantization step is still required and
still lies ahead of softmax.

Blackwell does introduce block-scaled microscaling formats
(MXFP8, MXFP6, MXFP4; NVIDIA-specific NVFP4).  These store
elements in a narrow FP8 or FP4 representation with a shared
8-bit scale per 32-element block, and the hardware MMA
datapath has direct support for consuming the per-block scales
during the matmul.  In other words, on Blackwell with an MXFP8
KV cache, the hardware can perform the $q \cdot k$ matmul with
the per-block K scale folded into the accumulator, eliminating
the separate dequantization pass — the same architectural
pattern AMD WMMA already implements for per-tensor FP8 scales.

Whether a production-quality MXFP8 paged-attention kernel exists
in the open-source inference software stack at the time of
writing has not been verified here.  FlashInfer's current FP8
KV-cache path is plain E4M3, not MX-scaled.  Rewriting the paged
cache layout to MXFP8 and plumbing the MX MMA intrinsics through
decode and prefill is real engineering work, not a configuration
change.

## Open questions and future work

A few concrete items this line of work would benefit from
investigating:

1. **Blackwell MXFP8 KV cache.**  Does a plumbed MXFP8
   paged-attention kernel close the Hopper critical-path gap on
   Blackwell, matching what CDNA 3 WMMA does today?  This is a
   software engineering question with a likely-yes answer but no
   measured evidence at the time this note was written.

2. **MXFP8 vs asymmetric FP16-K / FP8-V on Blackwell.**  If an
   MXFP8 path is available, does it dominate the asymmetric
   Hopper-style workaround on B200, or do the fragile-key
   models that motivate the asymmetric path still require K
   precision above MXFP8?  The answer likely depends on whether
   the MXFP8 per-block scales carry enough K precision for
   Qwen-family models to survive; that is an empirical question.

3. **CDNA 4 and future AMD architectures.**  Does the
   scale-absorbed matmul pattern extend to FP4 / MXFP4-style
   formats, and at what precision does the K-fragility
   phenomenon re-emerge on AMD hardware?  The cross-architecture
   question is symmetric with the NVIDIA side.

4. **Three-way cross-vendor evaluation.**  A matched-batch,
   matched-sequence evaluation of (a) symmetric FP8 on H100, (b)
   asymmetric FP16-K / FP8-V on H100, (c) symmetric FP8 on
   MI300X via WMMA, (d) symmetric FP8 on B200, and (e) MXFP8 on
   B200 — all on the same set of models — would be the
   definitive answer to the question of how much of the
   asymmetric-on-Hopper win is an ISA-workaround artefact and
   how much is a structural K-precision finding.

## Caveats

The instruction-set statements in this note are reasoned from
the public Hopper and Blackwell PTX ISA documentation and from
AMD's CDNA 3 architecture references.  The behaviour of specific
open-source kernels (in particular, whether any FlashInfer build
already contains sm_100 MXFP8 code paths at the time of reading)
has not been verified by running against the hardware.  A
concrete next step for anyone acting on this document is to
check the current FlashInfer source for `sm_100` MX intrinsics
and to consult the CUDA 12.x PTX ISA chapter on
`tcgen05.mma.*.mx\*` before treating the Blackwell MXFP8 claim
as fact.
