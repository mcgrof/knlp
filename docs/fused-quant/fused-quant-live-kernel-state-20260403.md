# Fused Quant Live Kernel State — 2026-04-03

## What this file is
`artifacts/fused-quant/fused_int4.py` is the exact live fused-quant Triton kernel file recovered from the H100 experiment lane.

Source:
- host: `103.207.149.108`
- original path: `/data/vllm/vllm/v1/attention/backends/fused_int4.py`

It was copied back into prune `knlp` because the latest fused-quant kernel advances were stranded on the remote H100 pod instead of living in the prune-owned research flow.

## Why it exists in `knlp`
Do not let the latest kernel state remain trapped on a transient experiment machine.
Bring the exact live file into prune so the current fused-quant state is inspectable, diffable, and preservable from the repo the user actually cares about.

## What it is expected to contain
The recovered file is expected to reflect the normative latest fused-quant state reached by the experiment lane:
- **S1** structural base
- **S2b** classifier-gated K-precision rule in the path logic
- **P1** promoted Triton decode defaults:
  - `DECODE_BLOCK_N = 32`
  - `DECODE_NUM_WARPS = 2`
  - `DECODE_NUM_STAGES = 3`

## What it is not
This file is not a claim that `knlp` is the runtime repo for serving this code directly.
It is a provenance and recovery artifact: the live kernel state brought home from the H100 lane so future work does not have to reverse-engineer a remote pod.

## Why the previous filename was wrong
The temporary name `fused_int4.py.remote-h100-20260403` described where the file came from, but it did not express what the file now means.
The canonical prune-side artifact should be named simply:
- `artifacts/fused-quant/fused_int4.py`

That name makes it clear that this is the current recovered fused-quant kernel artifact in the prune research flow.
