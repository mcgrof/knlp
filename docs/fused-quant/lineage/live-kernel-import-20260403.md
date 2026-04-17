# Fused Quant Live Kernel Import — 2026-04-03

## Purpose
Bring the live fused-quant Triton kernel state back from the H100 execution lane into prune `knlp` so the latest work is no longer stranded on the remote pod filesystem.

## Source of import
Remote H100 lane file:
- host: `103.207.149.108`
- path: `/data/vllm/vllm/v1/attention/backends/fused_int4.py`

Imported into prune `knlp` as:
- `/data/knlp/artifacts/fused-quant/fused_int4.py.remote-h100-20260403`

## Why this import exists
Do not leave the latest fused-quant kernel state trapped on the H100 pod.
Bring the exact live file into prune so the normative latest state can be inspected, diffed, documented, and committed from the repo the user actually cares about.
