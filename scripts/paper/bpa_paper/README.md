# BPA Paper Experiment Framework

This directory contains paper-facing orchestration scaffolding for the BPA KV
scaling paper. It is designed to make the private collection workflow cleaner,
cheaper to debug per GPU, and easier to export into a future public subset.

## What this is

- manifest schema and validation
- dry-run smoke-test planning
- matrix scheduling / rerun planning
- fit-output contract for scaling-law diagnostics
- packaging/export contract for `knlp-paper-memory-decode`

## What this is not

- completed paper measurements
- a claim that any GPU lane has already been rerun cleanly
- a replacement for the underlying Triton / decode kernels

## Expected flow

1. run `run_smoke.py`
2. run `run_matrix.py`
3. collect real GPU results with the wired experiment backend
4. run `fit_scaling.py`
5. run `package_results.py`

## Configs

The `configs/` directory contains lane definitions for:
- H100 reference lane
- A100 matched lane
- B200 provenance + long-context lane
- W7900 confirmation lane
