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

Use the unified public helper unless you are debugging internals.

1. run `run_dataset.py --stage smoke`
2. run `run_dataset.py --stage matrix-exec` (or `matrix-plan` first)
3. run `run_dataset.py --stage fit`
4. run `run_dataset.py --stage package`

The lower-level scripts still exist for debugging:
- `run_smoke.py`
- `run_matrix.py`
- `fit_scaling.py`
- `package_results.py`

## Configs

The `configs/` directory contains lane definitions for:
- H100 reference lane
- A100 matched lane
- B200 provenance + long-context lane
- W7900 confirmation lane

## Public commands

Pick a results root that you control. Do not hard-code a private artifact tree.

```bash
export RESULTS_ROOT=$PWD/results/bpa-multi-gpu

# one GPU, smoke only
python scripts/paper/bpa_paper/run_dataset.py \
  --results-root "$RESULTS_ROOT" \
  --gpu a100 \
  --stage smoke

# one GPU, plan the full matrix
python scripts/paper/bpa_paper/run_dataset.py \
  --results-root "$RESULTS_ROOT" \
  --gpu a100 \
  --stage matrix-plan

# one GPU, execute the configured point runner
python scripts/paper/bpa_paper/run_dataset.py \
  --results-root "$RESULTS_ROOT" \
  --gpu a100 \
  --stage matrix-exec

# all public lanes, dry-run the whole workflow
python scripts/paper/bpa_paper/run_dataset.py \
  --results-root "$RESULTS_ROOT" \
  --gpu all \
  --stage full-dry-run

# derive fit artifacts from the collected manifest
python scripts/paper/bpa_paper/run_dataset.py \
  --results-root "$RESULTS_ROOT" \
  --gpu all \
  --stage fit

# package a paper-facing export tree
python scripts/paper/bpa_paper/run_dataset.py \
  --results-root "$RESULTS_ROOT" \
  --gpu all \
  --stage package
```

If you need to restrict the matrix, pass through filters such as:

```bash
python scripts/paper/bpa_paper/run_dataset.py \
  --results-root "$RESULTS_ROOT" \
  --gpu h100 \
  --stage matrix-exec \
  --only-batches 1,2,4,8 \
  --only-contexts 1024,4096,16384 \
  --limit 6
```
