# Reproducing BPA KV Scaling Paper Experiments

This document explains the scaffolding used to reproduce the BPA KV scaling
paper results. The scripts under `scripts/paper/bpa_paper/` are orchestration
and packaging helpers. They do not claim to contain completed measurements by
themselves.

## Workflow

1. Run per-GPU smoke tests.
2. Launch full matrix collection with config-driven execution.
3. Fit scaling and context-linearity diagnostics on collected artifacts.
4. Package validated results into the `knlp-paper-kv-scaling/` tree.

## Directory Roles

- `configs/*.yaml`: GPU lane definitions and matrix settings.
- `run_smoke.py`: cheap preflight / mini-matrix validation.
- `run_matrix.py`: full matrix orchestration with rerun filters.
- `fit_scaling.py`: Hill/context fit scaffolding and derived artifacts.
- `manifest.py`: manifest schema and validation logic.
- `package_results.py`: export of paper-usable subset into clean results tree.

## Example Commands

Dry-run smoke test:

```bash
python3 scripts/paper/bpa_paper/run_smoke.py \
  --config scripts/paper/bpa_paper/configs/a100.yaml \
  --results-root /data/knlp-paper-kv-scaling \
  --dry-run
```

Full matrix orchestration:

```bash
python3 scripts/paper/bpa_paper/run_matrix.py \
  --config scripts/paper/bpa_paper/configs/b200.yaml \
  --results-root /data/knlp-paper-kv-scaling \
  --lane core
```

Rerun only failed points:

```bash
python3 scripts/paper/bpa_paper/run_matrix.py \
  --config scripts/paper/bpa_paper/configs/b200.yaml \
  --results-root /data/knlp-paper-kv-scaling \
  --lane core \
  --rerun-status failed,invalid
```

Fit derived metrics:

```bash
python3 scripts/paper/bpa_paper/fit_scaling.py \
  --manifest /data/knlp-paper-kv-scaling/manifests/run_manifest.json \
  --results-root /data/knlp-paper-kv-scaling \
  --dry-run
```

Package validated subset:

```bash
python3 scripts/paper/bpa_paper/package_results.py \
  --manifest /data/knlp-paper-kv-scaling/manifests/run_manifest.json \
  --source-root /data/knlp-private-results \
  --export-root /data/knlp-paper-kv-scaling \
  --dry-run
```

## Reproducibility Rules

- Never publish directly out of the private hodge-podge tree.
- Every exported file must be described in a manifest.
- Every paper-facing number should be traceable to raw artifacts and a script
  version.
- Treat long-context capacity evidence as a separate lane from the core
  scaling-law matrix.
