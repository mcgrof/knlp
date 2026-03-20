# BPA KV Scaling Paper Fortification Plan

This document defines the experiment plan needed to harden the BPA /
`paper-memory-decode` paper. It is intentionally broader than one missing GPU lane:
it covers the remaining weak spots in cross-GPU scaling-law validation,
long-context evidence, fit diagnostics, artifact provenance, reproducibility, and
future open-source packaging.

## Goals

The paper should be able to make five claims cleanly and with auditable evidence:

1. **Cross-GPU decode shape**: decode remains dominated by KV-memory traffic
   across W7900, H100, A100, and B200.
2. **Hardware-shifted saturation**: the qualitative form is stable while fit
   parameters (`S_max`, `B_half`, `gamma`) shift by GPU class.
3. **Linear context scaling**: latency is approximately affine in context length
   over the tested regime, with residuals and fit quality reported.
4. **Long-context consequences**: capacity-rich GPUs extend practical context
   while preserving the same decode shape and measurement discipline.
5. **Reproducibility**: raw artifacts, derived fits, manifests, figures, and
   system metadata can be regenerated into a clean paper-facing results tree.

## Weak Spots This Plan Addresses

- A100 currently exists as a bridge subset, not a first-class matched lane.
- B200 has useful paper-facing numbers but weaker provenance than it should.
- W7900 supports the story qualitatively, but needs a documented confirmation
  lane with the same orchestration and result schema as the NVIDIA lanes.
- Long-context results need to be treated as their own lane, not mixed casually
  into the core scaling-law section.
- Fit diagnostics and uncertainty need explicit artifacts, not just prose.
- Cross-GPU comparison needs a single artifact schema and manifest layer.
- The private `knlp-key-results` tree is too messy to point at directly; we need
  a clean `knlp-paper-memory-decode` export layout.

## Coverage Audit of the Current Framework

What is now covered by the current `knlp` scaffolding:

- a paper-facing results tree contract
- explicit GPU lanes for A100, B200, H100, and W7900
- smoke-test stages and stop/go criteria
- fit-output contracts for Hill fits, context linearity, and plateau summaries
- rerun rules, multi-instance scheduling rules, and paper-usable vs junk criteria
- public-subset packaging rules for a future `knlp-paper-memory-decode` export

What remains weak or only partially addressed in code:

- the underlying collectors are still mostly older lane-level scripts rather than fully normalized per-point runners
- `fit_scaling.py` and `package_results.py` are still contract-first scaffolds rather than complete production pipelines
- cross-GPU comparison tables and figures still need the real fitting/aggregation backend behind the manifest layer
- long-context collection still needs a dedicated runner path so capacity evidence is gathered separately from the core matrix

The first hardening step after this document is therefore to wire `run_matrix.py` to a real subprocess-backed point runner contract, keep configs honest about whether a lane is runnable or still plan-only, and then add GPU-specific adapters incrementally.

## Results Tree Layout

All paper-relevant outputs are staged into a clean tree rooted at:

`knlp-paper-memory-decode/`

Suggested layout:

```text
knlp-paper-memory-decode/
  manifests/
    run_manifest.json
    export_manifest.json
    public_subset_manifest.json
  reports/
    lane_summary_a100.md
    lane_summary_b200.md
    lane_summary_h100.md
    lane_summary_w7900.md
    cross_gpu_summary.md
  system/
    <gpu>/<lane>/system_info.json
    <gpu>/<lane>/env.txt
  raw/
    <gpu>/<lane>/<run_id>/latency_grid.jsonl
    <gpu>/<lane>/<run_id>/latency_grid.csv
    <gpu>/<lane>/<run_id>/stderr.log
    <gpu>/<lane>/<run_id>/stdout.log
  derived/
    <gpu>/<lane>/hill_fit_T4096.json
    <gpu>/<lane>/hill_fit_T16384.json
    <gpu>/<lane>/context_linearity.json
    <gpu>/<lane>/bootstrap_ci.json
    cross_gpu/comparison_table.json
  figures/
    <gpu>/<lane>/*.png
    cross_gpu/*.png
  logs/
    scheduler/
    smoke/
    reruns/
```

The export tree is what the paper references. `knlp-key-results` remains the
private source of truth during collection, but only data that survives manifest
validation and packaging should move into the paper-facing tree.

## Experiment Lanes

### Lane A: H100 reference lane

Purpose: retain the existing reference lane, but rerun with the same manifest,
smoke-test, and fit-diagnostic plumbing used by other GPUs.

Required matrix:
- model: `Qwen2.5-7B`
- batch: `1,2,4,8,16,32,64`
- context: `1024,2048,4096,8192,16384,32768`
- reference fit slices: `T=4096`, `T=16384`

Deliverables:
- full latency grid
- Hill fits with residuals
- context linearity fits per batch
- plateau bandwidth summary

### Lane B: A100 matched scaling lane

Purpose: promote A100 from bridge subset to first-class evidence or explicitly
prove that it cannot be promoted.

Minimum required matrix:
- same model and grid as H100
- same number of repetitions
- same fit workflow

Required outputs:
- exact raw grid with per-point mean/std
- Hill fits at `T=4096` and `T=16384`
- residual plots / residual JSON
- bootstrap confidence intervals for `S_max`, `B_half`, `gamma`

Stop/go:
- **go** if fits are stable under rerun, context fits stay linear, and manifest
  coverage is complete
- **stop / retest** if fit parameters swing heavily across reruns, if points are
  missing, or if smoke-test mismatches suggest script/kernel issues

### Lane C: B200 provenance lane

Purpose: preserve the strong B200 story but rebuild it with a clean artifact
trail and reproducible packaging.

Required matrix:
- same core grid as H100/A100
- optional extension to `B=128` if saturation is not reached

Required long-context extension:
- `B=1` at `T=65536,131072,262144,393216` where feasible
- optional `B=2` or `B=4` capacity slice if runtime permits

Deliverables:
- same raw grid schema as other GPUs
- extreme-context raw logs and system metadata
- a single manifest showing which numbers support core scaling versus capacity
  narrative only

Stop/go:
- **go** if raw provenance for every paper-facing B200 number is present in the
  export tree
- **stop / retest** if numbers only exist in prose, ad hoc notebooks, or manual
  logs

### Lane D: W7900 confirmation lane

Purpose: keep W7900 as the low-bandwidth contrast, but collect it under the same
paper-facing orchestration.

Required matrix:
- same core grid where feasible
- if runtime is expensive, allow a reduced matrix approved by manifest flag

Required outputs:
- full system metadata (ROCm / Triton / torch build)
- same latency, bandwidth, and fit outputs as NVIDIA lanes
- explicit note if no saturation is observed through maximum batch

Stop/go:
- **go** if the lane is complete enough to support qualitative contrast and
  context-linearity claim
- **stop / retest** if runtime bugs or backend issues invalidate enough points

### Lane E: Long-context lane

Purpose: separate capacity evidence from the core scaling-law evidence.

GPU focus:
- B200 required
- H100 optional
- W7900 optional for practical limit confirmation

Outputs:
- memory usage
- latency at very long contexts
- context-linearity residuals over the long-context regime
- one summary distinguishing **capacity consequence** from **core scaling-law**

## Smoke Tests Before Expensive Runs

Every GPU must pass smoke tests before a full matrix is scheduled.

### Smoke stage 0: environment capture
- verify device query
- record runtime versions
- verify output directory writable
- verify model path / tokenizer availability

### Smoke stage 1: kernel sanity
- run a tiny decode point (`B=1`, `T=1024`) on the selected model
- compare output shape and basic latency sanity against expected envelope
- for Triton-backed paths, fail closed if kernel import / launch fails

### Smoke stage 2: mini-matrix
- run `B in {1,8}` and `T in {1024,4096}`
- confirm logs, manifests, and raw metrics serialize correctly
- confirm rerun filtering works

Only after passing stages 0-2 should a lane be upgraded to full matrix.

## Repetitions and Measurement Discipline

Per grid point:
- warmup iterations: minimum 10
- measured iterations: minimum 30
- target measured iterations: 50 for fit-critical slices

Scheduling rules:
- randomize point order within a lane
- never silently drop outliers
- persist every measured point before reduction
- rerun only through manifest filtering, not ad hoc file deletion

## Fit Diagnostics and Uncertainty

For each lane:

### Hill fit
Fit throughput vs batch at `T=4096` and `T=16384`:

`S(B) = S_max * B^gamma / (B_half^gamma + B^gamma)`

Artifacts:
- parameter JSON
- residual JSON
- fit summary markdown
- optional bootstrap CI JSON

### Context linearity fit
For each fixed batch:
- fit latency vs context with affine model
- store slope, intercept, `R^2`, residual summary

### Plateau bandwidth summary
For each lane:
- store max observed effective bandwidth
- store per-slice plateau estimates
- explicitly distinguish practical plateau from hardware peak bandwidth

## Cross-GPU Comparison Artifacts

Required final artifacts:
- cross-GPU fit table (`S_max`, `B_half`, `gamma`, fit quality)
- cross-GPU context-linearity table
- cross-GPU bandwidth plateau table
- lane coverage table (complete / partial / failed / smoke-only)

These artifacts should be machine-readable first (`json`, `csv`) and then turned
into paper tables and figures.

## Partial Reruns and Bug-Fix Passes

The scripts are expected to be imperfect at first. The workflow must make quick
GPU-specific debugging cheap.

Rules:
- every run writes a manifest entry per point
- every point has status: `pending`, `ok`, `failed`, `invalid`, `skipped`
- reruns operate only on selected statuses or explicit `(B,T)` subsets
- smoke reruns do not contaminate full-matrix summaries

## Multiple-Instance GPU Scheduling

We want multiple instances where needed, but not chaos.

Rules:
- one manifest namespace per GPU instance
- one lane per GPU instance at a time unless explicitly split by context slice
- split full matrices by disjoint point subsets, never by overlapping writes
- package only after per-instance manifests reconcile cleanly

Recommended split strategy:
- A100 / H100: split by batch slices or context bands
- B200 long-context: isolate long-context lane from core lane
- W7900: prefer single-lane serial execution unless runtime is prohibitive

## What Counts as Paper-Usable vs Junk

### Paper-usable
- point recorded in manifest with `ok` status
- raw and reduced artifacts both present
- system metadata captured
- fit provenance reproducible from committed scripts
- no unresolved smoke-test mismatch

### Junk / exclude from paper
- quota errors
- aborted runs
- simulated-zero-error artifacts that are not valid for the claim being made
- points from script versions not captured in metadata
- hand-copied numbers without raw provenance

## Public-Subset Packaging Plan

`knlp-key-results` remains private. Public release should export only the
paper-relevant subset through `package_results.py`.

Export rules:
- copy validated raw / derived / figure / report files into
  `knlp-paper-memory-decode/`
- generate export manifest with checksums
- redact private hostnames and secrets from environment captures
- keep lane summaries honest about incomplete coverage

## Documentation and Open-Source Framing

The paper should point to:
- `knlp` for kernels, orchestration scripts, and reproducibility docs
- `knlp-paper-memory-decode` as the clean artifact layout

The repo docs should distinguish:
- smoke tests
- full paper runs
- packaging/export
- future public subset publication
