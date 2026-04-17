# Fused INT4 v0.0.2 — CUDA graph decode support fix (2026-04-04)

## Provenance
- Based on: 
- Change: 
- Previous value: 

## Why
A bounded H100 latency retest showed the fused INT4 backend was restricted to PIECEWISE-only graph capture while FP16 used FULL + PIECEWISE graphs. Restoring uniform single-token decode graph support closed most of the decode-heavy latency gap.

## Verified result
- FP16 graph: ITL , TPS 
- Fused graph: ITL , TPS 
- Ratio: 
- Delta: 
- Prior fused graph ITL: 
- Gap closed: about 

## Artifact roots
- Pod run root: 
- Prune archive: 
