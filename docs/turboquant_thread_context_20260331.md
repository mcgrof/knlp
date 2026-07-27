# TurboQuant thread context — 2026-03-31

Use this file list to bring the TurboQuant Discord follow-up thread up to speed.
Target thread:
- `<#1488646053207998464>`

## Read first
1. `turboquant_public_note_20260331.md`
2. internal working notes (logs and rerun plans), kept privately

## Durable results
- `turboquant-abcd-h100-20260331T195241Z/` (in the private results archive)
- `turboquant-public-informed-h100-20260331T203533Z/` (in the private results archive)

## Repo-local results copies
- `results/turboquant-abcd-h100-20260331T195241Z/`
- `results/turboquant-public-informed-h100-20260331T203533Z/`

## Repo-local code copies
- `tools/turboquant_eval/turboquant_abcd_ablation_h100.py`
- `tools/turboquant_eval/turboquant_public_informed_rerun_h100.py`

## Main conclusions to remember
- D = current fused quantization baseline
- At `head_dim=128`, TurboQuant did **not** beat the current fused baseline
- Public-informed rerun conclusion: TurboQuant does **not** currently justify a long-context-only follow-up line at `d=128`
- Revisit only if:
  - larger head dims (`>=256`)
  - more aggressive quantization regime
  - better fused rotation implementation
  - or more pathological / non-uniform KV distributions

## Source repos / hosts
- Main repo with copied context/results: `prune:/data/knlp`
- H100 execution pod used for TurboQuant work: RunPod H100 pod `53sfxfhjbswt8f`

## Best starting summary request for the thread
Ask it for:
- corrected A/B/C/D framing recap
- why D still won
- what public evidence changed in the fairer rerun
- whether any future TurboQuant line is still worth pursuing
