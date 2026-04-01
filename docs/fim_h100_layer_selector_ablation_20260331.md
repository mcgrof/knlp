# H100 RA ablation: layer selector only, head selector fixed — 2026-03-31

## Purpose

This ablation isolates **layer selection** while holding **head selection fixed**.

We discovered algorithm drift across GPT-2 / llama150m / llama1b / 32b:
- trusted smaller-model path: layer selection by per-layer FIM traces, head selection by eigenvalues
- later 1B configs drifted head selection to `inbound_mass_var`
- 32B was further confounded by a blunt uniform middle-band headcount screen

This ablation is intended to answer a narrow question on an H100 lane:

> If we keep head selection fixed to eigenvalues, is an attention-derived layer selector better than FIM trace for choosing candidate RA layers?

## Arms

### Arm A — FIM-trace layers + eigenvalue heads
- layer_selector: `fim_trace`
- head_selector: `max_eigenvalue`

### Arm B — attention-stat layers + eigenvalue heads
- layer_selector: `attn_prob_sq_mean`
- head_selector: `max_eigenvalue`

## Exact algorithm

### Shared head selection
For both arms:
- choose candidate layers first
- within those candidate layers, rank heads by per-head max eigenvalue
- keep top-N heads globally across the candidate layers

### Arm A layer selection
- use `per_layer_traces`
- sort descending
- skip the single highest layer
- keep the next top layers above threshold

### Arm B layer selection
- compute an attention-derived layer score from the causal attention probabilities
- current implementation uses:
  - `attn_prob_sq_mean`
  - defined as the mean of squared attention probabilities over batch, heads, query positions, and key positions
- sort descending
- skip the single highest layer
- keep the next top layers above threshold

## Why this is narrow

This does **not** introduce a new head selector.
It tests only whether the layer selector should stay FIM-trace-based or move to an attention-derived functional signal.

## Recommended initial target
- llama1b matched harness on H100
- preserve the current matched-harness eval path as much as possible

## Expected outputs
- generated selection JSON for each layer selector
- downstream matched eval summary for each arm
- explicit bundle metadata that records:
  - `layer_selector`
  - `head_selector=max_eigenvalue`
  - candidate layers
  - selected heads
