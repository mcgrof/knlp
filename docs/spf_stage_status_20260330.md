# SPF stage status — 2026-03-30

SPF code continues to live in the dedicated implementation tree:
- `/data/vllm-spf`
- branch: `spf-p2-20260329`

Durable results live in:
- `/data/knlp-key-results/spf/`

This note records the current experimental state from the perspective of `knlp`.

## Current completed stage

The latest scaled single-A100 run is:
- `/data/knlp-key-results/spf/spf-p2-a100-scale-20260330T043144Z/`

Its result is a **GO** for the current bounded SPF approach under richer
trace structure.

Per-workload verdicts:
- `shared_prefix`: IMPROVED
- `conversation_tree`: IMPROVED
- `mixed_session`: IMPROVED
- `batched_burst`: INCONCLUSIVE

The strongest interpretation is not that SPF is finished, but that richer
live-trace structure finally produced measurable signal beyond the earlier
cache-light online run.


## Interpretation guardrail

If the bar is **real scheduler-integrated online TTFT / latency improvement**,
this stage is still **not enough**.

The scaled single-A100 run is meaningful because it proved stronger signal than
the earlier cache-light online attempt, but it is still a bounded live-trace A/B
stage. It does **not** yet prove final serving-path value in the true scheduler
integration path.

Operationally, that means this result should be read as:
- stronger justification to continue,
- better than the earlier neutral result,
- but still short of a production or final benchmark claim.

## What the next stage is

Yes, the next stage has already been thought about.

The next stage should move from a single-GPU, live-trace A/B demonstration into
**true scheduler-integrated online evaluation** under stronger cache pressure.

That means:
- keep the exact baseline-vs-SPF A/B discipline,
- preserve per-phase verdicts rather than blending workloads together,
- promote the richer workload families that already showed signal,
- continue to treat `batched_burst` as the regression canary,
- wire the evaluation into the actual scheduler-side serving path rather than
  stopping at trace collection + replay alone.

## Immediate next-stage goals

1. Keep `shared_prefix`, `conversation_tree`, and `mixed_session` as the main
   signal phases.
2. Keep `batched_burst` as an explicit guardrail phase.
3. Preserve cache-hit / occupancy / eviction telemetry as first-class metrics.
4. Move the online evaluation closer to the true scheduler-integrated path.
5. Continue exporting all SPF runs durably into `knlp-key-results`.

## Canonical references

- Implementation/docs: `/data/vllm-spf/docs/design/spf_p2_integration.md`
- Results map: `/data/knlp-key-results/spf/README.md`
- Latest scale result: `/data/knlp-key-results/spf/spf-p2-a100-scale-20260330T043144Z/summary_note.md`
