# SPF

SPF is a scheduler-side speculative-prefetch experiment for vLLM's KV cache.

## What SPF tried

Predict which KV blocks a request will need before it arrives. Touch
those blocks on the GPU block pool so they survive the next LRU eviction
round (retention mode), or initiate async promotion from a slower tier
(prefetch mode). The framework lives in `vllm/v1/core/spf/` and is opt-in
via `VLLM_SPF_ENABLED=1`.

## The bar SPF was held to

The survival gate at `benchmarks/spf_survival_ab.py` runs five arms
through a real vLLM engine under cache pressure:

```
LRU                            - vLLM default prefix cache
session-TTL + pinning          - the trivial baseline a sophisticated
                                  scorer must beat by a real margin
frequency admission            - LFU-style admission
SPF                            - the production scorer (ExpectedUtility)
reuse-distance oracle (Belady) - the unimplementable upper bound
```

A candidate scorer earns its place in the tree iff, across at least
three deterministic seeds:

- p95 TTFT improvement vs session-TTL is at least 5%, OR
- recomputed prefill tokens drop by at least 10% vs session-TTL,
- throughput regression vs LRU is under 1%,
- wasted promoted bytes are under 10% of total promoted,
- paired bootstrap CI on the headline metric excludes zero.

## The result

SPF does not clear the bar on Llama-3.2-3B under 2.3x cache pressure
(16-persona, round-robin, 32 sessions x 4 turns, gpu_memory_utilization
0.3, max_model_len 12288, 3 seeds). The five arms land within 1% of
each other on every measured axis. Even the Belady oracle stays inside
0.5% of session-TTL on round-robin pressure, which says the selector
ceiling on this workload class sits below the gate. The bridge does
issue hints (430+ touches per run); the policy does not produce a
useful signal in this regime.

| arm          | p95 TTFT (ms) | prefill recomputed | cache hit |
|--------------|--------------:|-------------------:|----------:|
| lru          |        772.90 |          1,064,960 |     0.00% |
| session_ttl  |        778.45 |          1,063,392 |     0.15% |
| frequency    |        779.53 |          1,063,392 |     0.15% |
| spf          |        775.93 |          1,063,392 |     0.15% |
| oracle       |        777.86 |          1,063,392 |     0.15% |

SPF vs session_ttl: p95 TTFT improvement +0.32% (gate >=5%), prefill
reduction 0.00% (gate >=10%).

## What survives

- The survival gate methodology and the
  [`spf_survival_ab.py`](https://github.com/mcgrof/vllm/blob/20260622-spf/benchmarks/spf_survival_ab.py)
  harness. Any future scheduler-side admission or retention policy can
  be measured against the same arms.
- The disposal pattern: when an experiment cannot clear its own bar,
  delete the runtime, preserve a single archive commit and the
  benchmark, document the failure as a structural finding.
- The substrate that other lines actually use: the `BlockManifest`
  schema and `CartridgeKRIProvider` were originally written for SPF
  consumption. The vLLM routing branch now consumes them as
  routing-prior loaders. See [KRI](kri.html) and
  [routing](routing.html).

## Branch

The SPF experiment lives intact at
[`github.com/mcgrof/vllm` branch `20260622-spf`](https://github.com/mcgrof/vllm/tree/20260622-spf).
One commit on top of `20260430-cartridges-upstream` contains the
controller, the five scorers (`ExpectedUtilityScorer`, `SessionAwareScorer`,
`SessionTTLScorer`, `FrequencyAdmissionScorer`,
`ReuseDistanceOracleScorer`), the scheduler bridge, the manifest
provider, the survival benchmark, and the tests.

## If you want to try

1. Check out the SPF branch.
2. Replace the scorer in `vllm/v1/core/spf/scorer.py` or add a new
   class and route it through `SPFController._build_scorer`.
3. Run `benchmarks/spf_survival_ab.py` against your scorer.
4. Clear the gate or document the next negative.

The bar exists so the line either delivers or stays parked.

## Related

- [KRI](kri.html) - the training-free routing-prior family that
  the routing branch actually serves.
- [routing](routing.html) - the serving substrate that hosts KRI
  variants as pluggable algorithms.
- [KRI-FT](kri_ft_visualization.html) - the PEFT vehicle that
  fine-tunes a model under a routing mask.
