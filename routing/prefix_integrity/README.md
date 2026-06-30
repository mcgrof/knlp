# Prefix Integrity Analysis (PIA)

PIA is a preflight harness for any KV-cache compression, pruning, routing,
quantization, or offload algorithm. Before an algorithm is tested inside LMCache
or a distributed KV store, PIA asks a narrower question than a benchmark does:
after the algorithm touches a logical prefix cache entry, is that entry still a
reloadable, reusable, position-compatible, and semantically close cache object?

This is a cache-contract evaluation, not a leaderboard. The failure it is built
to catch is the one that is invisible if you only plot compression ratio against
model accuracy: an algorithm holds next-token accuracy on one prompt while
quietly making two requests that share a prefix disagree on what the prefix
object is. That breaks prefix sharing and KV offloading even though the accuracy
plot looks fine.

PIA uses **Cartridges as deterministic stand-ins for prefix-cached / offloaded
KV blocks**. A cartridge is a fixed precomputed KV cache with stable block
boundaries, so it is a cheap, reproducible prefix object you can run an algorithm
against without rerunning prefill. It does not require LMCache or distributed
serving.

## What it measures

The harness runs a candidate algorithm against a cartridge in one of three
modes — `selector` (chooses which blocks survive), `codec` (transforms KV
tensors but keeps every logical block), or both — and replays many suffix/query
requests that share the same `prefix_hash`. From that replay it computes:

- **Block survival**: prefix reuse efficiency (PRE), hot-block PRE, anchor and
  recent survival, contiguous-prefix survival, partial-block rate. A partial
  block is *not* intact: it may be usable by a custom kernel but is not reusable
  by ordinary block-hash prefix caching.
- **Determinism / cache-key safety**: how many distinct selected-block sets and
  artifact digests the same `prefix_hash` produces, across re-runs (determinism)
  and across queries (query-dependence). A query-dependent artifact keyed only
  by `prefix_hash` is the silent danger.
- **Storage geometry**: contiguous read ranges, read amplification, and the
  compression-ratio coefficient of variation across requests.
- **Semantic drift** (GPU): next-token KL divergence and top-1 / top-k
  agreement of the candidate against the full-cartridge baseline.

Hard gates turn these into a `PASS` / `WARN` / `FAIL` status, a `danger_score`,
and one of five plain-language classifications: `SAFE_FOR_PREFIX_OFFLOAD`,
`SAFE_ONLY_WITH_EXTENDED_CACHE_KEY`, `SAFE_ONLY_WITH_CUSTOM_CONNECTOR`,
`ROUTING_ONLY_NOT_PREFIX_CACHE_SAFE`, `DANGEROUS_FOR_PREFIX_SHARING`.

## CPU path (selector mode, no model)

The block-survival, determinism, and storage metrics need no model and no GPU:

```bash
python -m routing.prefix_integrity.cli validate \
  --cartridge /path/to/cartridge_dir \
  --algorithm query_aware \
  --queries queries.jsonl \
  --budgets 8,16,32 \
  --pins A1R2K13 \
  --out /path/to/report_dir
```

`--algorithm` is a built-in (`full`, `recency`, `random`, `anchor_recency`,
`a1r2k13`, `kri_prior`, `query_aware`) or a `package.module:factory` spec for a
custom adapter. Each K budget writes `result.json`, `result.md`, and
`block_survival.csv`.

## GPU path (semantic drift)

Semantic drift loads a real model and the cartridge tensors and measures how far
the candidate moves the next-token distribution from the full-cartridge
baseline. The query's position ids are pinned to the original prefix offset, so
drift is attributable to the algorithm rather than to re-indexing:

```bash
python -m routing.prefix_integrity.semantic_drift \
  --model Qwen/Qwen2.5-7B-Instruct \
  --cartridge /path/to/cartridge_dir \
  --queries queries.jsonl \
  --algorithm query_aware --mode selector --budget-k 16 \
  --out semantic.json
```

Fold the result back into the verdict with `--semantic-json semantic.json` on
the `validate` call.

## Writing an adapter

Implement `select_blocks(request, num_blocks, budget_k) -> list[int]` (selector)
and/or a `transform_cartridge` step (codec). Declare intent with class
attributes `policy` (`prefix_cache` | `routing` | `offload_codec`),
`cache_key_fields`, and `has_custom_restore_path`; the harness checks those
claims against observed behavior. The verdict is the collision between what the
algorithm claims and what it does.
