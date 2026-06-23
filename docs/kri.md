# KRI

KRI is a family of training-free algorithms for picking which KV blocks each
request should attend to. Compute a per-block score offline, write it as a
`.pt` prior, load it at serve time through the routing substrate, take the
top K. No model changes; no custom inference.

## What KRI is

KRI stands for KV Routing Index. Each variant answers the same question
with a different signal: *how much does each KV block contribute to the
quality of the output, conditional on the rest of the cache?* Variants split
on whether the score comes from attention weights, logit shifts under
leave-one-block-out, gradients, query-aware geometry, or spectral structure
on the block graph.

Bake the prior once per cartridge. Ship the `.pt` file alongside the
cartridge. Load it through the `CartridgeKRIProvider` on the
[routing branch](routing.html); the connector handles top-K selection and
sparse inject.

## Variants

| variant            | per-block score                                                  | cost                        | notes                                                                       |
|--------------------|------------------------------------------------------------------|-----------------------------|-----------------------------------------------------------------------------|
| **KRI-D-kv-sum**   | Sum across layers of L2 between full hidden and leave-block-out  | offline; one prefill / block | Production leader. Robust on multi-needle long-context.                     |
| KRI-D              | KL(logits_full \|\| logits_leave-block-out)                       | offline; one prefill / block | Direct logit-level signal; weaker than kv-sum.                              |
| KRI-A              | Mean per-layer prefill attention probability mass per block      | one prefill                 | Cheapest baseline; recency bias.                                            |
| KRI-G              | L2 of gradient of LM-head loss w.r.t. block-mean K                | one backward                | Loss-curvature signal; numerically sensitive on fp16.                       |
| KRI-Q              | Q dot K agreement against block-mean K                            | online                      | Query-aware; flat on untrained block-mean K.                                |
| KRI-T (TriAttention) | Pre-RoPE Q/K trig factorization                                 | online                      | Query-aware, position-aware, zero bake cost.                                |
| kmeans_keys        | K-means clusters over block-mean K; rank by cluster size          | offline                     | Captures structural redundancy.                                             |
| attention_prefill  | Layer-mean prefill attention (post-softmax)                       | one prefill                 | Recency-skewed.                                                             |
| eigenvector centrality | Dominant eigenvector of block-level attention graph           | one prefill + power iteration | Addresses KRI-A recency bias and KRI-D marginal-MI redundancy in one shot. |

## The production stack

KRI-D-kv-sum for selection; structural pinning A1R2K13 (anchor=1, recent=2,
KRI-mid=13) at K=16; xa25 cross-attention refinement as the post-load repair
overlay. LongHealth K=16: **26.5% vs random middle 21.5%, 5σ at n=200**.

## How a prior moves through the system

Offline (once per cartridge):

```
cartridge.pt + model
    │
    ▼
KRI baker  (one of: bake_kri_d_kv_prior.py, bake_kri_t_prior.py, ...)
    │
    ▼
prior.pt = {
    block_affinities: Tensor[num_layers, num_kv_heads, num_blocks],
    # or kmeans_blocks_perK / kmeans_blocks legacy formats
}
```

At serve time (per request, on the [routing branch](routing.html)):

```
CartridgeConnector.__init__:
    routing.priors[cart_id] = path/to/prior.pt
    CartridgeKRIProvider.register_prior(prefix_hash=cart_id, ...)

CartridgeConnector.update_state_after_alloc(request, ...):
    manifest = CartridgeKRIProvider.get_manifest(
        prefix_hash=cart_id, query_hash=None, K=routing.K,
    )
    _request_selected_block_ids[req_id] = set(manifest.block_indices)

CartridgeConnector.get_block_skip_list(req_id, num_logical_blocks):
    # complement of selection; the scheduler turns this into null_block.
```

## Scope

- **Where KRI helps**: multi-block needles, long context. LongHealth K=16 +
  xa25, 5σ over random middle.
- **Where KRI is neutral**: single-needle NIAH-copy. Position-matched random
  blocks match KRI; placement carries the value, content selection adds
  nothing on this workload.
- **Where KRI breaks**: high-redundancy multi-needle. KRI-D's
  leave-one-block-out signal underweights blocks whose information appears
  in multiple other blocks. Use KRI-D-kv-sum or eigenvector variants.

## Adding a new KRI variant

1. Write a baker on the knlp side that produces a `.pt` with one of the
   supported formats.
2. Implement `BlockManifestProvider.get_manifest(prefix_hash, query_hash, K)`
   on the vLLM side, or reuse `CartridgeKRIProvider` if your output format
   matches.
3. Drop the prior under `extra_config["routing"]["priors"]` keyed by
   cartridge id.
4. Measure against the production stack (KRI-D-kv-sum + A1R2K13 + xa25) on
   LongHealth K=16.

## Related

- [routing](routing.html) — the serving substrate KRI plugs into.
- [KRI-FT](kri_ft_visualization.html) — the PEFT vehicle that fine-tunes a
  model under a routing mask. Composes with KRI in a 2×2 factorial.
- [SPF](spf.html) — separate scheduler-side prefetch experiment, parked.
