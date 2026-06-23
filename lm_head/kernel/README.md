# Certified-decode LM-head kernel

A Triton realisation of the shadow-bound certified decode for the LM head. It
replaces the dense LM-head GEMV (a hidden state against a ~150k-row unembedding,
run for every generated token) with a two-stage decode that reads ~25% of the
bytes while certifying the *exact* dense argmax — lossless, not approximate.

- **Stage 1** (`certdecode_kernel.py`, Triton): a per-token int8 shadow-bound
  GEMV `U_v = aq_v·(scale·q) + ρ·delta_v + cst` against the low-rank shadow head
  `aq[V, r]` (r ≪ d, int8 < bf16), reduced to per-idblock slab maxima. Streams
  ~18% of the dense head's bytes.
- **Stage 2** (host loop + exact GEMV): open idblock slabs in descending bound,
  fetch their true logits, and stop once the best fetched logit dominates every
  unopened slab's bound. The dense argmax is then certified having touched only
  the opened rows (mean ~7% on Qwen2.5-7B).

Headline (Qwen2.5-7B, r=1280, int8 shadow, hidden-PCA basis, W7900): worst bound
violation −2.73 ≤ 0 (provably valid), argmax_match 1.0000 (lossless), byte ratio
25.3%, **1.66× over the dense GEMV** at batch-1. See
[docs/lm-head-routing.md](../../docs/lm-head-routing.md) for the full story.

`gen_artifact.py` builds the offline shadow-head artifact (the `*.pt` tensors are
regenerated, not committed). The `*_bench.py` / `cert_graph_*` files are the
single-token, batched, and CUDA-graph timing harnesses; `h6_oracle_scan.py`,
`predictor_baseline.py`, and `shadow_bound_replay.py` are the snapshot of the
routing/replay base the kernel was developed against (the maintained copies live
one level up in `lm_head/`).
