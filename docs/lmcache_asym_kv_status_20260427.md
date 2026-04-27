# LMCache asymmetric K16/V8 — POC status, 2026-04-27 (post-review)

This doc has been folded with the design-review feedback (see
`docs/lmcache_asym_kv_review_brief.md` for the brief that was sent
out and `docs/lmcache_asym_kv_review_response.md` for the response).
It supersedes the earlier "patches landed" framing because several of
those patches are now known to be dangerous or wrong-headed.

## What the LMCache asymmetric work actually delivers today

### Storage tier — production-ready

The `AsymK16V8Codec` and the split-tier (`SPLIT_K_CPU_V_NVME`) backend
are landed in real LMCache and demonstrably correct on H100:

- **Storage ratio 0.7500** median (24 cells, 3 model shapes ×
  2 sequence lengths × 4 seeds), matching the paper's claim exactly.
- **K bit-exact** across every seed × layer × shape.
- **V relative error 0.0217** median and max, well under FP8 e4m3
  bound 0.075.
- **Split-tier moves 2/3 of bytes off NVMe** (NVMe traffic ratio
  0.333 across 1MB / 8MB / 32MB / 64MB chunks).
- **Small-chunk write speedup of 2.48×** (28.5 → 70.8 MB/s at 1MB),
  read parity at small/medium and +21% at 64MB.
- The encode hot path's `bytes(tensor.untyped_storage())` was an
  ~860ms-per-MB Python-level bottleneck; replaced with a numpy
  `tobytes()` reinterpretation, ~10000× faster. Committed on the
  `asymmetric-kv-codec` branch.

This half is shippable. JSONs at
`prune:/data/knlp-key-results/lmcache_asym_perf_quality_20260426/`.

### vLLM runtime forward path — incomplete in the upstream forks

The `mcgrof/vllm:asymmetric-kv-plumbing` and
`mcgrof/flashinfer:asymmetric-kv-dtype` forks plumb config through
`CacheConfig` / `ModelConfig` / the attention selector, but the
runtime path is not finished. Crucially:

- FlashInfer's `BatchDecodeWithPagedKVCacheWrapper.plan()` was
  extended with `k_data_type` / `v_data_type` and separate
  `_cached_{k,v}_data_type` validation.
- FlashInfer's `BatchPrefillWithPagedKVCacheWrapper.plan()` was
  **not**.  Same for `BatchDCPPrefillWrapper.plan()`.
- vLLM's KV cache allocator allocates one `[2, num_blocks, ...]`
  tensor per layer at one dtype.  No two-tensor support.
- vLLM's FlashInfer backend forward path passes a single
  `kv_data_type=self.k_cache_dtype` to every `plan()` call (even
  though the backend's `__init__` does resolve `k_cache_dtype` and
  `v_cache_dtype` separately).
- The C-binding `reshape_and_cache_flash` op takes a single
  `kv_cache_dtype: str`, so writing K and V into the cache uses one
  quantization spec — there's no asymmetric writer.

## Design decision: two-tensor paged KV cache, end-to-end

The review confirmed this is the right plan and the other two
alternatives I sanity-checked are precision theater:

- "Convert K to FP8 inside the cache writer" loses K precision
  before attention even reads it from cache.
- "Use LMCache as the asymmetric layer over a symmetric vLLM cache"
  doesn't work because once vLLM's paged cache stores FP8 K, the
  BF16 information is gone and LMCache cannot reconstruct it.

The target invariant is:

```
In vLLM paged KV cache:
    K is model dtype (BF16/FP16).
    V is FP8 e4m3.
During FlashInfer attention:
    K is read as model dtype.
    V is read as FP8 e4m3 with V scale.
During LMCache spill/fill:
    K is serialized losslessly as FP16/BF16.
    V is serialized as FP8 + V scales.
```

## Patches that have to go away

These were applied during the prior pod run.  They are documented in
`scripts/lmcache_asym_vllm_patches.py` as **not applied** so the
script doesn't propagate them:

1. **FlashInfer `_check_cached_qkv_data_type` relaxation** — DANGEROUS.
   Plans a homogeneous FP8 KV kernel but feeds it BF16 K.  Silent
   corruption surface.  Replaced in Milestone 1 by extending the
   prefill plan signature with separate `k_data_type` / `v_data_type`
   and validating exactly.

2. **`reshape_and_cache_flash` tuple-to-V collapse** — DANGEROUS.
   Quantizes K through the FP8 path.  Replaced in Milestone 3 by a
   new `reshape_and_cache_flash_asym` op that stores K losslessly.

3. **flash_attn schedule cache_dtype tuple unwrap** — wrong backend.
   FA3 doesn't support asymmetric K/V for `head_dim=128`.  Selector
   must route asymmetric to FlashInfer specifically.

4. **flashinfer init premature tuple unwrap** — fix is in the forward
   path (Milestone 4), not at init.

5. **Selector reduction asym→V-dtype** — wrong intent.  Picks a
   FP8-aware backend but erases that K is not FP8.  Selector must
   see asymmetric and pick FlashInfer with hard fail-closed.

6. **`_LMCACHE_ASYM_FORCE_FLASHINFER` env-var hack** — process-global
   junk.  Replaced in Milestone 5 with explicit selector logic.

The patches that remain in the script are scouting only — they let
the engine boot far enough to expose downstream gaps:

- Tuple-handling helper at every `kv_cache_dtype.startswith()` /
  `==` site (20 files, 58 substitutions; final landing must replace
  these with explicit asymmetric-aware checks).
- `get_kv_cache_torch_dtype` accepts tuple (config-plumbing only).
- `get_fp8_dtype_for_flashattn` accepts tuple (config-plumbing only).
- LMCache `vllm_service_factory._ensure_engine()` before
  `LookupClientFactory.create_lookup_client()` (orthogonal to asym;
  enables bypass-lookup in single-process LLM() runs).

## Landing plan (six milestones)

Each milestone is a gate: do not start the next one until the
previous one has tests passing.

### Milestone 1: FlashInfer standalone proof — IN PROGRESS

Extend `BatchPrefillWithPagedKVCacheWrapper.plan()` and
`BatchDCPPrefillWrapper.plan()` to accept `k_data_type` and
`v_data_type` mirroring the decode wrapper.  Defaults preserve
back-compat (omit → use `kv_data_type`).  Cache them as
`_cached_{k,v}_data_type`.  Include them in the JIT URI / dispatch
key so BF16-K + FP8-V doesn't reuse a homogeneous FP8 KV kernel.
Replace the relaxed `_check_cached_qkv_data_type` hack with exact
q/k/v dtype validation against the cached values.

Then a standalone test (no vLLM) covers:

1. decode  Q=BF16, K=BF16, V=FP8_e4m3, tuple paged_kv_cache, vs
   torch reference within FP8 e4m3 noise band.
2. prefill same, with causal mask.
3. negative — K=FP8 against cached K=BF16 must raise.
4. negative — V=BF16 against cached V=FP8 must raise.

Code:
- `scripts/flashinfer_asym_prefill_extension.py` — applies the plan
  signature + body + run validation extension to the fork.
- `scripts/test_flashinfer_asym_kv.py` — the gate test.

This is the gate.  If it fails, vLLM patches are confetti.

### Milestone 2: vLLM logical asym cache allocation

Add `is_asymmetric_kv(spec)`, `cache_dtype_k_str(spec)`,
`cache_dtype_v_str(spec)`, `cache_dtype_k_torch(spec, model_dtype)`,
`cache_dtype_v_torch(spec)`, `cache_dtype_v_is_fp8(spec)` as the
canonical dtype helpers.

Add an `AttentionSpec` (or extend the existing one) with `k_dtype`,
`v_dtype`, `k_cache_dtype_str`, `v_cache_dtype_str`,
`page_size_bytes = k_page + v_page + scale_overhead`.

In the KV cache materialization path, allocate per-layer
`kv_cache = (k_cache, v_cache)` with separate dtypes (or one raw
allocation sliced into two contiguous typed views).  Block IDs are
shared.

Update `bind_kv_cache` and forward context so per-layer cache can be
either `Tensor` (symmetric) or `tuple[Tensor, Tensor]` (asymmetric).

Fail closed: if asymmetric KV is requested and backend is not
FlashInfer, raise `NotImplementedError`.

### Milestone 3: vLLM asymmetric cache writer

New op `reshape_and_cache_flash_asym(key, value, k_cache, v_cache,
slot_mapping, k_cache_dtype, v_cache_dtype, k_scale, v_scale)`.
Stores K without FP8 quant.  Stores V as FP8 e4m3 with `v_scale`.
Do NOT reuse the symmetric writer with `kv_cache_dtype=V`.

Update FlashInfer backend `do_kv_cache_update` to dispatch to the
asymmetric writer when `is_asymmetric_kv(self.kv_cache_dtype)` is
true.

### Milestone 4: FlashInfer backend forward

For asymmetric mode, pass `(k_cache, v_cache)` tuple to every
`prefill_wrapper.run()` and `decode_wrapper.run()`.  Pass
`k_data_type=self.k_cache_dtype, v_data_type=self.v_cache_dtype` to
every `plan()` call (normal prefill, decode, CUDA graph decode, DCP
prefill, cascade).  Never `.view(fp8)` the whole cache.

For paths not patched on first POC, raise `NotImplementedError` with
a specific message.

### Milestone 5: explicit selector + fail-closed

Replace the env-var force-FlashInfer hack with explicit logic in
`vllm/platforms/cuda.py`:

```python
if is_asymmetric_kv(cache_config.cache_dtype):
    if requested_backend not in (None, "FLASHINFER"):
        raise NotImplementedError("Asymmetric KV requires FlashInfer")
    return AttentionBackendEnum.FLASHINFER
```

Add a hard runtime assert in the FlashInfer backend ctor that
asymmetric was actually selected.

### Milestone 6: block movement + LMCache + end-to-end quality

Update `copy_blocks` and `swap_blocks` to handle tuple cache (apply
op to K and V tensors).  Update LMCache connector to ingest tuple
cache directly (K → codec K-half, V → codec V-half, scales →
codec scales).  On LMCache read, restore K to K-cache and V to
V-cache, not a synthetic symmetric tensor.

End-to-end quality test on Qwen2.5-7B:
- Baseline BF16
- Symmetric FP8 KV (expected: corruption)
- Asymmetric K16/V8

Compare logits / prefix loss — not just final strings.  Verify logs
report `K cache dtype = BF16/FP16`, `V cache dtype = FP8 e4m3`,
`FlashInfer plan k_data_type != v_data_type`.

## What we DO have right now

- Storage tier (B1+B2): production-ready, results archived
- B3 in-band evidence of motivation: symmetric FP8 gives **+39%
  throughput** on Qwen2.5-7B but **corrupts output** (`Linux pérdida
  de memoria management...`), reproducing Qwen K-fragility live in
  the same battery
- Patch script and standalone-gate scaffolding committed under
  `knlp/scripts/`
- This milestone plan, sized into honest gates

## Pointers

- `scripts/lmcache_asym_vllm_patches.py` — scouting-only patches,
  with dangerous ones removed and documented
- `scripts/flashinfer_asym_prefill_extension.py` — Milestone 1
  prefill plan() extension
- `scripts/test_flashinfer_asym_kv.py` — Milestone 1 gate
- `docs/lmcache_asym_kv_review_brief.md` — original brief sent out
- `docs/lmcache_asym_kv_status_20260427.md` — this doc
- LMCache codec branch: `mcgrof/lmcache:asymmetric-kv-codec`
  (committed on `prune:/data/lmcache`)
- vLLM fork: `mcgrof/vllm:asymmetric-kv-plumbing`
- FlashInfer fork: `mcgrof/flashinfer:asymmetric-kv-dtype`
- Storage results: `prune:/data/knlp-key-results/lmcache_asym_perf_quality_20260426/`
