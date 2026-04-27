# Asymmetric K16/V8 KV-cache in vLLM + LMCache — design review brief

This is a request for a second-opinion review. The goal is asymmetric
K/V quantization in vLLM serving: keep K at FP16/BF16 (Qwen and
similar models have fragile K activations that FP8 corrupts) and
quantize V to FP8 e4m3 (V tolerates it). Storage codec + split-tier
land cleanly. The vLLM runtime forward path does not, and I want a
review of where the design is wrong before I keep grinding.

## What's working (storage tier)

A pure-Python codec `AsymK16V8Codec` lives in
`lmcache/v1/kv_codec/asym_k16_v8.py`. It serializes a `(K_fp16,
V_fp8, V_scales_fp16)` tuple to bytes with a small header containing
shape, dtypes, and per-tensor scale scope. K is bit-exact across
serialize → deserialize. V relative error is bounded by FP8 e4m3
quantization noise (median 0.0217, max 0.0217 across 24 cells).
Storage size is exactly 0.75× of FP16 K+V at every model shape we
tested (Qwen2.5-7B, Llama-3.1-8B, Qwen3-27B-FA at T=1024 and 4096).

A `SplitTierStorageBackend` (also pure-Python on the put/get hot
path, but C++-equivalent on the codec) wraps the codec with a
placement policy:

- `ALL_NVME`: K, V, scales all written to NVMe (baseline)
- `ALL_CPU`: all to CPU pinned (debug)
- `SPLIT_K_CPU_V_NVME`: K to CPU pinned, V (FP8) + scales to NVMe

The split-tier layout puts the heavy fp16 K (which dominates bytes)
on host pinned memory and only the compressed FP8 V on disk. Real
disk benchmarks on H100 show **NVMe traffic is exactly 0.333 of
all-NVMe** (matches the K=2/3, V=1/3 byte ratio under FP16 K + FP8
V). Write throughput is **2.48× faster at 1MB chunks** (28.5 → 70.8
MB/s) and at parity for larger chunks (encode dominates beyond
~4MB). Reads are at parity for small/medium and +21% at 64MB.

This half is shippable. It's tested, it's correct, and it produces
the predicted 0.75 storage ratio across realistic model shapes.

## What's not working (vLLM runtime forward path)

The goal: have vLLM serve a model with `kv_cache_dtype=("auto",
"fp8_e4m3")` so K stays at the model's native dtype (BF16 for
Qwen2.5-7B) and V is FP8 e4m3 in the paged KV cache. LMCache then
absorbs the cache via the asymmetric storage codec.

A "vLLM asymmetric-kv-plumbing" fork was started by another author.
That fork:

- Plumbs `kv_cache_dtype` through `CacheConfig`, `ModelConfig`,
  attention selector, etc. as a tuple.
- Adds helpers `cache_dtype_k(spec)`, `cache_dtype_v(spec)`,
  `is_asymmetric_kv(spec)` in `vllm/config/cache.py`.
- Updates the per-layer `Attention.__init__` to extract the K dtype
  for `kv_cache_torch_dtype` (with a comment "K stays at high
  precision").

That's where the fork stops. There are at least 70 sites in the
vLLM tree that still treat `kv_cache_dtype` as a string and call
`.startswith("fp8")` or `== "fp8_e4m3"` on it, producing immediate
`AttributeError: 'tuple' object has no attribute 'startswith'`.

A companion "flashinfer asymmetric-kv-dtype" fork added one CUDA
commit (`default_prefill_params.cuh: asymmetric K/V plumbing`) but:

- The Python `BatchDecodeWithPagedKVCacheWrapper.plan()` signature
  was extended with `k_data_type` and `v_data_type` keyword args and
  separate `_cached_{k,v}_data_type` validation in `decode.py`.
- The Python `BatchPrefillWithPagedKVCacheWrapper.plan()` signature
  was *not* extended. It still accepts only `kv_data_type`.

In other words: the flashinfer fork covers the asymmetric **decode**
path (Python wrapper + CUDA kernel), but the asymmetric **prefill**
path is half-done — kernel exists, Python wrapper signature does
not. There is no version of `plan()` you can call from vLLM that
passes the prefill an asymmetric tuple.

## What we built on top of those forks

A patch script (`scripts/lmcache_asym_vllm_patches.py`) captures
every change we discovered while trying to land an end-to-end POC:

1. **Tuple-handling helper at every string-method site.** Inserts
   `from vllm.config.cache import cache_dtype_v as _cdv` and
   rewrites every `kv_cache_dtype.startswith("fp8")` /
   `== "fp8_e4m3"` to `_cdv(kv_cache_dtype).startswith("fp8")` etc.
   Touches 20 files, 58 substitutions across `vllm/v1/attention/`
   and `vllm/model_executor/layers/attention/`. Idempotent.
2. **C-binding wrappers** (`reshape_and_cache`,
   `reshape_and_cache_flash` in `_custom_ops.py`): inject a
   `if isinstance(kv_cache_dtype, tuple): kv_cache_dtype =
   kv_cache_dtype[1]` at the top so the C++ op (which expects
   `str`) gets the V-half.
3. **Boundary helpers**: `get_kv_cache_torch_dtype()` and
   `get_fp8_dtype_for_flashattn()` accept the tuple form by
   collapsing to V-half. The first is on the engine init path; the
   second is on the FA backend path.
4. **flash_attn schedule + flashinfer init**: both read
   `cache_config.cache_dtype` directly; we unwrap the tuple at each
   site.
5. **Selector reduction**: `vllm/v1/attention/selector.py` reduces
   the asym tuple to V-dtype (was K-dtype) so that an FP8-aware
   backend gets selected. We also write an env-var flag
   `_LMCACHE_ASYM_FORCE_FLASHINFER=1` whenever the selector sees a
   tuple, so the platform code knows to demote FA3.
6. **Force FlashInfer for asym** in `vllm/platforms/cuda.py`:
   when `selected_backend is None` *and* the env-var flag is set,
   force `selected_backend = AttentionBackendEnum.FLASHINFER`. FA3
   cannot serve asymmetric K/V for `head_dim=128` ("If V headdim is
   different from Q/K dim, we only support Q/K headdim in (128,
   192] and V headdim in (96, 128]"); we have to demote it.
7. **lmcache `vllm_service_factory`**: call `_ensure_engine()`
   before `LookupClientFactory.create_lookup_client()` so the
   bypass-lookup path works in single-process `LLM()` runs.
8. **vLLM flashinfer caller**: pass `k_data_type=self.k_cache_dtype,
   v_data_type=self.v_cache_dtype` to every `plan()` call (the fork
   already resolves k/v separately in
   `FlashInferMetadataBuilder.__init__`, but the call sites were
   never updated to forward them).

Plus a flashinfer-fork patch:
**relax `_check_cached_qkv_data_type`** so that K's BF16 is accepted
when the planned `kv_data_type` is FP8 (asymmetric design intent).
Without this the warmup tensors get rejected before the cache is
even allocated.

After applying all of the above, on a fresh H100 SECURE pod with
torch 2.10.0 + flashinfer-fork + vllm-fork built from source +
lmcache asymmetric-kv-codec branch:

- Engine boots end-to-end.
- LMCache config reports `Using asymmetric KV cache: K=auto,
  V=fp8_e4m3. Keys stay at higher precision to protect models with
  fragile key activations (e.g. Qwen family).`
- `SplitTierStorageBackend` is created.
- `LMCacheBypassLookupClient` initialises (no ZMQ broker needed).
- Both worker-side and scheduler-side LMCache instances log success.
- FlashInfer is selected as the attention backend (FA3 demoted).
- Model warmup hits `BatchPrefillWithPagedKVCacheWrapper.plan() got
  an unexpected keyword argument 'k_data_type'`.

## What's missing (the actual question)

I see three gaps. I'd like a review on whether these are right and
which order to attack them in.

### Gap 1: prefill wrapper signature

`BatchPrefillWithPagedKVCacheWrapper.plan()` in the fork still has
only `kv_data_type`. The CUDA prefill kernel can already accept
asymmetric (per the commit message). Adding `k_data_type` and
`v_data_type` to the signature, defaulting both to `kv_data_type`
when omitted, and threading them through to the JIT URI / kernel
arg pack is what's needed. The decode wrapper already shows the
pattern.

**Question for review:** is there a reason the fork only did
decode? Is there a hidden assumption that prefill always
re-quantizes V on the way in (so the cache writer still sees a
homogeneous dtype during prefill) and only decode reads back
asymmetric? If so, the design is correct and we just need vLLM's
**writer path** to convert V → FP8 before storing (and the
prefill kernel only ever sees K=BF16, V=BF16 ephemerals during
attention computation, not the cache). That changes the strategy
substantially.

### Gap 2: KV cache allocation

vLLM's `KVCacheManager` allocates one `(2, num_blocks, block_size,
num_kv_heads, head_size)` tensor per layer at one dtype. For
asymmetric K/V we need either:

(a) Two separate tensors per layer — `(num_blocks, block_size,
    num_kv_heads, head_size)` at K-dtype and another at V-dtype.
(b) One uint8 tensor sized to `K_bytes + V_bytes` and a per-element
    accessor that interprets the K-half as BF16 and the V-half as
    FP8.
(c) Keep one combined tensor at the V-dtype (FP8) and convert K to
    FP8 only at write time, accepting the K-fragility hit. (This is
    what symmetric FP8 already does and it corrupts Qwen output, so
    this is a non-starter for the actual research goal.)

I lean toward (a) because it's the cleanest split and the storage
codec already mirrors that layout. But (a) requires touching the
allocator, the per-layer cache binding (`bind_kv_cache`), the
attention forward-pass tensor unpack (`reshape_and_cache_*`), and
the FlashInfer wrapper input plumbing.

**Question for review:** what's the right split? Is there an
existing vLLM hook for "two-dtype paged KV" that I'm missing? Some
of the MLA paths look two-tensor-ish but I haven't traced the full
contract.

### Gap 3: FlashInfer prefill/decode mode contract

Even if (1) and (2) are solved, the FlashInfer fork only validates
asymmetric in `decode.py`'s `_check_cached_qkv_data_type` call site.
It doesn't say what the prefill kernel actually does with K=BF16 and
V=FP8 paged inputs. Reading the CUDA commit, it changed the
`AttentionParams` struct to carry K and V dtypes separately and the
kernel reads them per tensor — so this should work, but I haven't
proven it on a real workload.

**Question for review:** has anyone exercised this path? Is there a
flashinfer test or example that calls the asymmetric decode path
end-to-end with real K=BF16, V=FP8 paged tensors and produces
sensible attention output?

## Specific design alternatives I want a sanity-check on

1. **"Convert K to FP8 inside the cache-write path, keep K at BF16
   in flight."** That is, vLLM's forward pass uses K at BF16, but at
   the moment of paged KV write, K gets quantized to FP8 only for
   storage; then on cache-read for decode, K is dequant back to
   BF16. This avoids needing two-dtype paged tensors — the cache is
   FP8-symmetric on the wire but K's quant noise is contained to
   read paths only. Does this preserve Qwen quality? Probably not
   (the K-fragility shows up at attention time, and attention reads
   from the cache). So this is probably wrong, but I want a second
   opinion.

2. **"Use LMCache as the asymmetric layer; keep vLLM symmetric."**
   That is, vLLM allocates a symmetric FP8 cache and accepts the
   Qwen K-fragility cost in flight. LMCache absorbs it as
   (K_BF16, V_FP8) at *spill* time only — when blocks are evicted
   to LMCache, LMCache reconstructs K from the FP8 cache and quant
   noise it absorbed. But you can't reconstruct K's BF16 precision
   from an FP8 cache; the information is gone. So this is also
   wrong unless LMCache intercepts cache writes too, which makes
   LMCache the authoritative store and vLLM's paged cache a dirty
   write-through buffer. Big architectural change.

3. **"Two-tensor paged KV cache, end-to-end."** What I think is
   right. Costly to land but it's the only path that preserves K
   precision in flight *and* compresses V on the wire and on disk.

## Stack and reproduction

- Pod type: H100 SXM 80GB SECURE on RunPod ($2.99/hr)
- PyTorch 2.10.0+cu128, CUDA 12.8, FlashInfer 0.6.7, vLLM
  asymmetric-kv-plumbing fork (commit 5b4cc97)
- Patch script: `knlp/scripts/lmcache_asym_vllm_patches.py`
  (idempotent, applies all 8 patch families above)
- Storage results JSON + logs:
  `prune:/data/knlp-key-results/lmcache_asym_perf_quality_20260426/`
- Status doc: `knlp/docs/lmcache_asym_kv_status_20260427.md`

## What I'm asking for

1. Validation or refutation of the "two-tensor paged KV" plan. If
   it's right, what's the cleanest way to land it in vLLM 0.x
   without a multi-month patch series?
2. A read on whether the FlashInfer fork's prefill-wrapper-signature
   gap is just an oversight, or whether the design intent was that
   prefill never sees asymmetric (V is converted at write time in
   the cache writer, not in attention).
3. Any reference to existing vLLM code that handles two-dtype paged
   KV (MLA-like paths, mixed-precision experiments, anything) that
   we could pattern-match against.
4. Spot-check on whether anything in the patch list is wrong-headed
   — particularly the env-var force-FlashInfer hack (gap 5) and the
   selector reduction-on-V-dtype (gap 6).

If the right answer is "this design is fundamentally wrong, do X
instead," that is the most useful thing you can say.
