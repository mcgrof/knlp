# Path 3 implementation plan — asymmetric K16/V8 serving stack

This is the authoritative work plan for completing the asymmetric
K16/V8 vLLM + FlashInfer serving path, returned by the design
review.  Treat it as the canonical sequence of commits; everything
else in `docs/` is supporting context.

## Working bases

- FlashInfer: `mcgrof/flashinfer:asymmetric-kv-dtype` (HEAD
  `414b1875`).  Bit-identical between GitHub and
  `prune:/data/flashinfer-asym-work/flashinfer`.  Earlier framing
  that prune was "9 commits ahead" was wrong — the 9 decode-kernel
  commits (`9b4598aa..2b6b2eb5`) are reachable as ancestors of
  `414b1875` on both remotes.
- vLLM: `mcgrof/vllm:asymmetric-kv-plumbing` (HEAD `5b4cc9780`).
  Functionally identical to `prune:/data/vllm`.
- LMCache: `mcgrof/lmcache:asymmetric-kv-codec` (storage-side, ok
  as is, not part of this work)

Workflow: edit on `monster`, `git format-patch` deltas, `scp` to a
runpod GPU pod for build + measure, `git push monster→prune` when
stable.  No public GitHub push until the stack serves end-to-end.

## Target invariant

```
vLLM paged KV cache:
    K cache: BF16/FP16 native model dtype
    V cache: FP8 e4m3

FlashInfer prefill/decode:
    Q: BF16/FP16
    K: BF16/FP16
    V: FP8 e4m3
    O: BF16/FP16

LMCache:
    K serialized losslessly
    V serialized as FP8 + scales

No reinterpret-cast.  No pretending FP8 bytes are BF16.  The
compiler was right to reject that and gets to keep its dignity.
```

## Step 0 — allocator ownership sanity check

`_reshape_kv_cache` currently uses `.contiguous()` on the K and V
slices, which can allocate fresh storage and break the contract
that the per-layer cache is a view into the block-manager-owned
raw tensor.

**First action**: add an assertion to the M2 fork-side test:

```python
raw_start = raw_tensor.data_ptr()
raw_end = raw_start + raw_tensor.untyped_storage().nbytes()
for side, cache in [("K", k_cache), ("V", v_cache)]:
    ptr = cache.data_ptr()
    assert raw_start <= ptr < raw_end, (
        f"{side} cache is not a view into raw KV allocation"
    )
```

If the assertion fails, replace `.contiguous()` with `as_strided`
typed views over the raw uint8 buffer (helper sketched in the
review under `_typed_paged_view`).

**Acceptance gate**: M2 test must prove K/V tuple has correct
dtypes/shapes, K and V do not alias, and both are views into the
block-manager-owned raw allocation (or the allocator explicitly
accounts for separate K/V allocations).

## Step 1 — vLLM `FlashInferImpl.forward()` tuple plumbing

In `vllm/v1/attention/backends/flashinfer.py`:

1. Add helpers:
   - `_is_asym_paged_kv_cache(kv_cache)` — tuple check
   - `_derive_4d_stride_order_from_5d(stride_order_5d)` — drop
     original dim 1, renumber dims > 1 down by one
   - `_prepare_flashinfer_paged_kv_cache(kv_cache)` — returns 5-D
     permute for symmetric, `(k_permute, v_permute)` tuple for asym

2. Replace `kv_cache_permute = kv_cache.permute(*stride_order)`
   with `paged_kv_cache = _prepare_flashinfer_paged_kv_cache(kv_cache)`
   and route `paged_kv_cache` to every call that previously took
   `kv_cache_permute`.

3. Update the `forward()` signature type annotation to
   `kv_cache: torch.Tensor | tuple[torch.Tensor, torch.Tensor]`
   and the docstring to describe both shapes.

4. Do NOT normalize tuples in `bind_kv_cache`.  Keep that boring.

5. Plan() callsites already pass `k_data_type` and `v_data_type`
   in the prune fork; verify they reach the JIT URI, fail closed
   if not.

6. Fail closed with `NotImplementedError` for asym in cascade,
   DCP, and TRTLLM paths until each is explicitly supported.

7. Tests:
   - helper shape/path test (symmetric vs tuple)
   - fake `prefill_wrapper.run` monkey-patched to assert tuple
     received, not 5-D tensor
   - vLLM warmup with asym kv_cache_dtype must reach FlashInfer
     prefill compile (not crash on `.permute()` first)

After this step, expected failure mode is the FlashInfer compile
error at `prefill.cuh:1758`, not `AttributeError: 'tuple' object
has no attribute 'permute'`.

## Step 2 — vLLM writer GPU contract test

Run the M3 writer contract test on GPU (currently CPU-only).
Inputs: random BF16 key, random BF16 value, asymmetric tuple
cache, slot_mapping with valid slots and -1 skips.

Assertions:
- K cache written slots equal input K bit-exact
- unwritten K slots untouched
- V dequantize-via-`v_scale` within FP8 e4m3 noise
- unwritten V slots untouched
- `slot_mapping == -1` writes nothing
- no aliasing between K and V regions

If the current two-call aliasing trick (line 1699+) fails this
test, implement a dedicated `reshape_and_cache_flash_asym` op:
- K stored without FP8 quantization
- V stored as FP8 e4m3 with `v_scale`

Do NOT quantize K, ever, in asym mode.

## Step 3 — FlashInfer prefill kernel staged refactor

Seven commits, each individually compilable, each preserving
homogeneous legacy behavior.

### FI-1 — semantic alias split (no-op)

```cpp
template <..., typename DTypeKV, ...>
struct PrefillKernelTraits {
    using DTypeK = DTypeKV;
    using DTypeV = DTypeKV;
    ...
};
```

Rename obvious K- and V-specific sites to `DTypeK` / `DTypeV`
while still homogeneous.  Compiles, passes legacy tests.

Touched sites: K pointer types, K shared memory, K page loads,
K fragments, K RoPE, K scale handling — all use `DTypeK`.  V
pointer types, V shared memory, V page loads, V fragments,
V vec_cast/dequant, V scale handling — all use `DTypeV`.

### FI-2 — split shared-memory byte sizing

`kv_tile_elems * sizeof(DTypeKV)` becomes side-specific:
`k_tile_elems * sizeof(DTypeK)` and `v_tile_elems * sizeof(DTypeV)`.
Raw shared memory pointers respect K-then-V layout with
explicit alignment.  Still homogeneous, still compiles.

### FI-3 — independent template parameters

Change template signature from `DTypeKV` to `DTypeK, DTypeV`.
Remove or poison the old `DTypeKV` alias.  At this point compile
the BF16-K/FP8-V specialization and let the compiler identify
remaining unified-type sites.

### FI-4 — split Params and locals

Params struct already split on prune (`414b187`).  Kernel locals:

```cpp
const DTypeK* k = params.k;
const DTypeV* v = params.v;   // line 1758 — the famous corpse
```

Do NOT just patch line 1758.  That's the first body, not the
murderer.  Address every site the compiler flags.

### FI-5 — dtype traits for FP8 behavior

Replace `sizeof(T) == 1` with `is_fp8_type<T>` traits.

```cpp
template <typename T> struct is_fp8_type : std::false_type {};
template <> struct is_fp8_type<__nv_fp8_e4m3> : std::true_type {};
template <> struct is_fp8_type<__nv_fp8_e5m2> : std::true_type {};
```

Then per-side gating: `if constexpr (is_fp8_type<DTypeK>)` and
`if constexpr (is_fp8_type<DTypeV>)`.

For K16/V8: `is_fp8_type<DTypeK> == false`,
`is_fp8_type<DTypeV> == true`.

### FI-6 — side-specific static_asserts

Replace `static_assert(sizeof(DTypeKV) == 2)` and similar guards
with K-side or V-side specific assertions.  E.g., RoPE on K
expects `sizeof(DTypeK) == 2`.  V dequant path expects
`is_fp8_type<DTypeV> || sizeof(DTypeV) == 2`.

### FI-7 — support matrix

```cpp
template <...> struct is_supported_prefill_dtype_combo
    : std::false_type {};
template <> struct is_supported_prefill_dtype_combo<
    nv_bfloat16, nv_bfloat16, __nv_fp8_e4m3, nv_bfloat16
> : std::true_type {};
template <> struct is_supported_prefill_dtype_combo<
    half, half, __nv_fp8_e4m3, half
> : std::true_type {};
// + existing legacy combos

static_assert(
    is_supported_prefill_dtype_combo<DTypeQ, DTypeK, DTypeV, DTypeO>
        ::value,
    "Unsupported FlashInfer prefill Q/K/V/O dtype combination"
);
```

INT4/INT8 variants are explicitly OUT OF SCOPE for this pass —
that's the `asymmetric-kv-int4v` follow-on branch.

## Step 4 — FlashInfer Python/JIT glue

Verify on prune `asymmetric-kv-dtype`:

1. `BatchPrefillWithPagedKVCacheWrapper.plan()` accepts
   `k_data_type` and `v_data_type`.  Backward-compat defaults
   route through `kv_data_type` then `q_data_type`.

2. JIT URI includes `dtype_q`, `dtype_k`, `dtype_v`, `dtype_o`.
   No collapse to `dtype_kv` for asym.  If any URI generator
   collapses, the refactor isn't done.

3. Runtime validation for tuple paged cache:
   `k_cache.dtype == planned_k_data_type` and
   `v_cache.dtype == planned_v_data_type`.  No relaxation.

4. DCP/cascade wrappers: pass through K/V dtypes or fail closed
   with `NotImplementedError`.  No silent fallback.

## Step 5 — acceptance gates (in order)

### Gate A: FlashInfer standalone decode

- Asym decode passes, rel err around prior 0.0254.
- Homogeneous FP16/FP8 decode regressions still pass.
- Catches accidental decode regressions from shared JIT/template
  changes.

### Gate B: FlashInfer standalone prefill

- JIT module name includes `dtype_q/dtype_k/dtype_v/dtype_o`
- compile succeeds for BF16-K/FP8-V
- median rel err vs BF16 reference <= 0.10
- no silent fallback to symmetric FP8
- multiple shapes: head_dim=128, page_size=16, num_kv_heads in
  {4,8}, q/kv ratio in {1,4,7,8}, prefill lengths in {1,16,128},
  NHD layout minimum

### Gate C: vLLM tuple forward smoke

vLLM `LLM()` warmup with asym config logs:

```
kv_cache allocated as tuple
K dtype = bf16
V dtype = fp8_e4m3
FlashInfer selected
prefill plan q=bf16 k=bf16 v=e4m3
decode plan q=bf16 k=bf16 v=e4m3
```

No `'tuple' object has no attribute 'permute'`.  No symmetric FP8
module selected for asym mode.

### Gate D: GPU writer contract

- K bit-exact at written slots
- V within FP8 tolerance
- `slot_mapping == -1` skipped
- no K/V corruption
- runs on CUDA, not just CPU

### Gate E: tiny end-to-end quality sanity

Deterministic small prompt set, compare:

- FP16 vLLM
- symmetric FP8 vLLM (expect Qwen corruption)
- asym K16/V8 vLLM (expect FP16-like output)
- HF `FP8VLayer` simulation (cross-check)

Use token-level / logit-level comparison first.  Final text is
too coarse.

## Step 6 — measurement plan (in layers)

### Phase 1: FlashInfer standalone

- decode BF16-K/FP8-V
- prefill BF16-K/FP8-V
- legacy FP16
- legacy FP8 symmetric

Collect: latency, rel err vs BF16 reference, compiled module
names, dtype logs.

### Phase 2: vLLM without LMCache

Same harness as paper baselines.  Configs: FP16, sym FP8, asym
K16/V8.  Models: Qwen2.5-7B-Instruct minimum, plus Llama-3.1-8B
or Mistral-7B.

Metrics:
- WikiText-2 PPL T=2048
- GSM8K 8-shot n=200 minimum
- H100 throughput, same batch/context as paper

Acceptance bands (not exact equality with HF `FP8VLayer`):
- Qwen asym vLLM PPL close to FP16 and HF asym, within ~0.05–0.30
- GSM8K n=200 same qualitative recovery as FP16
- symmetric FP8 still collapses on fragile Qwen

If vLLM asym differs substantially from HF asym, debug in order:
writer K exactness → V scale/FP8 cast policy → prefill kernel vs
torch ref → decode kernel vs torch ref → mask/causal → page
table/slot mapping.

### Phase 3: vLLM + LMCache

Only after Phase 2 is clean.  Cache miss / spill / hit-restore /
no-LMCache compare.  Measure 0.75x serialized ratio, 0.333x NVMe
ratio, cache-hit latency, tail latency, pinned-memory pressure.

## Hard constraints

- No reinterpret_cast on `params.v` to silence the compile error
- No `sizeof(T) == 1` for FP8 dispatch
- No symmetric path silently selected for asym mode
- No `.contiguous()` copy that breaks block-manager ownership
- INT4/INT8 are out of scope for this pass

## Source-of-truth pointers

- This plan: `knlp/docs/lmcache_asym_kv_path3_plan.md`
- Background brief: `knlp/docs/lmcache_asym_kv_path3_brief.md`
- Status (post-review): `knlp/docs/lmcache_asym_kv_status_20260427.md`
- Upstream PR draft: `knlp/docs/flashinfer_asym_prefill_upstream_issue.md`
