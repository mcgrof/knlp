# BatchPrefillWithPagedKVCacheWrapper.plan needs separate k_data_type/v_data_type for tuple paged KV cache

This is the text intended for submission to flashinfer-ai/flashinfer.
Two parts: the issue itself, then the PR plan.  Both written so a
maintainer can read them in order without backstory and decide.

---

## Issue (boring version, suitable for upstream)

### Problem

Batch prefill JIT can now emit an asymmetric q/k/v dtype
specialization, but `prefill.cuh` still assumes
`DTypeK == DTypeV` internally.

`BatchPrefillWithPagedKVCacheWrapper.run()` accepts a tuple paged KV
cache `(k_cache, v_cache)`, but `BatchPrefillWithPagedKVCacheWrapper.plan()`
only accepts and specializes on a single `kv_data_type`.  Even after
threading `dtype_k` / `dtype_v` through `plan()` and into the JIT
URI / dispatch (which the codegen + `_kv_uri_fragment` already
support), the kernel template fails to compile because the kernel
body still uses unified `DTypeKV` for V- and K-side pointers.

This blocks mixed-dtype tuple KV cache, specifically:

- Q: BF16
- K cache: BF16
- V cache: FP8 e4m3
- `paged_kv_cache`: `(k_cache, v_cache)`

This layout is needed by vLLM asymmetric KV cache support, where K
remains at the model's native precision (BF16/FP16) and only V is
stored as FP8.  The motivation is models like Qwen2.5 whose key
activations are fragile under FP8 quantization (we observe semantic
output corruption — Spanish words appearing in English completions —
on Qwen2.5-7B with symmetric FP8 KV cache, but BF16-K + FP8-V
preserves output quality).

### Why this matters for prefill specifically

In a vLLM-style serving stack, prefill writes K/V into the paged
cache and the FlashInfer prefill kernel reads back from that paged
cache to compute attention.  Prefill therefore needs the same
asymmetric cache dtype support as decode.

The decode wrapper already supports this in this fork —
`BatchDecodeWithPagedKVCacheWrapper.plan()` accepts `k_data_type`
and `v_data_type` separately, validates them at `run()`, and the
asymmetric decode kernel produces output within FP8 noise of a
BF16 reference (standalone test: rel err 0.0254 against a torch
reference using BF16 K and dequantized V).

The prefill wrapper does not.

### What goes wrong without this

If `plan()` only receives `kv_data_type=fp8_e4m3` (because that is the
only public knob), the JIT dispatch specializes a symmetric FP8 KV
module.  With tuple cache `(BF16 K, FP8 V)`, that module reads the
BF16 K tensor as FP8.  The bytes are reinterpreted, the kernel
computes nonsense, and you get garbage output.

In our standalone test (configuration: 4 sequences, kv_lens
[40, 64, 32, 56], 32 prefill query tokens per seq, head_dim 128, 8
KV heads, page_size 16, causal mask), this produced relative error
**3.49** vs a BF16 reference, far above the expected FP8-V-only
tolerance band of ~0.10.

### Evidence

vLLM-side asymmetric scaffolding is already in place:

- allocation split verified by direct `_reshape_kv_cache` test —
  `(k_cache, v_cache)` 4-D NHD tuple, BF16 K + FP8 V dtypes, total
  bytes exactly 0.75× of symmetric BF16 K+V.  Test commit:
  `mcgrof/vllm:asymmetric-kv-plumbing b9f38e9f3`.
- cache write split implemented in fork commit `84d8633a4`
  ("flashinfer: split cache write for asymmetric V FP8").
- vLLM FlashInfer plan() call sites already pass asymmetric args in
  fork commit `cb568db61` ("flashinfer: pass k_data_type/v_data_type
  to plan()").  Today these args are rejected with `TypeError` by the
  prefill wrapper because the signature does not accept them.

### Reproduction

Standalone test using only FlashInfer (no vLLM, no model):
`https://github.com/mcgrof/knlp/blob/main/scripts/test_flashinfer_asym_kv.py`

Decode sub-test passes (rel err 0.0254).  Prefill sub-test produces
rel err 3.49, demonstrating the JIT specialization issue.

---

## PR plan

The change is in `flashinfer/prefill.py` and is the symmetric
companion of the work already done in `flashinfer/decode.py` for the
asymmetric decode wrapper.

### Step 1 — extend `plan()` signature

For `BatchPrefillWithPagedKVCacheWrapper.plan()` and
`BatchDCPPrefillWrapper.plan()`, add two optional kwargs immediately
after `kv_data_type`:

```python
k_data_type: Optional[Union[str, torch.dtype]] = None
v_data_type: Optional[Union[str, torch.dtype]] = None
```

### Step 2 — backward-compatible defaulting

Inside `plan()`, after the existing `kv_data_type` canonicalization:

```python
if k_data_type is None:
    k_data_type = kv_data_type
k_data_type = canonicalize_torch_dtype(k_data_type)

if v_data_type is None:
    v_data_type = kv_data_type
v_data_type = canonicalize_torch_dtype(v_data_type)
```

Existing callers that pass only `kv_data_type` (the homogeneous case)
must remain bit-equivalent.  The asymmetric path is opt-in by
explicitly setting one of the two new kwargs.

### Step 3 — cache dtypes separately

Alongside `self._cached_kv_data_type`, also store:

```python
self._cached_k_data_type = k_data_type
self._cached_v_data_type = v_data_type
```

These mirror the fields the decode wrapper already has.

### Step 4 — JIT dispatch + kernel template

Two pieces here.  The dispatch part is small.  The kernel template
part is the actual blocker, and is the bigger lift.

**Step 4a — dispatch keying** (small).  In `prefill.py` the dispatch tuple
currently looks like:

```python
get_module_args = (
    q_data_type,
    kv_data_type,
    o_data_type,
    paged_kv_indptr.dtype,
    head_dim_qk,
    head_dim_vo,
    PosEncodingMode[pos_encoding_mode].value,
    window_left >= 0,
    logits_soft_cap > 0,
    use_fp16_qk_reduction,
)
```

Replace `kv_data_type` with separate `k_data_type, v_data_type`:

```python
get_module_args = (
    q_data_type,
    k_data_type,
    v_data_type,
    o_data_type,
    paged_kv_indptr.dtype,
    head_dim_qk,
    head_dim_vo,
    ...
)
```

This must propagate into the JIT module URI and the kernel cache key.
A BF16-K + FP8-V plan must not reuse a symmetric FP8 KV module.

When both `k_data_type == v_data_type == kv_data_type`, the dispatch
tuple is functionally equivalent to the old key — homogeneous callers
pick the same module they always did.  When they differ, a separate
asymmetric variant is selected/compiled.

**Step 4b — kernel template refactor** (the blocker).

The current kernel template in `include/flashinfer/attention/prefill.cuh`
uses a single `DTypeKV` type parameter ~80 times: K shared memory, V
shared memory, K-frag and V-frag types, FP8 dequant `vec_cast`, MMA
path, RoPE.  When the JIT keys produce a `dtype_k_bf16_dtype_v_e4m3`
module, codegen emits a `Params` struct with `DTypeK = bf16, DTypeV =
e4m3`, but the kernel still instantiates with one `DTypeKV`.  Result:

```
prefill.cuh:1758: error: a value of type "RaggedParams::DTypeV *"
    (aka "__nv_fp8_e4m3 *") cannot be used to initialize an entity
    of type "DTypeKV *" (aka "__nv_bfloat16 *")
      DTypeKV* v = params.v;
```

So the kernel template has to be refactored to accept and use
`DTypeK` and `DTypeV` separately:

- Add `DTypeK_, DTypeV_` template params to `KernelTraits` /
  `KTraits` alongside (or replacing) `DTypeKV_`.
- K-related sites (`k_smem`, `k_frag`, `UPCAST_STRIDE_K`,
  K-RoPE) use `DTypeK`.
- V-related sites (`v_smem`, `v_frag`, `UPCAST_STRIDE_V`)
  use `DTypeV`.
- FP8 dequant gates that currently key on `sizeof(DTypeKV) == 1`
  should be replaced with **dtype traits**, not byte-width tests.
  Use `is_fp8<DTypeV>` for the V dequant path and `is_fp8<DTypeK>`
  for any K-side quantized handling.  Byte-width conflates FP8,
  INT8, and any other 1-byte dtype, and the semantic intent here
  is "this side is FP8 quantized," not "this side is one byte."
  See FlashInfer issue #742 for the broader 8-bit KV tracking that
  motivates trait-based dispatch.
- Shared memory layout asserts must allow K and V at different
  per-element sizes.  Some current asserts require `sizeof(DTypeKV)
  == 2` (e.g. line 541 inside the RoPE-Llama codepath); those need
  to read `sizeof(DTypeK)` not `sizeof(DTypeKV)`.
- Phrase the API as "K and V may have independent dtypes" rather
  than "V is the FP8 side."  BF16-K + FP8-V is one supported case;
  the goal is generic split, not asymmetric-FP8-V specifically.

This is the core upstream work needed.  The Params struct already
carries split types in this fork (commit
`414b187 default_prefill_params.cuh: asymmetric K/V plumbing`).  The
JIT key + URI + codegen substitution already produce a unique
asymmetric variant when given `dtype_k`/`dtype_v` (verified by our
M1 rerun: the build cache directory under
`/root/.cache/flashinfer/.../batch_prefill_with_kv_cache_dtype_q_bf16_dtype_k_bf16_dtype_v_e4m3...`
is generated correctly).  The hole is the kernel itself.

### Step 5 — `run()` validation

Replace the existing `_check_cached_qkv_data_type(q, k_cache,
self._cached_q_data_type, self._cached_kv_data_type)` call (which
conflates K and V) with explicit checks against the new cached
dtypes:

```python
if q.dtype != self._cached_q_data_type:
    raise ValueError(...)
if k_cache.dtype != getattr(self, "_cached_k_data_type",
                              self._cached_kv_data_type):
    raise ValueError(...)
if v_cache.dtype != getattr(self, "_cached_v_data_type",
                              self._cached_kv_data_type):
    raise ValueError(...)
```

The `getattr` fallback preserves behavior on plans that didn't set
asymmetric dtypes.  No relaxation: a BF16 K tensor under a planned
FP8 KV module must still raise.

### Step 6 — tests

Add the standalone test from `scripts/test_flashinfer_asym_kv.py` (or
its equivalent) to FlashInfer's test suite:

1. Decode tuple cache, Q=BF16, K=BF16, V=FP8 e4m3, vs torch reference
   within FP8 e4m3 noise band (~0.10 rel err).
2. Prefill tuple cache, same dtypes, with causal mask, same bound.
3. Negative — K=FP8 against planned K=BF16 must raise ValueError.
4. Negative — V=BF16 against planned V=FP8 must raise ValueError.
5. Homogeneous backward-compat — Q=K=V=BF16 with only `kv_data_type`
   set must still pass.

### Success criterion

Not "TypeError on `k_data_type` is gone."

The success criterion is:

> FlashInfer prefill compiles or selects a distinct BF16-K / FP8-V
> module and produces attention output within FP8 e4m3 noise of a
> torch reference using BF16 K and dequantized V.

Quantitatively, the prefill standalone rel err drops from the current
**3.49** to a value comparable to the asymmetric decode result
(~0.025-0.05).

### Optional follow-ups

- Mirror in `BatchPrefillWithRaggedKVCacheWrapper.plan()` for
  symmetry (though ragged/non-paged is less load-bearing for the
  vLLM use case).
- Document the asymmetric tuple cache flow alongside the existing
  `paged_kv_cache=(k_cache, v_cache)` runtime documentation, so the
  full plan/run path is internally consistent.

---

## Important: do NOT "fix" with a cast

Once the JIT correctly emits an asymmetric specialization, the
compile error at `prefill.cuh:1758` looks superficially like a
type-conversion bug.  Resist the urge to write:

```cpp
DTypeKV* v = reinterpret_cast<DTypeKV*>(params.v);
```

That silences the compiler at the cost of letting the kernel read
FP8 V memory as BF16 (or BF16 K memory as FP8 depending on which
side gets lied to).  It turns a clean compile-time error into
silent numerical corruption.  The compiler is protecting against
exactly the failure mode that caused the original M1 prefill rel
err of 3.49.

The compile error is **the diagnostic**.  It proves the asymmetric
specialization reached the CUDA template:

```text
Params says:    v is DTypeV*  = __nv_fp8_e4m3*
Kernel says:    v must be DTypeKV* = __nv_bfloat16*
```

The fix has to make the kernel's internal type vocabulary match
the data layout that the wrapper, JIT, and Params struct already
agree on.

## Refactor workflow (deterministic ugly)

The kernel body has ~80 `DTypeKV` sites.  Don't try to be clever.

1. Change `KernelTraits` template signature: replace `DTypeKV_`
   with `DTypeK_, DTypeV_`.
2. Update Params / RaggedParams (already done in fork commit
   `414b187`).  Top-level pointers in the kernel body:
   ```cpp
   DTypeK* k = params.k;
   DTypeV* v = params.v;
   ```
3. **Poison the unified alias** during the refactor — do NOT
   define `using DTypeKV = DTypeK;` because half the V path will
   silently become wrong:
   ```cpp
   // Temporary during refactor only:
   using DTypeKV_DO_NOT_USE = void;
   ```
4. Compile the BF16-K + FP8-V specialization and let every
   remaining `DTypeKV` use fail.  Make the compiler the unpaid
   intern.
5. Classify each failure:

   ```text
   QK path:
     - K global pointer       -> DTypeK*
     - K page loads           -> DTypeK
     - K shared memory        -> DTypeK or K compute tile type
     - K fragments            -> DTypeK-derived
     - RoPE on K              -> DTypeK / K compute type
     - k_scale                -> only if is_fp8<DTypeK>

   PV path:
     - V global pointer       -> DTypeV*
     - V page loads           -> DTypeV
     - V shared memory        -> DTypeV or V compute tile type
     - V fragments            -> DTypeV-derived
     - v_scale                -> only if is_fp8<DTypeV>

   Shared / output path:
     - attention scores       -> existing score/accum type
     - softmax                -> existing softmax type
     - output accumulator     -> existing accum type
     - output store           -> DTypeO
   ```

6. Restore homogeneous behavior **only through explicit aliases /
   helpers**, not by recreating one global `DTypeKV`.

### Watch shared-memory byte layout

The second wave of bugs lives in shared-memory sizing.  Anything
like:

```cpp
num_kv_elems * sizeof(DTypeKV)
```

must become side-specific:

```cpp
num_k_elems * sizeof(DTypeK)
num_v_elems * sizeof(DTypeV)
```

Same for static asserts:

```cpp
// before:
static_assert(sizeof(DTypeKV) == 2);
// after, K-side:
static_assert(sizeof(DTypeK) == 2);
```

Pointer offsets into raw shared-memory buffers must respect
separate alignment for K and V regions.

### Use dtype traits, not byte-size tests

Avoid:

```cpp
if constexpr (sizeof(DTypeV) == 1) { /* FP8 path */ }
```

`sizeof(T) == 1` conflates FP8, INT8, and any other 1-byte dtype.
FlashInfer issue #742 (8-bit KV tracking) discusses INT8 KV
support; byte-width is the wrong semantic test.

Use traits:

```cpp
template <typename T> struct is_fp8_type : std::false_type {};
template <> struct is_fp8_type<__nv_fp8_e4m3> : std::true_type {};
template <> struct is_fp8_type<__nv_fp8_e5m2> : std::true_type {};

if constexpr (is_fp8_type<DTypeV>::value) {
    // apply v_scale / FP8 dequant
} else {
    // BF16/FP16 V path
}
```

Apply the same trait to K independently — for K16/V8 the K side
should behave like BF16 attention, not FP8.

### Add a static_assert support matrix

Don't let the JIT generate every Frankenstein combination:

```cpp
template <typename DTypeQ, typename DTypeK, typename DTypeV,
          typename DTypeO>
struct is_supported_prefill_dtype_combo : std::false_type {};

template <>
struct is_supported_prefill_dtype_combo<
    nv_bfloat16, nv_bfloat16, __nv_fp8_e4m3, nv_bfloat16>
    : std::true_type {};

// ... legacy homogeneous combos ...

static_assert(
    is_supported_prefill_dtype_combo<DTypeQ, DTypeK, DTypeV,
                                       DTypeO>::value,
    "Unsupported FlashInfer prefill dtype combination"
);
```

Failure on an unsupported combo is then explicit, not "kernel
silently produces garbage."
