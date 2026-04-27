# Path 3: complete the asymmetric K16/V8 stack and re-do measurements

This is a request for implementation advice.  We're going to take
the third reconciliation path: finish the vLLM + FlashInfer
asymmetric serving stack so that its measurements catch up to the
paper's asymmetric claims, instead of softening the paper or
re-wording the measurement protocol.

The artifacts will not be made public yet.  Workflow stays inside
our hosts: edit on a workstation, `git format-patch` the delta,
`scp` to a runpod-class GPU pod for build + measure, and push the
final branches from `monster` to `prune` (the local backup mirror)
when stable.  No GitHub push until the stack actually serves
end-to-end.

---

## Why this brief exists

In the previous round, an M1 standalone test of the FlashInfer
asymmetric prefill kernel showed prefill output relative error of
**3.49** versus a torch reference (decode was clean at **0.0254**).
The first read of that result was that the entire vLLM + FlashInfer
asymmetric stack was broken.  That read was wrong in scope.  Two
things got missed:

1. The mcgrof GitHub fork
   `mcgrof/flashinfer:asymmetric-kv-dtype` is **9 commits behind**
   the same branch on `prune:/data/flashinfer-asym-work/flashinfer`.
   The 9 missing commits are the ones that actually implement the
   asymmetric **decode** kernel body, the JIT plumbing, the
   paged-cache split, and the Jinja codegen.  The local clone we
   were testing against did not have any of those.
2. The paper's published asymmetric K16/V8 numbers
   (PPL, GSM8K, in-memory throughput) **do not run through vLLM +
   FlashInfer at all**.  They come from a HuggingFace
   `transformers` script
   (`/data/flashinfer-asym-work/qwen_fp8v_cache_eval_v2.py`)
   that subclasses `DynamicCache` and simulates K16/V8 in pure
   Python.  K stays in native BF16; V is round-tripped through
   FP8 e4m3 cast on write and dequantized on read.  The published
   numbers prove the **precision-asymmetry hypothesis** (V
   tolerates FP8, K does not for Qwen-class models) but not a
   working asymmetric vLLM serving stack.

This brief explains both findings in detail and asks for advice
on completing the actual vLLM + FlashInfer asymmetric serving
path, then re-running the asymmetric measurements through that
real stack so the artifact catches up to the paper.

---

## Branch landscape, with code citations

There are **two private mirrors** of relevance: `prune` (local
backup, full history) and `monster` (workstation).  The GitHub
side (`mcgrof/vllm`, `mcgrof/flashinfer`, `mcgrof/lmcache`) is the
public-facing slice that lags behind both.

### FlashInfer

| Where | Branch | HEAD | Asymmetric K/V scope |
|-------|--------|------|----------------------|
| github `mcgrof/flashinfer` | `asymmetric-kv-dtype` | `414b187` | only the prefill **Params struct** plumbing.  No decode kernel.  No JIT.  No paged-cache type split. |
| `prune:/data/flashinfer-asym-work/flashinfer` | `asymmetric-kv-dtype` | `9 commits ahead of github 414b187` | **decode kernel body fully refactored**, JIT codegen and Jinja templates parameterised on `(DTypeK, DTypeV)`, `paged_kv_t<DTypeK, DTypeV>`, `decode.py.plan()` accepts `k_data_type` / `v_data_type`, both decode/prefill module wrappers forward those.  **Prefill kernel body still uses unified `DTypeKV`.** |
| `prune` | `asymmetric-kv-int4v` | `15 commits ahead of asymmetric-kv-dtype` | adds INT4-V dispatch in `decode.cuh`, `update_local_state_int4v` for packed V, vectorized cp.async loads, fused dequant.  All decode-side.  Prefill kernel still untouched. |

The 9 critical commits on prune `asymmetric-kv-dtype` that
GitHub does not have:

```
414b187 default_prefill_params.cuh: asymmetric K/V plumbing
9b4598a decode.cuh: asymmetric K/V for SingleDecode
6d8f838 attention: fix vec_size + binding for asymmetric K/V decode
dfe0149 attention/decode: implement asymmetric K/V in BatchDecode kernel body
b2b7196 decode/prefill: forward dtype_k/dtype_v through cached module wrappers
e4a4d58 decode.py: add k_data_type / v_data_type to BatchDecodeWithPagedKVCacheWrapper.plan
be261d3 attention/decode: plumb DTypeK/DTypeV, static_assert equal until kernel rewrite
215a14d jit: thread dtype_k / dtype_v through batch decode and prefill generators
e7f5823 jinja: add DTypeK and DTypeV to decode/prefill config templates
2b6b2eb page.cuh: split paged_kv_t DType into DTypeK and DTypeV
```

So the **decode** kernel really does support asymmetric K/V on
prune.  Concretely, on prune `asymmetric-kv-dtype`,
`include/flashinfer/attention/decode.cuh` has:

```cpp
using DTypeK = typename Params::DTypeK;
using DTypeV = typename Params::DTypeV;

const DTypeK* k = params.k;
const DTypeV* v = params.v;

// ...

constexpr size_t k_tile_bytes = kv_tile_elems * sizeof(DTypeK);
constexpr size_t v_tile_bytes = kv_tile_elems * sizeof(DTypeV);

DTypeK* k_smem = (DTypeK*)smem;
DTypeV* v_smem = (DTypeV*)(smem + k_tile_bytes);

// ...

constexpr uint32_t vec_bits_k = sizeof(DTypeK) * vec_size * 8;
constexpr uint32_t vec_bits_v = sizeof(DTypeV) * vec_size * 8;
```

Source comment:
> `k_smem` stores K data typed as `DTypeK`; `v_smem` stores V data
> typed as `DTypeV`.  When `DTypeK == DTypeV` (the symmetric case),
> the layout is byte-identical.

That preserves homogeneous behavior and adds the asymmetric path
cleanly.  This is exactly the kind of refactor the prefill kernel
needs but never got.

In contrast, on the same prune branch, prefill still has unified
`DTypeKV` everywhere.  In `include/flashinfer/attention/prefill.cuh`:

```cpp
template <uint32_t HEAD_DIM_QK, uint32_t HEAD_DIM_VO,
          typename DTypeQ, typename DTypeKV, typename DTypeO>
struct PrefillKernelTraits {
  alignas(16) DTypeKV k_smem[CTA_TILE_KV * HEAD_DIM_QK];
  alignas(16) DTypeKV v_smem[CTA_TILE_KV * HEAD_DIM_VO];
  // ...
};

// later:
DTypeKV* k = params.k;
DTypeKV* v = params.v;     // <-- prefill.cuh:1758
```

When the JIT correctly emits an asymmetric specialization
(BF16-K, FP8-V), `params.v` has type `DTypeV* = __nv_fp8_e4m3*`
but the kernel body declares `DTypeKV* v` where `DTypeKV` is
`__nv_bfloat16`.  The compiler refuses to pretend FP8 is BF16
and rejects with:

```
prefill.cuh:1758: error: a value of type "RaggedParams::DTypeV *"
    (aka "__nv_fp8_e4m3 *") cannot be used to initialize an entity
    of type "DTypeKV *" (aka "__nv_bfloat16 *")
      DTypeKV* v = params.v;
```

So the prefill kernel needs the same refactor that `decode.cuh`
already has.  About 80 type sites in `prefill.cuh`: K shared
memory, V shared memory, K-frag, V-frag, `vec_cast` FP8 dequant,
RoPE, MMA, plus several `static_assert(sizeof(DTypeKV) == ...)`
guards.

### vLLM

| Where | Branch | HEAD | Asymmetric K/V scope |
|-------|--------|------|----------------------|
| github `mcgrof/vllm` | `asymmetric-kv-plumbing` | `5b4cc97` | tuple-handling at config / spec / selector level; Hybrid-Mamba tuple fix |
| `prune:/data/vllm` | `asymmetric-kv-plumbing` | `6fb8e95a0` (almost identical) | M2-verified allocator splits raw int8 into `(k_cache, v_cache)` tuple, M3-contract-verified writer splits into separate `reshape_and_cache_flash` calls per side |

The **allocator** on prune (`vllm/v1/worker/gpu/attn_utils.py`,
function `_reshape_kv_cache`) detects asymmetric specs:

```python
if (kv_cache_spec.v_dtype is not None
        and kv_cache_spec.v_dtype != kv_cache_spec.dtype):
    k_dtype = kv_cache_spec.dtype
    v_dtype = kv_cache_spec.v_dtype
    k_page_bytes = bs * nh * hd * get_dtype_size(k_dtype)
    v_page_bytes = bs * nh * hd * get_dtype_size(v_dtype)
    page_bytes = k_page_bytes + v_page_bytes
    raw_pages = raw_tensor.view(num_blocks, page_bytes)
    k_raw = raw_pages[:, :k_page_bytes].contiguous()
    v_raw = raw_pages[:, k_page_bytes:].contiguous()
    k_cache = k_raw.view(k_dtype).view(num_blocks, bs, nh, hd)
    v_cache = v_raw.view(v_dtype).view(num_blocks, bs, nh, hd)
    kv_caches[layer_name] = (k_cache, v_cache)
    continue
```

That tuple gets passed through `bind_kv_cache`
(`vllm/v1/worker/utils.py:513`):

```python
for layer_name, kv_cache in kv_caches.items():
    forward_context[layer_name].kv_cache = kv_cache
```

The **writer** on prune
(`vllm/v1/attention/backends/flashinfer.py:1699`) handles the
tuple correctly:

```python
if isinstance(kv_cache, tuple):
    k_cache, v_cache = kv_cache
    v_dtype = getattr(self, '_v_cache_str', None)
    v_dtype = v_dtype if v_dtype else "auto"
    # Write K at native dtype (no quantization)
    torch.ops._C_cache_ops.reshape_and_cache_flash(
        key, key, k_cache, k_cache, slot_mapping,
        "auto", layer._k_scale, layer._k_scale)
    # Write V at FP8 (or specified v dtype)
    torch.ops._C_cache_ops.reshape_and_cache_flash(
        value, value, v_cache, v_cache, slot_mapping,
        v_dtype, layer._k_scale, layer._v_scale)
```

The **forward attention path is the gap.**  In the same file,
`FlashInferImpl.forward` (line 1302) signature:

```python
def forward(
    self,
    layer: torch.nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    kv_cache: torch.Tensor,
    attn_metadata: FlashInferMetadata,
    output: torch.Tensor | None = None,
    output_scale: torch.Tensor | None = None,
    output_block_scale: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    kv_cache: KV cache tensor with different possible shapes:
        - NHD: [num_blocks, 2, block_size, num_kv_heads, head_size]
        - HND: [num_blocks, 2, num_kv_heads, block_size, head_size]
    """
```

`kv_cache: torch.Tensor`.  The function then does, at line 1433:

```python
stride_order = FlashInferBackend.get_kv_cache_stride_order()
kv_cache_permute = kv_cache.permute(*stride_order)
```

If the M2 allocator produced a tuple and `bind_kv_cache` handed
that tuple to the layer untouched, this `kv_cache.permute()` call
**raises `AttributeError: 'tuple' object has no attribute
'permute'`** before any kernel runs.  This is the exact gap the
asymmetric serving path needs closed: the read-side forward path
in vLLM never grew tuple support, even on prune.

The standard non-DCP prefill path then calls (line 1480):

```python
prefill_wrapper.run(
    prefill_query,
    kv_cache_permute,
    k_scale=layer._k_scale_float,
    v_scale=layer._v_scale_float,
    out=output[num_decode_tokens:],
)
```

That call goes into `BatchPrefillWithPagedKVCacheWrapper.run`,
which is what would have hit `prefill.cuh:1758` if we'd gotten
that far.  We don't, because we crash at `permute()` first.

So vLLM's asymmetric path has two missing wires, in this order:

1. **Forward read-side** must accept tuple `kv_cache` and route
   it to FlashInfer's tuple `paged_kv_cache=(k_cache, v_cache)`
   API.
2. **FlashInfer prefill kernel** must split `DTypeKV` into
   `DTypeK` / `DTypeV` so the asymmetric specialization the JIT
   already emits actually compiles.

If we close (1) but not (2), asymmetric prefill will still fail
at `prefill.cuh:1758`.  Asymmetric **decode** will work because
prune's decode kernel is already refactored.

### LMCache

The asymmetric storage codec
(`mcgrof/lmcache:asymmetric-kv-codec` — committed to prune as
well) is independent of the FlashInfer / vLLM refactor.  Storage
codec correctness, split-tier NVMe traffic ratio, and the 0.75x
storage ratio are measured and verified.  This brief is not
asking about LMCache; it's about the runtime forward path that
LMCache spills/restores into.

---

## What "M1 prefill rel err 3.49" actually means

The earlier session's standalone FlashInfer test has two
sub-tests, run on a fresh H100:

```text
[1] decode  Q=BF16  K=BF16  V=FP8_e4m3  tuple paged_kv_cache
    median relative error vs BF16 reference: 0.0254  (bound: <0.10)
    PASS

[2] prefill Q=BF16  K=BF16  V=FP8_e4m3  tuple paged_kv_cache
    median relative error vs BF16 reference: 3.4946  (bound: <0.10)
    FAIL
```

Decode passes because prune's decode kernel really is asymmetric.

Prefill "passes the JIT, fails the math" because:

- After we extended `BatchPrefillWithPagedKVCacheWrapper.plan()`
  with `k_data_type` / `v_data_type` and forwarded them to
  `get_batch_prefill_module(...)`, the JIT URI correctly becomes
  something like
  `batch_prefill_with_kv_cache_dtype_q_bf16_dtype_k_bf16_dtype_v_e4m3_...`.
- The codegen tries to compile that variant.
- The compile fails inside `prefill.cuh` because the kernel body
  still uses unified `DTypeKV`.

In the version of the prefill standalone we *can* run today, JIT
specialization keys still collapse to the symmetric FP8 module
when there is no separate `dtype_k` / `dtype_v` plumbing.  Then the
symmetric-FP8 module is asked to consume a BF16 K tensor.  It
reads the BF16 bytes as FP8.  Output relative error explodes to
3.49.

So "rel err 3.49" is the kernel-correctness signal you get when a
BF16 K tensor is silently fed through an FP8-K kernel.  It is real
and exactly the failure mode the asymmetric design is supposed to
prevent.

**The reason this didn't break the published paper measurements**
is that those measurements never went through this kernel.  The
paper's asymmetric K16/V8 numbers come from
`/data/flashinfer-asym-work/qwen_fp8v_cache_eval_v2.py`, which
runs HuggingFace `transformers` with a custom `DynamicCache`
subclass:

```python
class FP8VLayer(DynamicLayer):
    def update(self, key_states, value_states, cache_kwargs=None):
        if not self.is_initialized:
            self.lazy_initialization(key_states)
        v_fp8 = value_states.to(torch.float8_e4m3fn).to(value_states.dtype)
        self.keys = torch.cat([self.keys, key_states], dim=-2)
        self.values = torch.cat([self.values, v_fp8], dim=-2)
        return self.keys, self.values
```

K stays in native BF16, V is cast to FP8 e4m3 and immediately
dequantized back.  This is a clean simulation of K16/V8 storage
*precision* using HuggingFace's eager attention path.  It produces
real numbers for PPL, GSM8K, etc., and validates the precision-
asymmetry hypothesis (V tolerates FP8; FP8-K corrupts Qwen).

The result file `/data/flashinfer-asym-work/qwen_asym.jsonl` has
entries like:

```json
{"config": "Qwen-Asym-FP16K-FP8V",
 "model": "Qwen/Qwen2.5-7B-Instruct",
 "wikitext_ppl": 8.387, "gsm8k_acc": 0.55, ...}
```

These are valid measurements of the **precision asymmetry effect**.
They do not exercise vLLM's paged cache, FlashInfer's prefill
kernel, or any of the asymmetric scaffolding on the prune
branches.  The paper's wording "vLLM 0.19 + FlashInfer asymmetric
branch on H100" describes the eval-protocol harness and hardware
but does not describe the cache layer that produced the asymmetric
rows.  The non-asymmetric rows in the same battery (FP16 baseline,
calibrated and uncalibrated symmetric FP8) probably *do* run
through vLLM + FlashInfer normally because the legacy 5-D tensor
path supports them.

So the artifact and the paper are consistent on the asymmetry
*hypothesis* and inconsistent on what stack produces the
asymmetric *rows*.  This brief asks for advice on closing that
gap by completing the runtime stack, not by hedging the paper.

---

## What "Path 3" looks like concretely

Goal: drive an end-to-end vLLM + FlashInfer asymmetric K16/V8
serving path that produces asymmetric Qwen2.5-7B PPL / GSM8K /
throughput numbers from the **same** harness that produces the
symmetric baselines, on the **same** GPU, with the same kernel
infrastructure.

To get there, we need:

### Subgoal 1: vLLM `forward()` tuple support

In `vllm/v1/attention/backends/flashinfer.py`, make
`FlashInferImpl.forward` accept a tuple `(k_cache, v_cache)` for
asymmetric layers and route it to FlashInfer's tuple
`paged_kv_cache` API.

Concrete changes to design:

1. Detect tuple at function entry:
   ```python
   if isinstance(kv_cache, tuple):
       k_cache, v_cache = kv_cache
   else:
       k_cache = v_cache = None
   ```
2. Replace `kv_cache_permute = kv_cache.permute(*stride_order)`
   with side-aware permutes:
   ```python
   if k_cache is not None:
       k_cache_permute = k_cache.permute(*stride_order_4d)
       v_cache_permute = v_cache.permute(*stride_order_4d)
       paged_kv_cache_arg = (k_cache_permute, v_cache_permute)
   else:
       kv_cache_permute = kv_cache.permute(*stride_order)
       paged_kv_cache_arg = kv_cache_permute
   ```
   `stride_order_4d` differs from the 5-D one because the tuple
   tensors are 4-D `[num_blocks, block_size, num_kv_heads,
   head_size]` (NHD), not 5-D `[num_blocks, 2, block_size, ...]`.
3. Pass `paged_kv_cache_arg` to `prefill_wrapper.run` and
   `decode_wrapper.run` instead of `kv_cache_permute`.  FlashInfer's
   `run()` already accepts either a single 5-D tensor or a tuple.
4. Cascade attention path (line 1421) and TRTLLM paths (line
   1488+) probably also need tuple-aware handling or an explicit
   `NotImplementedError` for the asymmetric case.
5. Make `FlashInferMetadataBuilder.build` pass `dtype_k` and
   `dtype_v` to the `prefill_wrapper.plan()` call so the JIT URI
   is keyed asymmetrically — this is the small wiring fix already
   described in the M1 work, four lines per `plan()` callsite.
6. `_v_cache_str` is already cached on `FlashInferImpl` from the
   asymmetric tuple unpack at line 1240; reuse it for the read
   path.

### Subgoal 2: FlashInfer prefill kernel template refactor

`include/flashinfer/attention/prefill.cuh` needs the same kind of
refactor `decode.cuh` already has on prune `asymmetric-kv-dtype`:

1. `KernelTraits` template signature: replace `DTypeKV_` with
   `DTypeK_` and `DTypeV_`.  When equal, behavior must be
   bit-identical to today.
2. Top-level kernel-body locals:
   ```cpp
   DTypeK* k = params.k;
   DTypeV* v = params.v;
   ```
3. K-side sites use `DTypeK`: `k_smem`, `k_frag`,
   `UPCAST_STRIDE_K`, K RoPE, K page loads, K-side scale handling
   gated on `is_fp8<DTypeK>`.
4. V-side sites use `DTypeV`: `v_smem`, `v_frag`,
   `UPCAST_STRIDE_V`, V page loads, `vec_cast<DTypeQ, DTypeV>`,
   `v_scale` gated on `is_fp8<DTypeV>`.
5. **Do not gate quantized behavior on `sizeof(T) == 1`**; that
   conflates FP8, INT8, and any other 1-byte type.  Use dtype
   traits `is_fp8<T>` (FlashInfer issue #742 is the broader 8-bit
   KV tracking that motivates trait-based dispatch).
6. `static_assert(sizeof(DTypeKV) == 2)` and similar guards become
   side-specific (e.g., `sizeof(DTypeK) == 2` for the K-RoPE
   path).
7. Add a `static_assert` support matrix limiting allowed `(Q, K,
   V, O)` tuples so unsupported combinations fail explicitly
   rather than producing silent garbage.
8. Preserve homogeneous symmetric paths bit-identically when
   `DTypeK == DTypeV`.

This is the substantial CUDA template refactor.  The decode-side
work in commit `dfe0149` is the pattern.

### Subgoal 3: ports + glue

- `BatchDCPPrefillWrapper.plan()` and the cascade prefill plan
  call probably also need the same `dtype_k` / `dtype_v` kwargs.
- `_unpack_paged_kv_cache` already returns `(k_cache, v_cache)`
  if its input is a tuple, so once vLLM passes a tuple, the
  prefill `run()` validation we added in M1 keeps working.
- The FlashInfer prefill `run()`-time `_check_cached_qkv_data_type`
  helper currently rejects mixed dtypes (BF16 K with FP8 plan);
  the asymmetric-aware replacement is in our M1 work.

### Subgoal 4: re-do measurements

Once the stack runs, re-do the asymmetric Qwen2.5-7B battery
through the **same** vLLM + FlashInfer path that produces the
baseline FP16 and symmetric FP8 numbers:

- WikiText-2 PPL at T=2048
- GSM8K 8-shot strict-match n=200
- Throughput on H100 with at least one realistic batch / context
  combination

Compare against:
- The HuggingFace `FP8VLayer` simulation numbers (sanity check;
  asymmetric vLLM should match HF asymmetric within FP8 noise)
- The symmetric FP8 vLLM baseline (which corrupts Qwen output;
  asymmetric vLLM should be clean)
- The FP16 vLLM baseline

The replacement claim in the paper becomes: asymmetric K16/V8
numbers come from the modified vLLM + FlashInfer fork (matching
the symmetric measurement protocol), with the HF `DynamicCache`
simulation kept as a cross-check.

---

## Workflow constraint: no public publish yet

We are not pushing to `mcgrof/vllm` or `mcgrof/flashinfer` on
GitHub during this work.  The artifact is private until it
runs end-to-end.

```text
1. Edit on monster (or on the workstation running this brief).
2. git format-patch the deltas:
       cd flashinfer; git format-patch HEAD~N -o /tmp/asym-patches/
3. scp /tmp/asym-patches/*.patch to the runpod GPU pod.
4. On the pod:
       cd flashinfer-src; git am /tmp/asym-patches/*.patch
       pip install --no-build-isolation -e .
       python3 test_flashinfer_asym_kv.py    # M1 standalone
       # then build vLLM and run the Qwen2.5-7B asym battery
5. On stable:
       monster$ git push prune asymmetric-kv-prefill-refactor
   (prune is the local backup mirror; nothing public.)
```

This is just to confirm the workflow expectation in case the
advice has tooling assumptions.

---

## Specific questions for review

1. **vLLM forward tuple support.**  Is the design above sound?
   Is there a vLLM convention for "this layer's `kv_cache` may be
   a tuple" that we should match (MLA-like paths, mixed-precision
   experiments) instead of inventing one?  The M2 allocator
   already produces tuples; we're asking whether the read-side
   should detect at function entry, or whether `bind_kv_cache`
   should normalise into something else first.

2. **FlashInfer prefill kernel refactor sequencing.**  Is the
   "poison the unified `DTypeKV` alias and let the compiler
   classify each site as K-side / V-side / accum/output" workflow
   the right approach for a kernel template this dense?  Or is
   there a smaller, less risky landing path that splits the
   refactor into multiple commits with intermediate compile
   guarantees?

3. **Static_assert support matrix.**  What's the right scope for
   the supported `(Q, K, V, O)` dtype combinations?  Minimum
   useful set is BF16/BF16/FP8e4m3/BF16; do we also need
   FP16/FP16/FP8e4m3/FP16 and INT8 variants for the int4v
   follow-on, or is that scope creep?

4. **Decode regression risk.**  Prune `asymmetric-kv-dtype`
   already has a working asymmetric decode kernel.  What's the
   right way to verify the prefill refactor doesn't regress
   decode correctness?  We have an M1-style standalone decode
   test; is there a more demanding integration test we should
   add?

5. **Measurement replication strategy.**  Once the stack runs, is
   there a recommended approach for replicating the HF
   `FP8VLayer` numbers through vLLM + FlashInfer with confidence
   that small numeric differences are FP8 noise vs implementation
   bugs?  E.g., we should expect WikiText-2 PPL within ±0.05 of
   the HF result, not exact match.

6. **Cascade and TRTLLM paths.**  Both branches in
   `FlashInferImpl.forward` currently assume a 5-D paged tensor.
   For the first runnable end-to-end measurement, is it
   acceptable to `NotImplementedError` on those paths in
   asymmetric mode and only support the standard non-DCP prefill
   + standard decode?  Or are we going to need cascade asymmetric
   support before any practical Qwen run?
