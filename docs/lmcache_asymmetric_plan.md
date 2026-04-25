# LMCache asymmetric KV cache (FP16-K / FP8-V) plan

## Why this exists

The decode paper proves that asymmetric quantization — FP16 on the
keys, FP8 on the values — preserves quality on Qwen-class fragile
models where symmetric FP8 collapses, while still cutting KV traffic
by 25% per element pair (12 bits across the K/V boundary, vs 16 for
symmetric FP16 and 8 for symmetric FP8). That win is currently only
realized inside the vLLM/FlashInfer execution path. When KV cache
crosses a serialization boundary — page-out to disk, prefix sharing
across replicas, NIXL-based tiering — LMCache writes K and V at the
same dtype, so we either pay full FP16 storage everywhere or accept
the symmetric-FP8 quality cliff. This plan adds an explicit
asymmetric on-disk format to LMCache so cached state matches the
running execution path bit-for-bit, and so the storage savings
asymmetric provides in compute also show up on disk.

The plan also pins everything to a CPU-runnable test ladder. The
asymmetric path has three places it can break — the dtype tag, the
quantization arithmetic, and the cross-boundary roundtrip — and all
three can be exercised in pure PyTorch without a GPU. We only spin
up a GPU once those three are green.

## Table of contents

- [Current LMCache state](#current-lmcache-state)
- [Format design](#format-design)
- [Phase 0: read-only reconnaissance](#phase-0-read-only-reconnaissance)
- [Phase 1: format spec + CPU serde](#phase-1-format-spec--cpu-serde)
- [Phase 2: storage backend wiring](#phase-2-storage-backend-wiring)
- [Phase 3: vLLM connector hook](#phase-3-vllm-connector-hook)
- [Phase 4: GPU smoke + parity](#phase-4-gpu-smoke--parity)
- [Phase 5: end-to-end paper-relevant eval](#phase-5-end-to-end-paper-relevant-eval)
- [Test ladder summary](#test-ladder-summary)
- [Risks and open questions](#risks-and-open-questions)

## Current LMCache state

What exists today (as of `bb1a6bd` in `LMCache/main`):

`lmcache/v1/protocol.py` already maps `torch.float8_e4m3fn` → 7 and
`torch.float8_e5m2` → 8 in its `DTYPE_TO_INT` table, so the on-disk
type ID space is ready for FP8. `lmcache/v1/memory_management.py`
defines `MemoryObjMetadata` with a single `dtype` field — that
single-dtype assumption is the central thing we need to break.

The serde plugin layer at
`lmcache/v1/storage_backend/naive_serde/` has a clean abstract
interface (`Serializer.serialize`, `Deserializer.deserialize`) and
three implementations: `NaiveSerializer` (pass-through), `KIVISerializer`
(stubbed `TODO(Yuhan)`, no logic yet), and `CacheGenSerializer`. The
KIVI stub is informative — it shows the intended extension point and
also confirms that fancy KV compression is meant to live at the serde
layer, not be folded into the storage backend.

Storage backends below the serde layer (`local_backend`, `gds_backend`,
`weka_gds_backend`, NIXL connectors under `connector/`) are
dtype-agnostic byte-level — they take a `MemoryObj` and write its
backing buffer. The format change does not need to touch them.

Tests under `tests/v1/` have direct-coverage harnesses for the
storage backends, memory management, GPU connector, mem kernels,
NIXL, GDS/Weka. They use `pytest`, with shared fixtures in
`tests/conftest.py`. There is no asymmetric test today.

## Format design

The minimum viable extension is to lift `MemoryObjMetadata` from
single-dtype to dual-dtype, gated by a format flag so old readers
keep working on old blobs.

A cache entry under the asymmetric format contains, in order:

1. A magic + version preamble (already there in some form via
   `protocol.py`; we add a 2-byte format-flag field).
2. Layer index, block layout (unchanged).
3. **K blob** — typed by `dtype_K` (FP16 or BF16).
4. **V blob** — typed by `dtype_V` (FP8 e4m3, with optional FP8
   e5m2 fallback).
5. **V scale tensor** — one FP32 scalar per attention head, written
   inline after the V blob. Per-head is the granularity vLLM's
   asymmetric branch already uses for `_v_scale_float`; per-tensor
   would lose accuracy on Qwen-class models for the same reason
   per-tensor symmetric FP8 fails on activations.

Entries written under the legacy format (single dtype, no scale
tensor) keep working unchanged. Entries written under the asymmetric
format require a reader that understands the format-flag. A reader
encountering a future flag it doesn't recognize must hard-fail the
entry rather than silently misinterpret bytes.

Concretely, `MemoryObjMetadata` grows two optional fields:

```python
@dataclass
class MemoryObjMetadata:
    shape: torch.Size
    dtype: Optional[torch.dtype]               # legacy single-dtype
    fmt: MemoryFormat
    # NEW:
    kv_dtypes: Optional[Tuple[torch.dtype, torch.dtype]] = None
    v_scales: Optional[torch.Tensor] = None    # FP32, shape [num_heads]
```

When `kv_dtypes` is set, `dtype` is `None` and reads/writes use the
asymmetric path. Mirror the same on-disk: the format-flag bit decides
which header layout to parse.

We add an enum:

```python
class CacheFormat(IntEnum):
    LEGACY = 0
    ASYMMETRIC_K16_V8_E4M3 = 1
    ASYMMETRIC_K16_V8_E5M2 = 2  # reserved, not implemented in v1
```

## Phase 0: read-only reconnaissance

**Goal:** turn the rough understanding above into a per-call
inventory before touching code, so we don't break things we
didn't realize were load-bearing.

Tasks:

- Trace every read site of `MemoryObjMetadata.dtype`. List which
  ones can fall back to a default and which need to know
  `(dtype_K, dtype_V)` precisely.
- Trace every write site of `protocol.py`'s `DTYPE_TO_INT` mapping;
  confirm no third-party serializer has its own private dtype table.
- List every backend's serialization entry point. Confirm
  `csrc/`'s C++ paths only see opaque bytes.
- Walk `tests/v1/` and tag which tests will need fixture updates
  versus which can be left alone (most tests use FP16 only and
  should remain unchanged).

Deliverable: a short markdown note `tools/lmcache_asym/RECON.md`
checked into knlp listing each touch-point with file:line. No
LMCache source changes in this phase.

Tests added: none (this is a read).

## Phase 1: format spec + CPU serde

**Goal:** define the asymmetric format, implement
quant/dequant + serde in pure PyTorch, prove correctness without a
GPU.

Tasks in `LMCache/lmcache/v1/storage_backend/naive_serde/`:

- New file `asymmetric_kv_serde.py` with
  `AsymmetricKVSerializer` and `AsymmetricKVDeserializer`. Both work
  entirely on CPU tensors; GPU support is added in Phase 4 by
  preserving the device of the input.
- New file `asymmetric_kv_quant.py` with two helpers:
  `compute_v_scales(v: Tensor, axis: int) -> Tensor` and
  `quantize_v_fp8(v: Tensor, scales: Tensor) -> Tensor`.
- Register the new serde tag (`"asymmetric_k16_v8"`) in
  `naive_serde/__init__.py`.

Format-on-disk additions in `lmcache/v1/protocol.py`:

- Add `FORMAT_FLAG_ASYM_K16_V8_E4M3 = 1`.
- Extend the metadata serializer to write the flag, write `v_scales`
  inline, write K and V blobs back-to-back. Read path: parse flag
  first, dispatch.

Test ladder for Phase 1 (all CPU-only, run via
`pytest tests/v1/test_asymmetric_kv_serde.py`):

1. **Unit: scale computation.**
   - Random tensor, hand-computed expected per-head amax / 448, exact
     match.
   - All-zero head: scale should be 1.0 (or some sentinel that
     dequantizes to zero), not 0/0 NaN.
   - Single huge outlier: scale clamps the quant range correctly,
     the rest of the head dequantizes to within FP8 noise.
2. **Unit: quant/dequant roundtrip bound.**
   - For random tensors, max relative error after quant→dequant is
     below an empirically-chosen bound (≤ 6.3% for FP8 e4m3 around
     mean values; bound enforced in test).
   - Saturation at ±448: input tensors with values > 448 should
     saturate, not overflow to ±inf or wrap.
   - Idempotence: quantizing an already-quantized tensor is the
     identity.
3. **Unit: metadata serializer roundtrip.**
   - Build `MemoryObjMetadata` with `kv_dtypes=(half, e4m3)` and
     populated `v_scales`, serialize to bytes, parse back, fields
     match exactly.
   - Same for `dtype=half, kv_dtypes=None` (legacy path) — round
     trip must equal input byte-for-byte.
   - Format-flag mismatch: serialize as v1, hand-flip flag bit,
     deserializer raises explicit error rather than silently producing
     garbage.
4. **Unit: Serializer/Deserializer pair.**
   - Take a synthetic `(K, V)` pair as a `MemoryObj`, run through
     `AsymmetricKVSerializer.serialize` then `AsymmetricKVDeserializer.
     deserialize`, recover `(K_recovered, V_recovered)`.
   - `K` returns bit-exact (no quantization on K).
   - `V_recovered ≈ V` within FP8 quant noise (bound from #2).
5. **Property tests** (using `hypothesis`):
   - For random shapes
     `[B in 1..4, T in 1..256, H in 1..16, D in 32..128]`,
     dtypes `(float16, bfloat16) × (float8_e4m3fn,)`, the roundtrip
     never raises and stays within the noise bound.
   - For random byte-level corruption of the serialized header,
     deserializer detects and raises `CorruptHeaderError` rather
     than producing a tensor.
6. **Integration: NaiveSerializer parity.**
   - When asymmetric path is used with `dtype_K == dtype_V == fp16`
     (i.e., the degenerate case), output should be byte-equal to
     `NaiveSerializer` output. Catches accidental format drift.

Phase 1 exit criterion: `pytest tests/v1/test_asymmetric_kv_serde.py`
green on a CPU box, mypy clean, ≥ 60 cases including the property
tests, every assertion has a comment naming the bug it would catch.

## Phase 2: storage backend wiring

**Goal:** the new serde shows up at the storage-backend boundary so
disk writes actually exercise the format.

Tasks:

- Wire `CreateSerde("asymmetric_k16_v8", ...)` to return the new
  serializer/deserializer pair.
- Add a config knob `LMCacheEngineConfig.kv_format`: `"naive" |
  "kivi" | "cachegen" | "asymmetric_k16_v8"` with `"naive"` default
  for backwards compatibility.
- `local_backend.py` and the abstract base remain unchanged (they
  see opaque bytes).
- One adjustment in `MemoryObj` allocation: when `kv_dtypes` is set,
  the buffer needs sized for `K_bytes + V_bytes + scales_bytes`
  rather than `shape * single_dtype`. Add a helper
  `MemoryObj.physical_size_for_kv_dtypes(...)` so allocation logic
  is one place.

Test ladder for Phase 2 (CPU-only):

1. **Backend integration: local disk roundtrip.**
   - Start an in-process `LocalDiskBackend` with a tmpdir, configured
     for `kv_format="asymmetric_k16_v8"`.
   - Write a synthetic `(K, V)` pair against a synthetic cache key,
     read back. K bit-exact, V within noise bound, scales preserved.
   - Repeat 100 random keys to exercise concurrent puts/gets.
2. **Backend integration: naive backwards compat.**
   - Write under `kv_format="naive"`, read with the same config.
   - Cross-format read: writing under `naive` and reading under
     `asymmetric_k16_v8` must surface a clear "format mismatch"
     error, not return a corrupted tensor.
3. **Eviction interaction.**
   - Fill a small-capacity backend until eviction triggers, all
     entries asymmetric. Capacity accounting must use the
     asymmetric `physical_size`, not the legacy single-dtype size.
   - Test for off-by-25% capacity bug (the bug a naïve impl would
     have if it accidentally counted V as FP16 bytes).
4. **NIXL connector roundtrip.**
   - Either against a real NIXL endpoint (skip unless
     `NIXL_TEST_ENDPOINT` env var set) or against the
     fake-NIXL test double already in `tests/v1/test_connector.py`.
   - Same correctness expectations as #1.
5. **GDS/Weka.**
   - Skip on CPU (these are GPU-direct paths). Marked `@pytest.mark.gpu`
     for Phase 4.

Phase 2 exit criterion: `pytest tests/v1/storage_backend/` green on
CPU, capacity tests pass, format-mismatch error message includes the
expected vs observed format flag.

## Phase 3: vLLM connector hook

**Goal:** when vLLM uses LMCache and is itself running under the
asymmetric-kv-plumbing branch, the on-disk format matches what
vLLM holds in HBM.

Touch-points:

- `lmcache/v1/gpu_connector.py` already has logic to copy KV slabs
  between vLLM and LMCache memory. Add a path that, when vLLM
  reports `kv_cache_dtype = (auto, fp8_e4m3)`, treats the K and V
  slabs as separate buffers with the corresponding dtypes.
- The vLLM side already exposes per-head `_v_scale_float` on each
  `Attention` layer (asymmetric-kv-plumbing commit `e539b254a`).
  LMCache needs to read those scales when serializing on miss-fill,
  and write them back into the layer when restoring on miss-load.
- Add a config knob: when `kv_format="asymmetric_k16_v8"` is set
  but vLLM is running symmetric, refuse with a clear error message
  rather than silently quantizing V (which would break parity).

Test ladder for Phase 3 (CPU still — we mock the vLLM hook):

1. **Mocked vLLM call.**
   - Build a `MockVllmAttention` exposing `_v_scale_float` and a
     synthetic KV slab. Run a single put/get cycle.
   - Verify the scales LMCache writes are exactly the scales it
     reads back (no rescaling drift).
2. **Mismatch detection.**
   - Mock vLLM as symmetric while LMCache is asymmetric. Expect a
     hard error at engine startup, not at first cache write.
   - Same in reverse.
3. **Multi-layer plumbing.**
   - 16 layers, each with a different per-head scale tensor. Round
     trip preserves all 16 scale tensors with the right
     layer-to-tensor mapping.

Phase 3 exit criterion: a CPU integration test that simulates a full
vLLM-style put/get cycle through LMCache passes; the mismatch error
message tells the user how to fix their config.

## Phase 4: GPU smoke + parity

**Goal:** run the same Phase 1-3 tests on a real GPU to flush out
device-specific bugs (FP8 quantization on real hardware can differ
from CPU emulation in ways that matter — saturation, NaN handling,
tensor-core path).

Single H100 pod required for this phase. Reuse the same test files
written in earlier phases; mark them `@pytest.mark.gpu` and skip on
CPU. The asserts stay the same; only the device changes.

Specific GPU-only additions:

1. **CPU vs GPU determinism.**
   - For the same input tensor, `quantize_v_fp8` on CPU vs CUDA
     produces identical FP8 output bytes (up to the documented
     hardware rounding mode).
2. **GDS / Weka roundtrip.**
   - Now unmasked — write asymmetric KV via GDS direct, read back,
     correctness as in Phase 2.
3. **Vllm e2e: quality matches in-memory path.**
   - Run a 4-prompt prefill+decode with vLLM
     `kv_cache_dtype=(auto, fp8_e4m3)`, no LMCache.
   - Run the same 4 prompts with LMCache `asymmetric_k16_v8`,
     forcing one prompt to come from a cache hit.
   - Output token sequences must match within FP8 quantization noise
     (target: ≥ 95% of generated tokens identical for paper-relevant
     models on a deterministic decode setting).

Phase 4 exit criterion: full test suite green on a single H100 pod,
GDS roundtrip working, vLLM e2e quality parity demonstrated on at
least one tolerant model (Llama-3.1-8B) and one fragile model
(Qwen2.5-7B).

## Phase 5: end-to-end paper-relevant eval

**Goal:** measure storage savings, hit-path latency, and cache-quality
preservation under realistic prefix-caching workloads.

Workloads to drive (reuse code from
`tools/phase1_calib_saturation/`):

1. **Storage cost** — for a fixed corpus of long prompts, how many
   bytes does asymmetric K16/V8 use on disk vs symmetric FP16?
   Expected: ≈ 75% of FP16 (K unchanged + V halved).
2. **Hit latency overhead** — single-tenant decode, miss-then-hit
   pattern. Measure cache-fill (extra V quant cost on miss) and
   cache-restore (V dequant on hit) deltas vs FP16 baseline.
3. **Quality preservation** — WikiText-2 PPL at T=2048 on Qwen2.5-7B
   with prefix-cached KV. Result must match the in-memory asymmetric
   number from `kv_asymmetric.tex`.
4. **Capacity uplift** — at fixed disk budget, how many more prompts
   fit? Expected ≈ 1.33x more entries.

Deliverables:

- `tools/lmcache_asym/measure_storage_cost.py`
- `tools/lmcache_asym/measure_hit_latency.py`
- A short results section in the paper if the numbers warrant.
- Archive raw data under
  `prune:/data/knlp-key-results/lmcache-asymmetric-YYYYMMDD/`.

Phase 5 exit criterion: storage cost is within 1% of theoretical
3/4 ratio; hit-path overhead dominated by disk I/O (V dequant cost
≪ miss-fetch cost) on H100; quality matches in-memory asymmetric.

## Test ladder summary

The pyramid runs roughly:

- ~60 unit tests in Phase 1, all CPU, < 30s wall time.
- ~30 backend integration tests in Phase 2, CPU, < 2 min wall time.
- ~10 vLLM-mock integration tests in Phase 3, CPU, < 1 min.
- Phase 1-3 reused on GPU in Phase 4, plus vLLM e2e parity test, < 10 min on one H100.
- Phase 5 measurements as separate driver scripts, not in pytest.

We can ship a usable Phase 1+2 implementation against CPU-only CI,
land the v1 asymmetric format end-to-end in tree, and then validate
on GPU as a separate PR. That sequencing lets reviewers reason about
the format and serde correctness without any GPU pressure.

## Risks and open questions

**Per-head vs per-tensor scales.** Per-head is what the
asymmetric-kv-plumbing branch uses today (`_v_scale_float` is
per-layer right now, but the code path supports per-head). If LMCache
writes per-tensor and vLLM reads per-head (or vice versa), quality
will silently drop on Qwen-class models. The mismatch detector in
Phase 3 must check this dimension explicitly.

**Endianness.** The on-disk format must be little-endian regardless
of host. LMCache currently relies on whatever PyTorch's tensor
serialization does; we explicitly pin `dtype.itemsize`-sized
little-endian writes for the new format and add a unit test that
flips bytes on a big-endian-fake fixture.

**FP8 e5m2 vs e4m3.** Paper uses e4m3. e5m2 has wider range but
worse precision; we leave it as a reserved format flag and don't
implement the path in v1.

**Concurrent reader/writer with mixed formats.** Two processes,
one writing legacy and one writing asymmetric, hitting the same
storage backend. The format-flag check in Phase 1 makes this safe
(reader rejects format it doesn't expect for a given key) but the
test for it lives in Phase 2's "cross-format read" case.

**vLLM API churn.** Asymmetric tuple-API
(`kv_cache_dtype=("auto", "fp8_e4m3")`) lives on a branch, not
upstream. Phase 3 ships behind a feature flag; if upstream takes
the asymmetric API later we need to track whatever signature they
land on.

**KIVI overlap.** The existing `KIVISerializer` stub suggests an
intent to eventually support KIVI's 2-bit channel-aware path.
Asymmetric K16/V8 is orthogonal to that, but we should leave the
plugin dispatch table general enough that adding KIVI later doesn't
require restructuring the asymmetric code we just landed.
