# LMCache asymmetric KV cache plan: codec, native HBM layout, and tier placement

## Why this exists

The decode paper proves that asymmetric quantization — FP16 on the
keys, FP8 on the values — preserves quality on Qwen-class fragile
models where symmetric FP8 collapses, and that the same structural
asymmetry that makes K quality-fragile also makes K dequant
latency-critical while V dequant is throughput-tolerant. That win
is currently realized only inside the vLLM + FlashInfer execution
path. The moment KV state crosses a serialization boundary —
LMCache page-out to disk, prefix sharing across replicas, NIXL-based
tiering — both K and V get serialized at the same dtype, so we
either pay full FP16 on disk and lose the storage advantage, or
accept the symmetric-FP8 quality cliff that the rest of the paper
exists to avoid.

The honest framing of the win is **not** "NVMe decode acceleration."
Synchronous dense decode from secondary storage needs HBM-class
bandwidth at long context, and NVMe is not magically becoming HBM
because we wrote a nicer enum. The win is that asymmetric K16/V8
reduces LMCache offload size, cache-fill traffic, cache-hit restore
traffic, and storage-budget pressure for prefix reuse and
prefill→decode transfer. Decode still runs after the cache is
restored or prefetched. That is the line between science and
startup-pitch fog, and this plan stays on the science side of it.

The plan also distinguishes three concerns that the prior version
had quietly mashed into one blob:

1. **Storage compression** — what the codec writes to disk.
2. **Native asymmetric HBM layout** — whether vLLM's runtime KV
   cache is restored as `(K16, V8)` or as expanded FP16.
3. **Tier-placement policy** — which physical tier each piece
   lives on (HBM, CPU pinned, NVMe).

These travel together but are independent knobs. The previous draft
of this plan would let a reviewer correctly object that it could
not tell whether we had reduced storage or runtime KV cache. This
draft tells them, in three separate config axes.

The plan also changes the nominal headline result. The most
interesting LMCache contribution is not "asymmetric blobs save 25%
of disk." The interesting result is **K-hot / V-cold split tiering**:
keys stay in CPU pinned memory (fast tier, full FP16 precision),
values live on NVMe (cold tier, FP8). The NVMe read path for cache
restore drops from `K16+V16 = 4 B` per element pair to `V8 = 1 B`,
a 4× reduction in cold-tier read traffic, while K precision and
restore-time K bandwidth are unchanged. Precision-aware quantization
and precision-aware tier placement are the same idea applied at two
levels of the memory hierarchy.

## Table of contents

- [Current LMCache state](#current-lmcache-state)
- [Three knobs, not one](#three-knobs-not-one)
- [Format design](#format-design)
- [Phase 0: repo truth extraction](#phase-0-repo-truth-extraction)
- [Phase 1: KV storage codec](#phase-1-kv-storage-codec)
- [Phase 2: storage-only mode](#phase-2-storage-only-mode)
- [Phase 3: local disk / GDS wiring](#phase-3-local-disk--gds-wiring)
- [Phase 4: native asymmetric passthrough](#phase-4-native-asymmetric-passthrough)
- [Phase 5: K-hot / V-cold split tiering](#phase-5-k-hot--v-cold-split-tiering)
- [Phase 6: paper-relevant evaluation](#phase-6-paper-relevant-evaluation)
- [Test ladder summary](#test-ladder-summary)
- [Hardware progression](#hardware-progression)
- [Risks and open questions](#risks-and-open-questions)
- [Appendix: Claude Code implementation brief](#appendix-claude-code-implementation-brief)

## Current LMCache state

What exists today (head of `LMCache/main`, ~`bb1a6bd`):

`lmcache/v1/protocol.py` already maps `torch.float8_e4m3fn` → 7
and `torch.float8_e5m2` → 8 in `DTYPE_TO_INT`, so the on-disk
type-ID space is ready for FP8. `lmcache/v1/memory_management.py`
defines `MemoryObjMetadata` with a single `dtype` field — that
single-dtype assumption is the central thing the asymmetric path
must break.

The `naive_serde` plugin layer at
`lmcache/v1/storage_backend/naive_serde/` has a clean abstract
interface (`Serializer.serialize`, `Deserializer.deserialize`) and
three implementations: `NaiveSerializer` (pass-through),
`KIVISerializer` (a `TODO(Yuhan)` stub, no logic yet), and
`CacheGenSerializer`. The KIVI stub confirms compression schemes
are meant to live at the serde layer, not be folded into the
storage backend.

The GDS backend at `lmcache/v1/storage_backend/gds_backend.py`
already maps `torch.float8_e4m3fn` and `torch.float8_e5m2`, but
its metadata packer still serializes a single `"kvcache"` tensor
with one dtype, one shape, and one data offset. So the FP8 dtype
IDs are in the codepath but mixed K/V inside a single cache object
is not.

The vLLM LMCache connector API exposes layer save/load around a
single `kv_layer: torch.Tensor`. Native mixed K/V therefore needs
either a wrapper object or surgery on the connector contract; it
is not a pure file-format change. The asymmetric FlashInfer branch
is the real upstream dependency, not anything in vLLM's mainline
attention path.

Tests under `tests/v1/` cover storage backends, memory management,
GPU connector, mem kernels, NIXL, GDS/Weka. They use `pytest` with
fixtures in `tests/conftest.py`. There is no asymmetric test today.

## Three knobs, not one

The previous version of this plan had `kv_format = naive | kivi |
cachegen | asymmetric_k16_v8`. That single flag silently bundled
three independent decisions; reviewers would correctly ask which
one was responsible for any given measurement. This version
separates them:

```
kv_storage_codec     = naive
                     | asym_k16_v8_e4m3
                     | asym_k16_v8_e5m2  (reserved)
                     | cachegen          (existing)
                     | kivi              (existing stub)

kv_runtime_layout    = fp16
                     | storage_only_dequant
                     | native_asym

kv_scale_scope       = per_tensor
                     | per_layer_head
                     | per_page_head     (recommended default for asym)
                     | external

kv_placement_policy  = all_nvme
                     | all_cpu
                     | split_k_cpu_v_nvme       (split-tier headline)
                     | split_k_hot_layers_cpu_v_nvme
```

`storage_only_dequant` means LMCache writes asymmetric on disk and
reads back FP16 for vLLM's existing path. It saves disk but gives
zero HBM capacity. It is a useful fallback for unmodified vLLM and
an honest answer to "what if my vLLM doesn't have the asymmetric
branch."

`native_asym` means LMCache copies K bytes, V bytes, and V scales
into vLLM's native asymmetric paged cache without ever materializing
a full FP16 V tensor. This is the path that actually delivers HBM
capacity savings on top of the disk savings. It depends on the
asymmetric-kv-plumbing branch in vLLM and the asymmetric branch in
FlashInfer.

The connector must refuse to start if `native_asym` is requested
but the vLLM attention layers do not expose separate K/V dtype and
scale metadata. Silent fallback to FP16 here is exactly the bug
this plan exists to prevent.

## Format design

The on-disk format separates K, V, and V-scales into distinct
fields with explicit metadata, rather than a single typed blob.
Per-page-per-head is the recommended scale scope: paged KV chunks
do not naturally respect "whole layer tensor" assumptions, and the
overhead is tiny compared with KV bytes. Per-layer-head and
per-tensor remain available for compatibility with whatever the
asymmetric vLLM branch currently emits.

`EncodedKV` carries:

```
codec_magic
codec_version
model_id        / model_revision_hash
tokenizer_hash
rope_config_hash
attention_backend
kv_layout
page_size
chunk_size
layer_id
kv_head_count
head_dim
k_dtype
v_dtype
scale_dtype
scale_scope                 # per_tensor | per_layer_head | per_page_head | external
scale_shape
payload_offsets             # offsets into the trailing byte stream
payload_crc32c              # or xxhash64; integrity check
```

Concretely, `MemoryObjMetadata` does **not** grow new dtype fields.
Instead the asymmetric path writes a wrapped `EncodedKV` whose
header lives where today's `MemoryObjMetadata` lives, and whose
payload is `K_bytes ‖ V_bytes ‖ V_scales`. The legacy
single-dtype path is unchanged. A reader that does not understand
`EncodedKV.codec_magic` must hard-fail rather than silently
misinterpret bytes.

Cross-model and cross-layout cache poisoning is a real risk;
without `model_id` / `tokenizer_hash` / `rope_config_hash` /
`attention_backend` / `page_size` in the metadata, an old cache
key from a different config can load into a new run and produce
plausibly-shaped tensors with the wrong semantics. Hence the
header is verbose on purpose.

`torch.finfo(dtype).max` gates the scale arithmetic. No hardcoded
448 anywhere; if FP8 e5m2 ever becomes a real path the
finfo-gated code keeps working without a rewrite.

## Phase 0: repo truth extraction

Goal: machine-readable understanding of every site that touches KV
dtype/format/serialization, so later phases can make focused
changes rather than guessing what's load-bearing.

Tasks:

- Run `rg` over the LMCache and (asymmetric) vLLM trees for every
  reference to `MemoryObj`, `MemoryObjMetadata`, `MemoryFormat`,
  `pack_metadata`, `unpack_metadata`, `Serializer`, `Deserializer`,
  `local_backend`, `gds_backend`, `weka_gds_backend`, NIXL
  connectors, and `LMCacheConnectorV1`.
- Identify whether local disk, GDS, and remote serde share one
  serialization path or are separate ones.
- Identify exactly where vLLM's `LMCacheConnectorV1` passes the
  layer KV tensor into LMCache. Note signatures: a wrapped
  `(K, V)` object will need to bridge through here without
  changing the public contract for non-asymmetric users.

Deliverables in `tools/lmcache_asym/`:

- `RECON.md` — narrative summary, < 300 lines, with file:line
  references.
- `touchpoints.json` — structured: `{file, line, symbol, role,
  notes}` per touch-point. Machine-readable so Phase 1 and beyond
  can sanity-check that a change touches the right call sites.

No source changes in this phase.

Tests added: none (this is a read).

## Phase 1: KV storage codec

Goal: introduce a codec abstraction that all asymmetric work
plugs into, with bit-level metadata, no dependency on a GPU.

New module `lmcache/v1/kv_codec/`:

```python
class KVCodec(Protocol):
    def encode(self, layer_kv: LayerKV, metadata: CodecMetadata) -> EncodedKV: ...
    def decode(self, enc: EncodedKV, target_runtime_layout: RuntimeLayout,
               stream: Optional[CudaStream] = None) -> DecodedKV: ...
    def decode_into(self, enc: EncodedKV,
                    k_dest: Tensor, v_dest: Tensor, scale_dest: Tensor,
                    stream: Optional[CudaStream] = None) -> None: ...

class EncodedKV: ...           # header + payload references
class DecodedKV: ...           # K, V, scales, layout description
class CodecError(Exception): ...
class CodecMismatchError(CodecError): ...   # model/tokenizer/rope/etc. mismatch
class CorruptEncodedKVError(CodecError): ...

class AsymK16V8Codec(KVCodec): ...
```

`decode_into` exists separately from `decode` because the
restore-time hot path wants to write directly into pre-allocated
HBM buffers (Phase 4). Allocating a fresh `DecodedKV` per chunk
defeats the asymmetric capacity story.

Phase 1 is CPU-only. Quantization helpers live in
`asym_k16_v8.py` and use pure PyTorch; GPU support is added in
later phases by preserving the device of the input.

Test ladder (CPU, `pytest tests/v1/kv_codec/`):

1. **Scale schema roundtrip.** Build `EncodedKV` with each
   `scale_scope` variant and round-trip through serialize → bytes
   → deserialize. Fields equal exactly.
2. **Scale computation correctness.** Per-tensor, per-head,
   per-page-head scales. Hand-computed expected values, exact
   match. All-zero head: scale is the documented sentinel, not
   `0/0` NaN. Single huge outlier: rest of head dequantizes
   within FP8 noise.
3. **Quant/dequant roundtrip bound.** Random tensors;
   `max(|V - dequant(quant(V))|) / max(|V|) ≤ 6.3%` (FP8 e4m3
   bound around mean values). Saturation at `±finfo(e4m3).max`
   doesn't overflow or wrap. Idempotence: re-quantizing an
   already-quantized tensor is the identity.
4. **Header serializer roundtrip.** Build `EncodedKV` for both
   per-page-per-head and per-layer-head scopes; serialize, parse
   back, fields match. Legacy `MemoryObjMetadata` path unchanged.
5. **Header corruption detection.** Flip the `codec_magic` byte;
   `CorruptEncodedKVError` raised, not silent garbage. Flip the
   `payload_crc32c`; same. Truncate the payload mid-V; same.
6. **Cross-model / cross-config poisoning.** Encode under
   `model_id="qwen-7b"`, decode requesting `model_id="qwen-72b"`;
   `CodecMismatchError` raised. Same for `tokenizer_hash`,
   `rope_config_hash`, `page_size`.
7. **Codec serializer/deserializer pair.** Synthetic `(K, V, scales)`
   triple, encode → decode, K bit-exact (no quantization on K),
   V within FP8 noise bound, scales preserved exactly.
8. **Hypothesis property tests.** Random shapes
   `[B in 1..4, T in 1..256, H in 1..16, D in 32..128]`, dtypes
   `(float16, bfloat16) × float8_e4m3fn`, every roundtrip stays
   within the noise bound, every byte-corruption raises a typed
   error.
9. **Naive parity in degenerate case.** When asymmetric codec is
   given `K_dtype == V_dtype == fp16` and a no-op scale, output
   is byte-equal to `NaiveSerializer` output. Catches accidental
   format drift.

Phase 1 exit: 60+ tests green on CPU, mypy clean, every assertion
has a comment naming the bug it would catch, total wall time
< 30 s.

## Phase 2: storage-only mode

Goal: end-to-end disk path that quantizes V on write and
dequantizes V on read. The simplest deployment mode; useful as a
fallback for unmodified vLLM, and as a stepping stone for Phase 4.

Tasks:

- `kv_runtime_layout = storage_only_dequant`.
- `CreateSerde("asym_k16_v8_e4m3", ...)` returns the new
  serializer/deserializer pair.
- Encode path: split the incoming FP16/BF16 KV tensor into K and V,
  quantize V to FP8 (GPU if available, CPU allowed in tests), write
  K + V + scales as separate fields with the Phase 1 codec.
- Decode path: read K at native dtype, read V as FP8, dequantize V
  back to whatever dtype the existing vLLM path expects.
- Any decoded `DecodedKV` from this mode carries a flag
  `delivered_layout = fp16`. The mode must not claim native HBM
  capacity savings.
- `MemoryObj.physical_size_for_codec(...)` helper so capacity
  accounting is one place.

Test ladder (CPU, run by `pytest tests/v1/storage_backend/`):

1. **Local disk roundtrip with asymmetric codec.**
   In-process `LocalDiskBackend` with a tmpdir, `kv_storage_codec=
   asym_k16_v8_e4m3`, `kv_runtime_layout=storage_only_dequant`.
   Write 100 random `(K, V)` pairs against random keys, read back.
   K bit-exact, V within FP8 noise bound.
2. **Naive backwards compat.** Existing `kv_storage_codec=naive`
   tests unchanged.
3. **Cross-codec read.** Write under `naive`, read under
   `asym_k16_v8_e4m3`; surface `CodecMismatchError`, do not return
   a corrupted tensor.
4. **Capacity accounting under asymmetric.** Fill a small-capacity
   backend until eviction triggers, all entries asymmetric.
   Capacity tracking uses the asymmetric `physical_size`, not the
   legacy single-dtype size. Test for off-by-25% bug.
5. **Storage-only labelling.** A unit test that imports the codec
   in storage-only mode and asserts `delivered_layout == "fp16"`,
   plus a check that the runtime asserts cannot accidentally
   classify this as native HBM savings in any benchmark output.

Phase 2 exit: storage-only path works end-to-end on CPU, capacity
tests pass, format-mismatch errors include observed-vs-expected
codec.

## Phase 3: local disk / GDS wiring

Goal: move from in-process tmpdir to actual storage backends. Stay
narrow; do not start with NIXL, Weka, Mooncake, Redis at once.

Tasks:

- `local_backend.py` and the abstract base remain unchanged (they
  see opaque bytes).
- `gds_backend.py` already maps the FP8 dtype IDs but packs a
  single tensor; extend its metadata packer to round-trip the new
  `EncodedKV` header. This is the only backend touchpoint with
  non-trivial format awareness.
- Skip NIXL/Weka in this phase; come back later behind separate
  feature flags.

Tests:

1. **Local disk backend, asymmetric, end-to-end.** Reuse Phase 2
   tests, but run them against the file-backed local disk path,
   not the tmpdir mock.
2. **GDS roundtrip.** Same correctness expectations as #1, marked
   `@pytest.mark.gpu` and skipped without CUDA + GDS-capable disk.

Phase 3 exit: local disk and GDS roundtrip green; NIXL/Weka left
for later.

## Phase 4: native asymmetric passthrough

Goal: when vLLM is running the asymmetric-kv-plumbing branch and
LMCache is configured for `native_asym`, copy K bytes, V bytes,
and V scales without re-quantizing or materializing FP16 V at any
point.

Tasks:

- `lmcache/v1/gpu_connector.py` already copies KV slabs between
  vLLM and LMCache memory. Add a path that, when vLLM reports
  `kv_cache_dtype = (auto, fp8_e4m3)`, treats the K and V slabs
  as separate buffers with their corresponding dtypes.
- Read each layer's per-head `_v_scale_float` (asymmetric branch
  exposes this on `Attention` layers; verify in Phase 0 recon)
  on cache-miss-fill, write them back into the layer on
  cache-hit-restore.
- Feature detection: at engine startup, inspect a sample
  `Attention` layer to confirm separate K/V dtype and scale
  metadata. If `native_asym` is configured but the runtime does
  not provide them, hard-fail with a clear error message naming
  the missing attribute. No silent fallback.
- Add a runtime assertion (cheap, sample-mode for production) that
  verifies no full-size FP16 V buffer is allocated during the
  restore path. The assertion is the test that the mode's claim
  is honest.

Tests (mocked vLLM on CPU + real GPU smoke):

1. **Mocked vLLM put/get cycle.** `MockVllmAttention` exposes
   `_v_scale_float`, `kv_cache_dtype`, and a synthetic KV slab.
   Single put/get cycle. Scales LMCache writes are exactly the
   scales it reads back; no rescaling drift; no FP16 V allocation
   along the path.
2. **Mismatch detection.** Mock vLLM as symmetric while LMCache
   is `native_asym`; engine-startup error, not first-cache-write
   error. Same in reverse.
3. **Multi-layer plumbing.** 16 layers, each with a different
   per-head scale tensor. Round trip preserves all 16 with
   correct layer↔scale mapping.
4. **GPU smoke.** Single-H100 test that reuses Phase 1–3 tests
   under `@pytest.mark.gpu`. Adds: CPU vs GPU determinism for
   `quantize_v_fp8` (same input → identical FP8 bytes up to
   documented hardware rounding); GDS roundtrip unmasked; vLLM
   end-to-end token parity vs in-memory asymmetric path.
5. **No-FP16-V allocation assertion.** Wraps the restore path
   with `torch.cuda.memory._record_memory_history(...)` (or the
   PyTorch memory snapshot API), runs a hit, asserts no
   allocation matches `(numel(V), torch.float16)`. This is the
   single most important test of this phase; it is the
   difference between the mode being honest and being a
   subtly-disguised storage-only path.

Phase 4 exit: full test suite green on H100, GDS roundtrip
working, vLLM e2e token parity demonstrated on at least one
tolerant model (Llama-3.1-8B) and one fragile model
(Qwen2.5-7B). Memory snapshot test confirms no FP16 V
allocation on the hit path.

## Phase 5: K-hot / V-cold split tiering

Goal: implement the `split_k_cpu_v_nvme` placement policy. K
chunks live in CPU pinned memory (fast tier, full FP16/BF16
precision); V chunks live as FP8 on NVMe (cold tier).

Why this is the headline result: under `native_asym` the
all-NVMe path reads `K16+V8 = 3 B` per element pair (75% of FP16,
1.33× capacity) on a cache restore. Under `split_k_cpu_v_nvme` the
NVMe path reads `V8 = 1 B` per element pair (25% of FP16; 4×
reduction in cold-tier read traffic), while K bytes flow CPU→GPU
at roughly an order of magnitude more bandwidth than NVMe. The
serial-critical, quality-fragile half of attention sits in the
fast tier; the bandwidth-tolerant, quality-tolerant half sits in
the cold tier.

Architecture: each logical KV chunk stores three physical objects:

```
cache_key/layer_<L>/chunk_<C>/K.bin     # FP16/BF16, fast tier
cache_key/layer_<L>/chunk_<C>/V.fp8     # FP8 e4m3, cold tier
cache_key/layer_<L>/chunk_<C>/meta.json # codec header, scales, hashes
```

Restore pipeline:

```
lookup prefix chunks
for each layer L:
    issue async K CPU->GPU copy from pinned memory
    issue async V NVMe->GPU read (GDS when available)
    restore V scales (small, CPU or GPU)
    if native_asym:
        decode_into the live asymmetric paged cache
    else (storage_only):
        dequantize V8->V16 on GPU into FP16 paged cache
    pipeline layer L+1 disk read with layer L decode_into
```

Fallback: when CPU pinned memory budget is exceeded, demote K to
NVMe (degrades to all-NVMe asymmetric) or reject according to
config; never silently lose precision.

Tests:

1. **Split-tier byte accounting.** Encode 32 chunks across 16
   layers; verify K bytes land in pinned memory and V bytes land
   on disk; verify NVMe read on a hit equals
   `N_chunks × N_layers × chunk_size × kv_heads × head_dim × 1 byte`
   to within rounding/overhead.
2. **Pinned memory pressure.** Configure tiny CPU pinned budget;
   issue more chunks than fit; verify configured fallback behavior.
3. **Pipeline overlap.** With `--profile`, verify layer L disk
   read and layer L-1 GPU decode_into overlap (NVMe and GPU
   compute are not serialized).
4. **End-to-end: cache hit produces token-identical generation
   to in-memory asymmetric** on Qwen2.5-7B and Llama-3.1-8B for
   a deterministic decode setting.

## Phase 6: paper-relevant evaluation

Goal: numbers that survive review. Decode tokens/sec is **not**
the headline metric for a cache-hit path; cache-hit TTFT, restore
latency, and bytes-per-tier are.

Workloads:

- **W1: long system prompt + short user turns.** Repeated long
  prefix, many short queries. Maximizes prefix-cache hit rate.
- **W2: RAG prefix reuse.** Shared 8K / 16K / 32K context across
  queries.
- **W3: multi-replica cache sharing.** One writer, one reader,
  same NVMe / shared filesystem.
- **W4: adversarial mixed hit/miss.** 30% / 60% / 90% hit rate
  schedules.
- **W5: fixed NVMe budget.** Measure eviction behavior and cache
  entries per 1 TB.

Baselines:

```
B0  no LMCache, recompute prefix every query
B1  LMCache FP16, all-NVMe
B2  LMCache asymmetric K16/V8, storage_only_dequant, all-NVMe
B3  LMCache asymmetric K16/V8, native_asym, all-NVMe
B4  LMCache asymmetric K16/V8, native_asym, split_k_cpu_v_nvme   # headline
B5  LMCache CacheGen (existing baseline) for sanity
```

Models:

```
Qwen2.5-7B-Instruct      # fragile-key, mandatory
Llama-3.1-8B             # tolerant control
Mistral-7B               # tolerant, different GQA
```

Metrics, per (workload, baseline, model):

- encoded bytes per chunk
- actual bytes read from NVMe per cache hit
- actual bytes copied CPU→GPU per cache hit
- temporary FP16 allocation size on the restore path
- cache-hit TTFT: p50 / p95 / p99
- restore latency per layer
- tokens/sec after restore (sanity, not headline)
- WikiText-2 PPL on Qwen2.5-7B (must match in-memory asymmetric)
- NIAH multikey-3 at 16K and 32K on Qwen2.5-7B and Llama-3.1-8B
- cache entries per 1 TB at fixed quality

The killer table:

| Mode | Disk bytes vs FP16 | CPU bytes vs FP16 | Native asym HBM? | Useful claim |
|---|---:|---:|---:|---|
| FP16 all-NVMe | 1.00× | 0 | n/a | baseline |
| K16/V8 all-NVMe storage-only | 0.75× | 0 | no | disk capacity 1.33× |
| K16/V8 all-NVMe native asym | 0.75× | 0 | yes | disk + HBM capacity 1.33× |
| K16 CPU / V8 NVMe split-tier | **0.25× NVMe read** | 0.50× CPU | yes | **4× lower cold-tier read traffic; lower TTFT under cold-tier pressure** |
| storage-only K16/V8 (fallback) | 0.75× | 0 | no | disk-only win |

Reviewers will ask whether we reduced storage or runtime KV cache.
This table answers that without anyone needing to read a paragraph.

## Test ladder summary

The pyramid:

- ~60 unit tests in Phase 1, all CPU, < 30 s wall time.
- ~30 backend integration tests in Phase 2, CPU, < 2 min.
- ~10 GDS integration tests in Phase 3 (gated GPU), < 5 min on H100.
- ~10 mocked-vLLM + GPU smoke tests in Phase 4, including the
  no-FP16-V-allocation memory-snapshot test, < 10 min on H100.
- Phase 5 split-tier tests reuse the Phase 1–4 fixtures with new
  placement policies. End-to-end token parity on Qwen2.5-7B is
  the gate.
- Phase 6 measurement scripts run as standalone drivers, not
  pytest cases.

We can land Phase 1+2 against CPU-only CI, ship the storage-only
asymmetric format end-to-end, then land `native_asym` and
split-tier as separate PRs once the GPU lane is healthy.

## Hardware progression

| Phase | Hardware | Why |
|---|---|---|
| Codec, format, metadata tests | any x86 box, no GPU | schema, corruption, byte accounting |
| NVMe storage microbench | x86 + Gen4/Gen5 NVMe | `fio` baseline, O_DIRECT alignment |
| `native_asym` vLLM smoke | 1× H100 or H200 | FP8 + FlashInfer asymmetric branch |
| GDS path | 1× H100/H200 + GDS-capable NVMe | direct-GPU I/O + alignment bugs |
| Multi-replica + split-tier | 2× H100/H200 (one node or two) | prefix sharing, K/V tier orchestration |
| Optional cross-vendor | MI300X | confirm ROCm path matters; deferrable |
| Optional Blackwell follow-up | B200 | only if FP4/MXFP4 work materializes |

A100 is fine for compatibility and storage-only tests but should
not be the primary lane for the FP8 / native-asym story; that lives
on H100/H200 to match the paper's other FlashInfer results.

## Risks and open questions

**Silent V re-expansion.** The dominant subtle bug. If LMCache
receives a materialized FP16 KV tensor from the connector and only
quantizes V on the disk write, the disk wins but the HBM win
disappears. Worse, if the read path dequantizes V before handing it
back, even unaware reviewers will notice when memory accounting
doesn't match the claim. The Phase 4 memory-snapshot test that
asserts no FP16-V allocation on the hit path is the gate.

**CPU-emulated FP8 is not authoritative.** CPU emulation is good
enough to test the codec; it is *not* the authoritative source of
FP8 V bytes for production. The authoritative path is: when
vLLM/FlashInfer already has V in FP8, LMCache copies those bytes
and their scales without re-quantizing. Storage-only mode is the
only mode allowed to re-quantize, and it is labelled accordingly.

**Scale scope mismatch.** Per-tensor on disk + per-head at runtime
(or vice versa) silently produces the wrong dequantization on
fragile-key models. The mismatch detector in Phase 4 must compare
this dimension explicitly, not assume it.

**vLLM connector contract is single-tensor.** The public
`LMCacheConnectorV1` API saves a single `kv_layer: torch.Tensor`.
Native mixed K/V plumbing through that interface is branch-specific
work, not a generic LMCache feature. Phase 4 should ship behind a
feature flag and an explicit "this requires the asymmetric vLLM
branch" check.

**Cross-model cache poisoning.** Without `model_id`,
`tokenizer_hash`, `rope_config_hash`, `attention_backend`,
`page_size` in the codec header, an old cache entry from a
different config can load into a new run, producing
plausibly-shaped tensors with the wrong semantics. Hence verbose
metadata up front.

**Endianness.** Pin all integer fields little-endian regardless of
host. Do not rely on whatever PyTorch's tensor serialization does;
explicit `dtype.itemsize`-sized writes plus a unit test on a
big-endian-fake fixture.

**FP8 e5m2 vs e4m3.** Paper uses e4m3. e5m2 has wider range but
worse precision; reserved as a format flag, not implemented in v1.
`torch.finfo(dtype).max` keeps the math right when it does land.

**Concurrent reader/writer with mixed formats.** Two processes,
one writing legacy and one writing asymmetric, hitting the same
backend. The codec-magic check in Phase 1 makes this safe per-key
(reader rejects format it doesn't expect). The cross-format-read
test lives in Phase 2.

**KIVI overlap.** The existing `KIVISerializer` stub suggests
intent to land KIVI eventually. Keeping codec dispatch general
enough that adding KIVI later doesn't restructure the asymmetric
code is the simplest hedge; do not bake assumptions about codec
count or naming into the format header.

**Configuration sprawl.** Four orthogonal axes
(`kv_storage_codec`, `kv_runtime_layout`, `kv_scale_scope`,
`kv_placement_policy`) is a lot, but each one represents a real
decision. Bundling them back into one `kv_format` flag is what we
just stopped doing; resist the temptation to re-collapse them.

## Appendix: Claude Code implementation brief

The brief below is what to paste into a fresh Claude Code session
to actually build this. It is intentionally blunt and includes all
acceptance criteria so the implementation cannot drift quietly.

```text
You are implementing an LMCache extension for asymmetric KV-cache
storage and split-tier placement.

Goal:
Build and test a storage codec plus optional native vLLM/FlashInfer
passthrough for asymmetric K16/V8 KV cache:
- K remains FP16/BF16.
- V is FP8 e4m3.
- V scales are preserved.
- The implementation must support both storage-only and native
  asymmetric runtime modes.
- Do not silently dequantize V to FP16 unless the selected runtime
  mode is storage_only_dequant.

Repository tasks:

1. Recon
- Create tools/lmcache_asym/RECON.md.
- Create tools/lmcache_asym/touchpoints.json.
- Use rg to find all dtype, MemoryObj, MemoryObjMetadata,
  MemoryFormat, safetensors, pack_metadata, unpack_metadata,
  Serializer, Deserializer, local disk, GDS, NIXL, and vLLM
  connector touch points.
- Identify whether local disk, GDS, and remote serde share one
  serialization path or separate ones.
- Identify where vLLM LMCacheConnectorV1 passes the layer KV
  tensor into LMCache.
- Do not change source code in this step.

2. Codec interface
- Add a new module, preferably lmcache/v1/kv_codec/.
- Implement:
    KVCodec
    EncodedKV
    AsymK16V8Codec
    CodecError
    CodecMismatchError
    CorruptEncodedKVError
- Add config:
    kv_storage_codec: naive | asym_k16_v8_e4m3
    kv_runtime_layout: fp16 | storage_only_dequant | native_asym
    kv_scale_scope: per_tensor | per_layer_head | per_page_head
    kv_placement_policy: all_nvme | all_cpu | split_k_cpu_v_nvme
- Preserve existing defaults.

3. Format
- For asymmetric codec, store K payload, V payload, and V scales
  as separate fields.
- Metadata must include:
    codec version
    model id/revision hash if available
    tokenizer hash if available
    rope config hash if available
    layer id
    chunk id
    chunk size
    page size
    kv layout
    k dtype
    v dtype
    scale dtype
    scale scope
    scale shape
    payload offsets
    checksums
- Never use hardcoded FP8 max values. Use torch.finfo(dtype).max
  or the backend's actual scale convention.
- Add corruption detection and explicit errors.

4. Storage-only mode
- Input: existing FP16/BF16 KV tensor from LMCache/vLLM.
- Split into K and V.
- Quantize V to FP8 on GPU if CUDA is available; otherwise CPU
  path is allowed only for tests.
- Store K as native dtype, V as FP8, scales as FP32 or FP16
  according to config.
- On load, dequantize V back to the dtype expected by the
  existing vLLM path.
- This mode must report that it is storage-only and does not
  provide native HBM capacity savings.

5. Native asymmetric mode
- Detect whether the vLLM/FlashInfer branch exposes separate K/V
  dtypes and V scale tensors.
- If not detected, fail at startup with a clear error.
- If detected, copy K bytes, V bytes, and V scales without
  re-quantizing.
- On load, restore K, V, and scales into the native asymmetric
  paged cache.
- Add assertions or tracing to prove no full-size temporary FP16
  V buffer is allocated.

6. Split-tier placement
- Implement split_k_cpu_v_nvme policy.
- Store K chunks in CPU pinned memory when capacity allows.
- Store V chunks as FP8 on local disk/NVMe.
- On retrieve:
    issue K CPU->GPU copy and V disk->GPU/CPU read concurrently
    restore scales
    pipeline per layer
- Add fallback when CPU pinned memory budget is exceeded:
    demote K to NVMe or reject according to config.

7. Tests
CPU tests:
- scale schema roundtrip
- zero tensor
- outlier tensor
- random tensors over shape grid
- metadata corruption
- cache key/config mismatch
- storage-only encode/decode tolerance
- legacy naive compatibility
- capacity accounting

GPU tests on H100/H200:
- storage-only encode/decode
- native asymmetric passthrough
- no temporary FP16 V allocation in native mode
- vLLM Qwen2.5-7B and Llama-3.1-8B 4-prompt hit/miss test
- GDS/local NVMe roundtrip if available

8. Benchmarks
Create:
- tools/lmcache_asym/bench_codec_bytes.py
- tools/lmcache_asym/bench_nvme_restore.py
- tools/lmcache_asym/bench_vllm_hit_ttft.py
- tools/lmcache_asym/bench_split_tier.py
- tools/lmcache_asym/plot_results.py

Metrics:
- encoded bytes
- disk bytes read/write
- CPU bytes allocated
- pinned memory bytes
- GPU temporary bytes
- encode latency
- disk read latency
- GPU copy latency
- dequant latency
- end-to-end cache-hit TTFT
- p50/p95/p99 latency
- output token agreement
- WikiText-2 PPL for Qwen2.5-7B
- NIAH 16K/32K for Qwen2.5-7B and Llama-3.1-8B

9. Acceptance criteria
- Asym all-NVMe encoded size is within 1% of 0.75x FP16, excluding
  metadata.
- Split KCPU/VNVMe reads roughly 0.25x FP16 KV bytes from NVMe for
  cache-hit restore.
- Native asymmetric mode restores without materializing full V as
  FP16.
- Qwen2.5-7B quality matches the in-memory asymmetric path within
  measurement noise.
- Storage-only mode is labeled honestly and does not claim HBM
  capacity savings.
- All new errors are explicit: no silent fallback to FP16 unless
  config says so.
- Produce a results markdown file:
    tools/lmcache_asym/RESULTS.md
  with tables suitable for copying into the memory-decode paper.
```

The clean paper contribution: the same K/V asymmetry that makes
FP16-K/FP8-V the right *quantization* choice for fragile-key models
is also the right *tier-placement* choice for fragile-key models,
applied at a different level of the memory hierarchy.
