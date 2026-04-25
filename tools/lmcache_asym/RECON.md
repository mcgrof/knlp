# LMCache asymmetric KV — Phase 0 recon

LMCache repo: `/home/mcgrof/devel/lmcache`, branch
`asymmetric-kv-codec` off `origin/dev` at `7b09ac9d`.

## Headline findings

The "single dtype assumption" in `MemoryObjMetadata` is **less
load-bearing than the plan assumed**.  The dataclass already carries
plural `shapes: list[torch.Size]` and `dtypes: list[torch.dtype]`
fields with a `TODO` comment about migrating off the singular
fields.  `MemoryObjMetadata.get_size()` already prefers the plural
form when it's set.  The GPU connector at line 465 of
`gpu_connector/gpu_connectors.py` already uses `get_dtypes()` /
`get_shapes()` accessors and zips the plural lists.

Practically: the asymmetric K16/V8 path can use the existing plural
metadata instead of inventing a new dtype field on `MemoryObj`.
What we still need is a **codec** that knows how to lay out
K-bytes, V-bytes, and V-scales in one contiguous payload that maps
back to a `(shapes, dtypes)` pair plus a side-band scale tensor.

## Three categories of dtype touch-points

### A. Already plural-aware (no change needed in Phase 1-3)

These read `metadata.dtypes[0]` or use `get_dtypes()`/`get_shapes()`
accessors and would naturally pick up multi-dtype state if we
populate `dtypes = [K_dtype, V_dtype]`:

- `lmcache/v1/gpu_connector/gpu_connectors.py:465-470`
  `tmp_buf_dtypes = self.metadata.get_dtypes()` then zips with
  shapes.
- `lmcache/v1/storage_backend/connector/fs_connector.py:249`
  uses `metadata.shapes, metadata.dtypes, metadata.fmt`.
- `lmcache/v1/storage_backend/connector/infinistore_connector.py:107`
  `assert len(metadata.dtypes) == 1` → would need relaxation.
- `lmcache/v1/storage_backend/connector/mooncakestore_connector.py:442`
  reads `metadata.dtypes`.
- `lmcache/v1/storage_backend/connector/redis_connector.py:292,530,670`
  reads `metadata.dtypes`.
- `lmcache/v1/storage_backend/connector/sagemaker_hyperpod_connector.py:483,509`
  reads `metadata.dtypes[0]`.

### B. Singular-only readers (must be widened to support asym)

These read `metadata.dtype` (singular) directly and assume a single
dtype across the whole `MemoryObj`:

- `lmcache/v1/storage_backend/gds_backend.py:693`
  `dtype = memory_obj.metadata.dtype`
- `lmcache/v1/storage_backend/local_disk_backend.py:535`
  `dtype = memory_obj.metadata.dtype`
- `lmcache/v1/storage_backend/nixl_storage_backend.py:757`
  `dtype = metadata.dtype`
- `lmcache/v1/storage_backend/plugins/dax_backend.py:411`
  `dtype = obj_metadata.dtype`
- `lmcache/v1/storage_backend/plugins/rust_raw_block_backend.py:569`
  `dtype=obj.metadata.dtype`
- `lmcache/v1/storage_backend/connector/mock_connector.py:226,274`
  `metadata.dtype`
- `lmcache/v1/server/__main__.py:74`
  `lms_memory_obj.dtype`
- `lmcache/v1/memory_management.py:478`
  `assert metadata.dtype is not None`

For Phase 1-2 (storage-only mode), we leave these alone and route
asymmetric data through the new codec layer above the backends, so
each backend continues to see opaque bytes with a single `dtype`
flag.  For Phase 4 (native_asym passthrough) we need to widen at
least the local-disk and GDS readers, since those are the only
backends in scope for the headline result.

### C. Wire-protocol headers

`lmcache/v1/protocol.py` already has dual support:

- `RemoteMetadata` (lines 99-146) carries `shapes:
  list[torch.Size]`, `dtypes: list[torch.dtype]`, with a packed
  format of `length, fmt, (dtype, shape0..3) * num_groups`.  Plural
  by design.  Asymmetric K/V naturally encodes as `num_groups=2`
  with `dtypes=[K_dtype, V_dtype]`.  V-scales need to ride
  alongside as either an additional group or as side-band metadata.
- `ClientMetaMessage` and `ServerMetaMessage` (lines 152-239) use
  singular dtype + 4D shape.  These are the simple cache server
  path; not in scope for headline result.

`DTYPE_TO_INT` already maps `torch.float8_e4m3fn` → 7 and
`torch.float8_e5m2` → 8 (lines 47-48).  No work needed here.

## Serde plugins

`lmcache/v1/storage_backend/naive_serde/`:

- `serde.py` — abstract `Serializer` / `Deserializer` interface.
  Single-method, single-arg, returns `MemoryObj`.  No `target_layout`
  argument, which means storage-only-dequant vs native-asym needs
  to be encoded into the `MemoryObj` itself or into config seen by
  the serde.
- `naive_serde.py` — pass-through (`ref_count_up()` and return).
- `kivi_serde.py` — stub:

```python
class KIVISerializer(Serializer):
    def serialize(self, memory_obj: MemoryObj) -> MemoryObj:
        # TODO(Yuhan)
        return memory_obj
```

  Confirms the intended extension point but has zero logic.
- `cachegen_*.py` — real implementation, opaque to us.
- `__init__.py:21` — `CreateSerde(serde_type, metadata, config)`
  factory.  Adds new serde types here.

## vLLM connector contract

`lmcache/integration/vllm/lmcache_connector_v1.py:90-99`:

```python
def save_kv_layer(
    self,
    layer_name: str,
    kv_layer: torch.Tensor,        # <-- single tensor, not (K, V)
    attn_metadata: "AttentionMetadata",
    **kwargs,
) -> None:
```

Two more variants: `lmcache_connector_v1_085.py` and
`lmcache_mp_connector_0180.py` — same single-tensor signature.

Native asymmetric passthrough (Phase 4) cannot extend this signature
without touching upstream vLLM.  The passthrough work plumbs
asymmetric state through the *implementation* of `save_kv_layer`
on the LMCache side: the single `kv_layer` tensor is interpreted
based on the `kv_cache_dtype` reported by vLLM's attention layers.
When `kv_cache_dtype = (auto, fp8_e4m3)`, the K and V buffers within
the paged cache are at their respective dtypes and we read each
half accordingly.  The interface stays unchanged; the implementation
becomes dtype-aware.

## What has to change vs. what doesn't

**Add (new files):**

- `lmcache/v1/kv_codec/` — the codec abstraction module.
- `lmcache/v1/kv_codec/asym_k16_v8.py` — first concrete codec.
- `tests/v1/kv_codec/test_*.py` — Phase 1 unit tests (~60).

**Modify (existing files, minimal):**

- `lmcache/v1/storage_backend/naive_serde/__init__.py:21`
  add `"asym_k16_v8_e4m3"` to `CreateSerde`.
- `lmcache/v1/storage_backend/naive_serde/asym_serde.py`
  new file mirroring `naive_serde.py` that wraps the codec.
- `lmcache/v1/config.py`
  add the four orthogonal config knobs from the plan.
- `lmcache/v1/memory_management.py:478`
  relax assertion to allow `dtypes` plural when `dtype` is None.

**Defer to Phase 4:**

- Widening B-category dtype readers (gds, local_disk, nixl, dax,
  rust_raw_block, mock_connector, server).  Phase 1-3 routes
  asymmetric through the codec above the backend, so the backend
  still sees an opaque single-dtype-tagged blob.

## Risk noted in plan, confirmed in repo

The plan's "silent V re-expansion" risk is real because
`Serializer.serialize(memory_obj) -> MemoryObj` does not carry a
`target_runtime_layout` parameter.  If we route asymmetric through
this path naively, the `MemoryObj` returned to the storage backend
has to fully describe its own layout (K + V + scales) so the
deserializer doesn't accidentally produce an FP16 V tensor on the
read path.  The codec layer must own that decision; the serde
shim is a thin wrapper.

The simplest design that prevents silent expansion: the
deserializer returns a `MemoryObj` whose `metadata.dtypes ==
[K_dtype, V_dtype]` and whose payload is K-bytes + V-bytes +
scales.  Whoever consumes that `MemoryObj` (the GPU connector for
runtime, or a higher layer that wants a re-expanded tensor)
explicitly chooses what to do.  No silent dequant inside the serde.
