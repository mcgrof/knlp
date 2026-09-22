# Transform-aware cache identity for LMCache token dropping

## Problem

LMCache's token-dropping SDK can retrieve a request's KV tensors, let a caller
replace them, and store the replacement for real vLLM decode. That is enough to
run SnapKV, but it is not enough to make a SnapKV artifact safe to discover and
share in a mixed deployment.

The multiprocess key currently carries the model, world and worker identity,
token IDs, token range, and `cache_salt`. The stored `ObjectKey` carries a
chained token-chunk hash, model, rank, object group, and `cache_salt`. Neither
key names the transform that produced the tensor.

SnapKV demonstrates why token identity is insufficient. Its retained KV:

- depends on the recent query tensors used to score past positions;
- contains vectors computed from the complete original prefix;
- retains a selected set of original positions; and
- rerotates retained keys into new dense positions before decode.

The returned token IDs describe the compacted logical sequence. They do not
prove which source prefix, query, selection policy, positions, or rerotation
produced its tensors.

## Present behavior

Two vLLM instances attached to the same LMCache service share the same default
namespace when model, layout, tokens, and `cache_salt` match. LMCache does not
know that one instance permits only ordinary KV while another creates SnapKV.

The ordinary full prompt does not normally hit the newly stored compacted
entry: the SDK stores that entry under the returned, shorter token sequence.
The original unmodified full-prefix entry may remain separately shareable.
The unsafe case is a later lookup whose token-derived key equals the compacted
sequence. That lookup can come from a literal short prompt, another source
prefix that selected the same IDs, a different query, or a different transform.
Those tensors need not be interchangeable.

`cache_salt` can isolate an experiment as a temporary defense, but it is a
per-user isolation and quota field. Overloading it with transform semantics
prevents deliberate sharing, does not define compatibility, and leaves no
machine-checkable provenance.

## Proposed contract

### 1. Return a structured transformed artifact

Extend the modifier result without breaking the current tuple form:

```python
@dataclass(frozen=True)
class CacheTransformManifest:
    algorithm: str                 # "snapkv"
    algorithm_version: str         # implementation or schema version
    parameters_digest: bytes       # window, ratio, pooling, kernel, etc.
    source_prefix_digest: bytes    # full source tokens and original length
    retained_positions_digest: bytes
    position_map_digest: bytes     # old -> new positions / rerotation
    query_digest: bytes | None     # required when selection is query-dependent
    payload_format_digest: bytes   # model, layout, dtype, RoPE contract
    sharing_scope: Literal["request", "query", "prefix", "global"]

@dataclass(frozen=True)
class ModifiedCache:
    kv: torch.Tensor
    logical_token_ids: Sequence[int]
    manifest: CacheTransformManifest
```

Keep accepting `(kv, token_ids)` during migration, but assign that legacy form
`sharing_scope="request"`. An unlabelled transform must never become a shared
artifact by accident.

### 2. Add a cache variant to storage identity

Compute a canonical `transform_digest` from the manifest and carry it through:

1. `LMCacheRequestStream.modify_kv()` and `update()`;
2. `LMCacheSDKContext._create_key()`;
3. `IPCCacheServerKey` and its msgspec wire representation;
4. token-chunk hashing or `ObjectKey`;
5. coordinator, directory, event, serializer, and L2 adapter paths; and
6. vLLM connector lookup metadata.

Keep tenant isolation (`cache_salt`) separate from artifact compatibility
(`transform_digest`). The ordinary KV format uses an empty variant so existing
unsalted key serialization remains compatible. A transformed lookup must ask
for the exact variant; an ordinary lookup must not fall back to transformed KV.

### 3. Make sharing scope enforceable

- `request`: keep the result private to its producing request.
- `query`: share only when the query digest and complete manifest match.
- `prefix`: allow sharing when the transform is query-independent and binds the
  complete source-prefix digest.
- `global`: reserve for transforms whose implementation proves that token and
  format identity completely determine the payload.

SnapKV should default to `query`, because its selection uses recent-window query
tensors. Operators may choose `request` until the query digest and manifest are
fully propagated.

## Transition plan

1. Add the manifest and request-local default to the SDK.
2. Add `transform_digest` to multiprocess and object keys with an empty default.
3. Teach L1 and L2 serializers to preserve the field and reject unknown
   transformed variants instead of silently treating them as ordinary KV.
4. Propagate requested variants from vLLM metadata and expose hit/miss counters
   labelled by transform.
5. Enable query-scoped sharing for SnapKV after the negative test matrix passes.

## Required tests

| Producer | Consumer | Expected result |
|---|---|---|
| Ordinary KV | Ordinary, same prefix | Hit |
| SnapKV | Ordinary, retained IDs as literal prompt | Miss |
| SnapKV query A | SnapKV query A, same parameters | Hit |
| SnapKV query A | SnapKV query B | Miss |
| SnapKV ratio 0.5 | SnapKV ratio 0.25 | Miss |
| SnapKV source A | SnapKV source B, same retained IDs | Miss |
| SnapKV implementation v1 | implementation v2 | Miss unless compatible |
| Any transform | Different model/layout/RoPE/dtype | Miss |

Run the matrix through in-process L1, multiprocess shared memory, and every
enabled L2 adapter. Include restart/reload tests so serialized identity cannot
lose the variant.

## Scope of the KNLP reproduction

`make defconfig-lmcache-snapkv && make` validates the current real data path:
vLLM prefill, SDK retrieval of KV and query tensors, SnapKV selection and key
rerotation, LMCache store, and vLLM decode. It compares answer accuracy and
decode throughput with a clean baseline and records the LMCache revision.

The harness deliberately does not claim mixed-instance safety. Add the test
matrix above when LMCache carries transform identity end to end.
