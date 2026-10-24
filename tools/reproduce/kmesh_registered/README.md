<!-- SPDX-License-Identifier: MIT -->
# Registered serial model replay

This bounded replay moves the TokenDance-style serial repair baseline from
in-process retrieval to completed native span restores in a separate cache
process. It uses the actual Qwen model, rotary table, engine cache allocations
and live request slots. The leader publishes four partial spans containing 416
tokens. Four siblings restore them at shifted positions and consume them through
the existing serial CacheBlend repair.

LocalCPU lookup and storage remain scheduling controls. This duplicates storage;
elapsed time cannot establish serving latency, capacity savings or a DMA-BUF
benefit. The ordinary multiprocess serving profile remains disabled. This replay
qualifies CPU-L1 model consumption before changing the storage route.

Use the exact source commits in `profile.json` with the existing
[RTX runtime and model pins](../tokendance/profile.json). Install the matching
native extensions and generated protobuf modules as described by the component
projects. Keep source checkouts clean. Generate the frozen trace using the
[TokenDance reproducer](../tokendance/README.md), which also produces the
independently verified serial reference used by the comparison.

Run the lower-level gates first with `defconfig-kmesh-registered-inputs-cpu`
and `defconfig-kmesh-registered-inputs-cuda`. These use the existing
`kmesh-contracts` and `kmesh-native` entry points and their caller-selected source
and external output paths. Both pin the model replay's LMCache source. The
CUDA gate covers separate and fused NHD/HND views, exact partial spans,
synthetic layer input/writeback, physical drain and exporter retirement. Its
synthetic transform does not replace the real model replay below.

```sh
make defconfig-kmesh-registered-model-cuda
make kmesh-registered-plan
```

Set `KMESH_REGISTERED_VLLM_SOURCE`, `KMESH_REGISTERED_LMCACHE_SOURCE`,
`KMESH_REGISTERED_WHEEL_PACKAGE` and `KMESH_REGISTERED_MODEL_DIR` to the pinned
local inputs. Set `KMESH_REGISTERED_TRACE` and `KMESH_REGISTERED_MANIFEST` to the
generated trace files. Set `KMESH_REGISTERED_OUTPUT` to a fresh external capture
directory. First set `KMESH_REGISTERED_ROUTE=reference` and run
`make kmesh-registered` with the matching Python interpreter selected through
`KMESH_REGISTERED_PY`. This captures the in-process reference on the new source.
Verify it against the independently verified historical serial result. Then use
`KMESH_REGISTERED_ROUTE=registered` with another fresh output directory and run
the model again. Both runs use the same source checkout, workload and runtime.

For the first comparison, set `KMESH_REGISTERED_REFERENCE` to the historical
bundle's `arms/01-serial_oracle/artifacts` directory and
`KMESH_REGISTERED_VERIFICATION` to its independently generated `verified.json`.
Run `make kmesh-registered-verify` to compare saved CPU tensors. For the paired
comparison, point `REFERENCE` to the new reference capture and `VERIFICATION`
to that capture's passing `comparison.json`. The pair also requires identical
actual repair selections at every layer. The policy requires
bitwise K/V, identical selected spans and retrieval masks, logits within the
existing serial comparison tolerance, and identical generated tokens. A failed
comparison is retained as evidence; it must not be relabeled a passing run.

All captures remain outside the source trees. The service process is owned by
the replay and exits before engine allocations can be freed. Checked retirement
must cover all imported allocations; an uncertain reply requires actual importer
exit. This runner does not change kernels, drivers or storage media.

## Model failure controls

Select a new `KMESH_REGISTERED_OUTPUT` directory and set
`KMESH_REGISTERED_ORDINARY_RESULT` to the independently verified historical
bundle's `arms/00-ordinary/artifacts/result.json`. Set
`KMESH_REGISTERED_ORDINARY_VERIFICATION` to that bundle's `verified.json`, then
run `make kmesh-registered-controls` with the same pinned source and runtime.
Each control runs in a fresh process and records its actual operations:

- Omit one registered handle while the source remains intact. The model must
  consume the other three spans and recompute all missing and private positions.
- Corrupt an actual CPU-L1 tensor, or delete its actual L1 key before lookup.
  Both must refuse before native scatter and complete through fallback.
- Raise after a genuine native restore enqueue. Successful physical drain must
  permit full-prompt invalidation and recomputation.
- Refuse one service drain, then require an actual later drain before fallback.
- Keep refusing service drain, or refuse worker drain after a real layer gather.
  Neither may mark blocks reusable for fallback or return a completed request.
  Disarming the control must allow real cleanup and checked retirement.

A revoked lease disables this bounded replay backend; later drained requests
recompute. This is an explicit availability limit. Controls compare completed
generated tokens with the ordinary baseline. They do not replace the successful
pair's raw K/V, logit and selection comparison.

## Matched storage routes

The storage comparison asks whether direct reads into a GPU arena improve this
same serial repair workload compared with reads into pinned host memory. Select
`defconfig-kmesh-registered-storage-cuda` and inspect `make kmesh-registered-plan`.
This is a separate experimental profile, `storage-profile.json`; it requires the
eager source-arena component commits and native extensions named there. First
qualify those inputs with `defconfig-kmesh-registered-arenas-cpu` and
`defconfig-kmesh-registered-arenas-cuda`. The ordinary multiprocess serving guard
remains closed. The comparison is implemented but has no accepted GPU result in
this document.

Use the source, model, trace, historical reference and independent verification
paths described above. Choose a fresh external `KMESH_REGISTERED_OUTPUT`, then
run `make kmesh-registered-storage-pair` with the matching interpreter. The
default runs seven pairs, alternating which storage route runs first. Set
`KMESH_REGISTERED_TRIALS` to choose another number of pairs. Each replay owns a
fresh regular backing file in its capture directory. It uses the existing raw
block adapter over that file; it does not initialize a raw device. The filesystem
must be XFS or ext4 on an NVMe namespace and support the selected registered
`io_uring` path. Both routes use the same filesystem and storage geometry.
Both fully initialize and sync the fresh backing file before native open.
Preallocation alone leaves unwritten extents, which the strict GPU route refuses.

The pinned model has 24 layers, two logical K/V planes, two KV heads, 64 elements
per head and two bytes per BF16 element. A padded 256-token object is therefore
`256 × 24 × 2 × 2 × 64 × 2 = 3 MiB`. Four such objects supply the four partial
spans. Both routes reserve a 64 MiB canonical arena and a 1 MiB metadata area.
The file has eight slots of 3 MiB plus a 4 KiB header each, providing one source
set and an equally sized reserve without overwrite. Both use one worker for
each store, lookup and load queue, an I/O queue depth of 32, and a 128 KiB maximum
transfer. These are fixed experiment settings, not tuned production defaults.

The staged arm needs a locked-memory allowance larger than its 64 MiB host
arena. Native I/O workers can register the same arena again, and metadata also
uses registered memory. Configure the process's soft and hard
`RLIMIT_MEMLOCK` before running; for example, `ulimit -l 1048576` permits 1 GiB
when the hard limit allows it. Use the same allowance for both arms. The pair
refuses a limit smaller than one arena before starting the model and records
the actual limits. Passing this minimum check does not reserve memory or prove
that every native worker registration will succeed.

External retention is explicitly enabled in both arms. Before each child, the
runner confirms that the exact four immutable source keys are visible in the
NVMe tier, releases its lookup locks, waits for outstanding L1 leases, and evicts
those keys from external L1. This deliberate cold-read control measures the
transfer path. It does not repeat the retention ablation: retention is the
mechanism that avoids these repeated reads during normal reuse, whereas
singleflight coalesces overlapping loads. SSD caches can remain warm.

Before collecting timing samples, each route must pass a separate real model
replay that fills its entire freed canonical arena with a BF16 NaN pattern and
drains that fill before loading from NVMe. Freed allocations can retain old KV
bytes; this control prevents a numerically correct stale buffer from passing as
new data arrival. It poisons only an exclusively reserved, otherwise free arena,
never a live immutable source. An unresolved device operation retains its
reservation and physical leases until the owned process exits. Timing runs omit
this conditioning and are admitted only after both qualified routes pass.

Every measured child must show native registered-read submission and completion
for its exact source keys and byte counts, followed by bitwise K/V, matching
repair selections and retrieval masks, the existing logit tolerance, and exact
generated tokens against the same-source serial reference. The staged route
uses classic registered host buffers; the direct route uses a Vulkan/RM-exported
GPU DMA-BUF arena. Checked physical drain and arena shutdown are mandatory.

`paired.json` binds both data-arrival qualifications and every numerical
comparison. It reports per-pair differences between child medians for restore,
actual storage calls, full integrity hashing and consuming-model work. Restore
contains nested storage and hash costs; these columns must not be added. GPU
source integrity checking still copies the complete object to the host for
SHA-256, and the direct route copies canonical bytes into private engine slots
before repair. Reference storage, model instrumentation and raw tensor capture
also remain active. The comparison does not establish production serving speed,
total memory savings or elimination of all host transfers.

## Registered layer-major model group

The group consumer asks whether several children can safely consume the same
immutable registered sources while keeping their repair buffers and engine
destinations private. It advances all children through one layer before moving
to the next layer. Model tensor operations stay request-local; this first group
comparison does not batch query/key rotation or selection differences. Cached
keys were already rotated during native restore and must not be rotated again.

Select `defconfig-kmesh-registered-group-model-cuda`. Its `group-profile.json`
pins the complete-group owner implementation and accepts serial and layer-major
execution with the same model, trace and runtime envelope. First pass
`defconfig-kmesh-registered-groups-cpu` and
`defconfig-kmesh-registered-groups-cuda`. The latter checks simultaneous native
inputs and synthetic private writeback, which do not replace this real model
comparison. The model group is implemented but awaits GPU qualification here.

Set the input paths described above, using the source commits in the group
profile. Set `KMESH_REGISTERED_MODEL_EXECUTION=serial` and
`KMESH_REGISTERED_ROUTE=reference` to capture the same-source serial reference.
Compare it with the independently verified historical serial result using
`make kmesh-registered-verify`. Then use fresh external captures for
`ROUTE=registered`, first with `MODEL_EXECUTION=serial`, then with
`MODEL_EXECUTION=collective-lockstep`. Compare both with that new reference and
its passing `comparison.json`, retaining bitwise K/V, identical spans, masks and
actual layer selections, the original logit tolerance and exact tokens.

The group validates every child's full block table before any native restore.
All placements complete before the group opens any model input. Exact request,
generation, tokens and spans bind before the first CacheBlend session starts.
Cleanup attempts every member and requires actual physical drain before the
adapter can publish fallback or destination reuse. A callback failure must
escape request-local error containment.

The first model group uses CPU-L1 canonical storage and the existing
`collective_lockstep_v1` engine profile. Combined group/storage timing and source
fault injection are refused by this entry point until separately integrated.
The ordinary multiprocess serving guard stays closed. Duplicated reference
storage and model capture remain; matching this bounded approximate-repair
baseline does not establish equivalence to full recomputation, production
serving speed or total memory savings.

For a member-failure control, retain the group model configuration and choose
another fresh `KMESH_REGISTERED_OUTPUT`. Point `REFERENCE` and `VERIFICATION`
to the same-source serial reference and its passing `comparison.json`, and set
the verified ordinary result paths described under model failure controls.
Run `make kmesh-registered-group-failure`. The inherited model instrumentation
fails the second child after its first repaired layer. The other three must
finish with serial-reference tokens; the failed child must drain, invalidate
its full prompt and complete through ordinary recomputation. The original
request/plan evidence enforces the causal chain through the empty retry plan
and actual request completion. All four native group sources must close with
no physical work pending. This token/cleanup control has mixed repaired and
recomputed K/V and is separate from the complete raw-tensor success comparison.
