# Native cache-layout and registered-copy qualification

Before extending shared-state recovery across processes, check the native
executor's memory layout and positional transformations. These controls use
synthetic BF16 payloads with head-major and token-major paged storage,
separate K/V planes and current fused content-width formats.

Select `paged-cache` for that original gate. The `registered-copy` and
`rotary-transfers` profiles supply the later copy-lifetime prerequisites.
Each pins an independent LMCache source and case set:

| Profile | LMCache commit | Required cases |
| --- | --- | ---: |
| `paged-cache` | `e5dded61d56e220bcd98c95e6a8bb13148a989cf` | 32 |
| `registered-copy` | `6776516b3efd156df1fa014bc6f346adce374379` | 3 |
| `rotary-transfers` | `062c4ea6487f86c91e30b3c98ac7e4e520f19106` | 6 |
| `ordered-registration` | `27e256e1aae7090c4ccfc5eea587a7fbf93fddae` | 6 |
| `raw-ipc-lifetime` | `cb31561f6877f16d50e74891d3dc9bf00f888532` | 3 |
| `raw-ipc-outcomes` | `b08f4c339c73c23d2ba46b7e2c17e880c34a7b19` | 6 |
| `native-spans` | `daa0b4ddc8551d29d21ff4ac51f913eb29dfed9a` | 24 |
| `interior-spans` | `e23d4acd29420a0258a9666b98d9a10fdec12b53` | 36 |
| `managed-raw-service` | `d3a4b9a6dd2a8edb649d24d48cb6e6b46b10bd9e` | 10 |
| `managed-raw-context` | `304df82413dbb9d9d12de145963b5451d1e9eeae` | 9 |
| `owned-raw-copy` | `ca65f2820a218a84ecba4d452c47da2c1ac1c4f5` | 4 |

The profiles pin LMCache source and native binary hashes. Reuse a matching
prepared CUDA environment; this runner does not install or rebuild packages.
Binary hashes describe the reference build, not a portable ABI. To qualify a
different build, compile the pinned LMCache source using its build instructions
and review a separate copy of `profile.json` with the resulting binary hashes.
Keep the source hashes, selected cases and numerical controls intact. Record
that profile and build provenance with the new capture; it is a fresh
qualification, not a relabeling of the reference build's receipt.
The paged-cache grouped controls reach both the public flat executor and public
registration/retrieve handlers with their real Python planner. Two object
groups have different source bytes, block tables and rotation caches.

Values and keys at unchanged positions must preserve raw BF16 storage words,
including negative zero. Shifted keys are checked against independent CPU
rotation with a declared 0.03125 absolute tolerance. The older executor
controls separately require exact parity with per-chunk native recovery.
Poison checks cover unwritten destinations; partial tails, repeated slot
reuse and the native sixteen-pointer boundary are included.

```sh
make defconfig-kmesh-native-cuda
make kmesh-native-plan
KMESH_NATIVE_SOURCE=/path/to/pinned/LMCache \
  KMESH_NATIVE_OUTPUT=/path/outside/checkouts/native-capture \
  KMESH_NATIVE_PY=/path/to/prepared/environment/bin/python \
  make kmesh-native

KMESH_NATIVE_OUTPUT=/path/outside/checkouts/native-capture \
  make kmesh-native-verify
```

The registered-copy profile uses an exporter and a separate test server with
actual CUDA interprocess communication (IPC) tensors, streams and events.
The key/value cache is a single
BF16 layer with shape `[2, 16, 4, 2, 8]`; the rotary cache is `[64, 8]` FP32,
so each host snapshot contains `64 * 8 * 4 = 2048` bytes. Registration starts
from drained inputs. A later write runs on a non-default producer stream, and
the server must receive its event while that producer is still pending before
copying. A bounded sequence of 512 FP32 matrix multiplications on `[2048, 2048]`
inputs supplies that challenge. It is a synthetic delay fixture, not model work
or a performance measurement. If the producer has already drained, the challenge
cannot qualify.

Success requires raw snapshot bytes to match the independent host input.
Two error controls inject an exception after actual host-copy submission. One
requires a successful physical stream drain; the other injects a drain refusal
and checks that both registration holds remain pending until a later real drain.
The exporter keeps its allocations until the child has completed cleanup.
These controls do not qualify actual device faults or exporter death.

The rotary-transfers profile preserves that download challenge and adds
independent upload cases. The server prepares completed host inputs in a new,
unpublished rotary generation, uploads them into new device tensors and checks
their raw bytes after physical completion. Successful, drained-error and
undrained-error cases run in each direction. These are separate transfer cases;
they do not implement the complete ordered registration and publication path.

Use the same commands with the separate defconfig and a fresh capture:

```sh
make defconfig-kmesh-registered-copy-cuda
make kmesh-native-plan
CUDA_VISIBLE_DEVICES=0 \
  KMESH_NATIVE_SOURCE=/path/to/pinned/LMCache \
  KMESH_NATIVE_OUTPUT=/path/outside/checkouts/copy-capture \
  KMESH_NATIVE_PY=/path/to/prepared/environment/bin/python \
  make kmesh-native
KMESH_NATIVE_OUTPUT=/path/outside/checkouts/copy-capture \
  make kmesh-native-verify
```

Use `make defconfig-kmesh-rotary-transfers-cuda` with the same commands for the
six-case profile, selecting its exact source commit and a fresh output path.
Its [CPU controls](../kmesh_contracts/README.md) run separately through
`defconfig-kmesh-rotary-transfers-cpu`.

Use `make defconfig-kmesh-ordered-registration-cuda` for the later public-server
gate. It starts from invalid rotary values, challenges an actually pending
non-default producer before the first value read, normalizes the completed host
snapshot and checks the private GPU bytes after upload. Both ZeroMQ and gRPC
exercise success, disabled admission and an injected undrained copy followed by
actual server drain and retry. Fresh PyTorch export wrappers supply each attempted
import; the disabled case reuses already-consumed KV metadata and must refuse
before importing it. An unused export cannot establish a receiver close.

This profile pins all loaded generated wire descriptors as well as source and
native binaries. Regenerate the LMCache gRPC bindings using its pinned generator
before execution. Its complete process-exit log is hashed and an outstanding
exporter-reference warning rejects qualification even after green pytest results.
Inspecting normalized private GPU bytes is a test hook; readiness and publication
still pass through the real public server entry point and typed controls.
No positive exporter teardown, raw/VMM import lifecycle, actual device fault,
model-serving, DMA-BUF I/O or speed qualification follows. Engine admission stays
closed. Use its [316-case CPU profile](../kmesh_contracts/README.md) separately.

The `raw-ipc-lifetime` profile qualifies the next lower-level prerequisite.
Select `make defconfig-kmesh-raw-ipc-lifetime-cuda` and its pinned source with
the same native runner commands. Two new cases put the exporter in the parent
and raw imports in a separate child. They test an injected refusal before native
close followed by real successful retry, and retirement before the last alias
releases. The existing shared-allocation reference-count control also runs.
Each allocation is 64 MiB FP32 with two interior views. Keep the exporter alive
until its importer exits, including test failure cleanup.

Checked close preserves the mapping and final reference on native failure.
Retirement blocks new raw imports while existing aliases retain their references.
Inspection counts only raw-wrapper imports; it cannot prove that PyTorch storage,
VMM or unrelated driver imports closed. Retired handles remain tombstoned for the
process lifetime. This prototype has no integrated server receipt or tombstone
reclamation policy and cannot authorize general exporter release. Actual driver
faults, process death and model serving remain separate gates. The profile freezes
wire schemas, checks the complete exit log and records CUDA-binding/CuPy versions.

For later lifetime work, select `make defconfig-kmesh-raw-ipc-outcomes-cuda`
with its own pinned source. It preserves the three-case checkpoint and extends
the controls to distinguish safe preflight retry from an ambiguous native result.
Unknown import or close effects stay quarantined until importer teardown; the
same native call must not run a second time. An absent completed mapping record
does not prove import never occurred. The six cases also verify that serialized
wrappers reset receiver-local counts while the sender keeps its own references.
Controlled native errors occur before driver effects; no actual device-fault
recovery or positive service teardown receipt is qualified.

The `owned-raw-copy` profile connects raw import ownership to actual downloads.
Select `make defconfig-kmesh-owned-raw-copy-cuda`, its pinned source and a fresh
capture with the same native commands. The manager owns every incoming wrapper
before importing the first table. It validates metadata, waits for a pending
producer, copies the table and closes owned raw references only after physical
stream completion. A dedicated 64 MiB rotary allocation contains a `[64, 8]`
FP32 view, so the payload is 2,048 bytes and its final close is independent of
the paged BF16 allocation. Default same-host CUDA event IPC supplies ordering;
the isolated timeline backend is a separate qualification.

Four cases check exact bytes, cleanup refusal before checked close, a submitted
copy with failed drain, and partial second import combined with cleanup refusal.
Refusals retain both borrows, every mapping owner and all buffers until later
physical drain and checked reference release. Ambiguous native outcomes remain
quarantined for importer teardown; the stream drain cannot make them retryable.
The tests supply receiving identity explicitly and do not expose a raw control
RPC or service retirement receipt. Run its separate 323-case
[CPU controls](../kmesh_contracts/README.md) with
`defconfig-kmesh-owned-raw-copy-cpu`. No model or storage performance is measured.

The `managed-raw-context` profile uses
`make defconfig-kmesh-managed-raw-context-cuda` and nine separate-process cases.
Construction retains the context before its first import and drains pointer-table
uploads on their original initialization stream. Checked close drains the later
transfer stream, permanently refuses new raw aliases and releases every owned
reference. Remaining aliases or uncertain native outcomes prevent release.
A successful repeated close returns the same allocation identities without a
second driver call. Partial construction remains held until importer exit.

The parent retains a 64 MiB BF16 allocation with two nonoverlapping
`[2, 16, 4, 2, 8]` views filled with 7 and 9. Actual work on the context stream
must still be pending before close. Cases include alias lifetime, initialization
and transfer drain refusal, pre-native close refusal and ambiguous native close.
Active GDS refusal uses a policy stand-in, not actual backend cleanup. This
bounded resource primitive requires inactive GDS and does not provide a typed
managed RPC, service receipt, storage or model result. Ten separate ordinary
registration/raw-copy regression cases also pass against this source, but are
outside this nine-case profile. Its paired CPU profile requires 327 cases.

Select `make defconfig-kmesh-managed-raw-service-cuda` for the later ten-case
profile. A real exporter and importing raw-only service use both public request
transports. The server records boot, receiving client, nonce, exact views and
context ownership before import. Repeated registration/rotary preparation/status
lookup preserve saved state without another import. Changed views, foreign
sessions and stale generations refuse.

The producer must still be pending when ordered rotary preparation begins.
Host snapshots and private uploaded bytes compare exactly; later exporter writes
do not change private state. Retained copy debt and a known pre-native close
refusal prevent release until actual cleanup. Partial construction and an
ambiguous native close require importer exit without another native pointer call.
Native errors are controlled before driver effects, not actual device faults.

Successful clients validate the complete release receipt, drop exported tensors,
empty the allocator cache and allocate replacements while the service remains
alive. Private readback and repeated retirement still pass; native close counts
do not grow. Two requested 64 MiB exporter allocations contain a BF16
`[2, 16, 4, 2, 8]` paged view and a 2,048-byte FP32 `[64, 8]` rotary view.
Repeated calls test lost-reply recovery semantics without dropping network packets.

This bounded service admits one context and rotary preparation per lifetime,
world size one, CPU DRAM L1 and same-host CUDA event IPC. It exposes no legacy
transfer, lookup, peer, HTTP or automatic reaper interface. Active GDS remains
refused; storage namespace execution, a model consumer and DMA-BUF comparison
remain separate gates. Use the paired 331-case CPU profile for ownership and
schema controls. Counts across prerequisite profiles overlap and are not additive.

The plan and saved-capture verifier use the standard library and allocate no
GPU memory. Execution requires one selected CUDA device and a fresh external
output directory. It writes a log, full test XML and a source/native/runtime
receipt. Every selected setup, test body and teardown must pass. Missing,
duplicate, skipped or expected-failure cases are rejected. Keep these captured
files outside source control. `make kmesh-native-test` exercises the verifier's
rejection controls on CPU with synthetic receipts.

The paged-cache grouped registration tests use local storage and event-handle
transport stand-ins. Pinned allocation, geometry, planner, CUDA events and kernels are
real. This qualifies the declared synthetic native cases; it does not qualify
inter-process ownership, full-model recovery, DMA-BUF I/O, serving latency,
compressed KV or padded pools. Those need their own subsequent controls.
The registered-copy primitive is separate from runtime discovery. Its tests
supply the logical receiving identity and use already-drained legacy
registration; they do not exercise the public control RPCs. Ordered registration
before its first read, exporter teardown acknowledgement, namespace/data
execution and model validation remain required before multiprocess KMesh engine
admission can open. No DMA-BUF storage path or speed comparison is exercised.
Offline receipt verification establishes file consistency, not independent
authentication of hardware execution. Keep each capture tied to its helper
revision, and verify earlier captures with the revision that produced them.

Select `make defconfig-kmesh-native-spans-cuda` for the native storage profile,
then use the same plan/run/verify commands with its exact source pin. It retains
ten raw service lifetime cases and adds fourteen span cases across ZeroMQ and
gRPC. Regenerate bindings from that source before execution. The paired CPU
profile is `defconfig-kmesh-native-spans-cpu`, requiring 341 cases and the vLLM
source pinned in the [CPU instructions](../kmesh_contracts/README.md).

The span fixture registers synthetic BF16 `[2, 16, 4, 2, 16]` paged storage and
FP32 `[64, 16]` rotary state, in distinct 64 MiB exporter allocations. Its lazy
pinned CPU L1 is 64 MiB, the allocator's registration granule. Two eight-token
chunks hold a sixteen-token source. Real native gather saves that source after
a producer submitted behind 512 FP32 matrix multiplications on `[2048, 2048]`
inputs. The worker poisons its original GPU slots after the stored receipt.
Native restore then scatters at token position three through a reversed,
noncontiguous destination block table. Untouched prefix and tail slots remain
poisoned. Values must match raw BF16 bits; keys use an independent CPU rotation
with `rtol=0.012` and `atol=0.025`. A second same-position restore verifies that
shared canonical source bytes were never rotated in place.

The remaining controls cover repeated-generation idempotency, foreign clients,
altered handles, eviction, corruption before scatter, failed admission and
stale publication. An exception after real native enqueue plus an injected
drain refusal must retain physical ownership. Retirement refuses; later actual
device-context drain releases the debt without another enqueue. These injected
refusals do not establish recovery from a real device fault.

This profile qualifies native CPU-L1 storage transactions, not model recovery.
The `loaded-needs-repair` receipt does not authorize computed vLLM tokens.
Request-local deeper-layer repair, model output checks, NVMe-backed cold reads,
GPU-L1/GDS, the staged/DMA-BUF comparison and collective consumers remain separate
qualification gates. Timings from these correctness controls are not a serving
latency result.

Select `make defconfig-kmesh-interior-spans-cuda` to qualify source positions
inside a block. Its paired CPU profile is `defconfig-kmesh-interior-spans-cpu`.
The new source pin preserves the earlier `native-spans` profile and captures.
This profile requires 36 GPU cases: ten service lifetime controls and twenty-six
span cases. Interior-source cases use 24 synthetic layers in both NHD and HND,
source position five, and destination positions three and nine. Each layer view
shares the same retained exporter allocation. Source writes remain genuinely
pending when submitted to the cache service.

The service gathers exact source slots into complete pinned CPU chunks. The
fixture checks independent key rotation, bitwise values, untouched destinations
and canonical source reuse after the original slots are poisoned. An injected
failure after actual source gathering must keep admission and retirement blocked
until physical drain; cleanup must not gather again. This extends storage
correctness coverage, without enabling model repair or measuring DMA-BUF.

Select `make defconfig-kmesh-registered-arenas-cuda` for eager canonical sources.
Its 184 required cases retain the registered-input controls and add matched
eager pinned-host and explicit Vulkan/RM GPU arenas. They cover separate and
per-head fused K/V, NHD/HND, aligned chunks, interior starts and partial tails.
The optional `_vulkan_rm` extension is pinned and checked alongside the native
transfer bindings. Provider failure is a failed test, never a host fallback.
Use the ordinary plan/run/verify commands with this profile and a fresh capture.

Every arena control uses a 64 MiB canonical pool. Working copies must preserve
canonical bytes across shifted and same-position restores. Corrupt or evicted
sources refuse before scatter; failed physical drain retains ownership. A
five-chunk source also checks refusal when its plan would reuse one of the
context's four working slots. The paired CPU prerequisite is
`make defconfig-kmesh-registered-arenas-cpu`.

These are synthetic memory-transfer controls awaiting GPU execution. They do
not perform NVMe I/O or run a model. GPU canonical hashing still copies the
full payload to the CPU. A configured DMA-BUF export and a passing arena test
do not establish the storage route, model correctness or lower serving latency.

Select `make defconfig-kmesh-registered-groups-cuda` for simultaneous private
layer inputs. Its 208 required cases include the arena controls and four-member
registered groups over both transports, separate/fused K/V, and NHD/HND. Each
synthetic group uses the same nine-token canonical source across 24 layers,
with destination starts one through four. Each member has a private four-block
fixture region; its actual block table covers only its 12–15 input tokens.
The windows include declared misses and private tail tokens. This complete
profile has a 20-minute execution budget to cover the additional separate-process
cases and checked shutdown. Earlier profiles retain their 15-minute default.
Budgets must be positive integers of at most one hour; invalid budgets refuse
before capture creation or worker launch.

The group must reject a changed member before binding, and the worker must
refuse other restores while group inputs are live. All four layer buffers stay
independent as a synthetic additive transform writes them back to their private
engine slots. Early close must preserve the engine bytes. Refusing device drain
after actual gathers must attempt all member cleanups, retain all buffers, and
require a later actual drain before checked retirement. The paired prerequisite
is `make defconfig-kmesh-registered-groups-cpu`.

These native group controls await GPU execution and do not run a model or NVMe
I/O. The implemented CacheBlend group consumer keeps model tensor operations
request-local and advances them layer by layer. Actual serial-versus-group
model equivalence and storage performance remain separate experiments.
