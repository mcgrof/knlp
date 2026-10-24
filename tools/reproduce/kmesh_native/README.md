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
