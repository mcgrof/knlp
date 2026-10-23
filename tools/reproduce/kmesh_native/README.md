# Native cache-layout and position-recovery qualification

Before extending shared-state recovery across processes, check the native
executor's memory layout and positional transformations. These controls use
synthetic BF16 payloads with head-major and token-major paged storage,
separate K/V planes and current fused content-width formats.

The profile pins the LMCache source and native binary hashes. Reuse a matching
prepared CUDA environment; this runner does not install or rebuild packages.
Binary hashes describe the reference build, not a portable ABI. To qualify a
different build, compile the pinned LMCache source using its build instructions
and review a separate copy of `profile.json` with the resulting binary hashes.
Keep the source hashes, selected cases and numerical controls intact. Record
that profile and build provenance with the new capture; it is a fresh
qualification, not a relabeling of the reference build's receipt.
The new grouped controls reach both the public flat executor and the public
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

The plan and saved-capture verifier use the standard library and allocate no
GPU memory. Execution requires one selected CUDA device and a fresh external
output directory. It writes a log, full test XML and a source/native/runtime
receipt. Every selected setup, test body and teardown must pass. Missing,
duplicate, skipped or expected-failure cases are rejected. Keep these captured
files outside source control. `make kmesh-native-test` exercises the verifier's
rejection controls on CPU with synthetic receipts.

The grouped registration tests use local storage and event-handle transport
stand-ins. Pinned allocation, geometry, planner, CUDA events and kernels are
real. This qualifies the declared synthetic native cases; it does not qualify
inter-process ownership, full-model recovery, DMA-BUF I/O, serving latency,
compressed KV or padded pools. Those need their own subsequent controls.
Offline receipt verification establishes file consistency, not independent
authentication of hardware execution.
