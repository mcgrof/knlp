# KMesh CPU ownership controls

This reproducer checks the ownership rules needed before KMesh can move
position-recovered key/value state through a separate cache process. It runs
the domain registry, real ZeroMQ/gRPC request transports and actual CPU cache
registrations, including retirement and replacement with different tensors.
For example, a replacement client cannot inherit the old client's registration,
and a retired context cannot release its tensors while accepted work still
holds them. The registered-state profile also exercises rotary-state ownership,
typed control messages and refusal of unqualified engine configurations.
The source profile names every required case. Skipped tests, missing phases,
unexpected expected failures, changed source or incomplete receipts cannot
qualify.

This is a CPU prerequisite. KMesh multiprocess serving admission remains closed.
The tests control completion explicitly; they do not qualify GPU events,
exporter death, model outputs, DMA-BUF transfers or performance. The shared native
extension's hash is recorded rather than restricted to a particular toolchain's
binary. These CPU tests do not qualify GPU kernels.

Use Python 3.11 or newer and a prepared LMCache environment with its shared native
extension, pytest and request-transport dependencies installed. Select a profile
and use clean checkouts at its exact commits:

| Profile | LMCache commit | vLLM commit | Required cases |
| --- | --- | --- | ---: |
| `contexts` | `93926179d1f6e431093bd504624d3ea68f25d2c9` | No additional source pin | 91 |
| `registered-state` | `a301244af6d202f041fe321c6497963b16f626e9` | `9df1493cae0b98380a3c3ecb9b2d41776a6d8735` | 268 |
| `rotary-transfers` | `062c4ea6487f86c91e30b3c98ac7e4e520f19106` | `9df1493cae0b98380a3c3ecb9b2d41776a6d8735` | 292 |

The registered-state profile requires the paired vLLM source and a prepared
environment containing its dependencies. It runs real server processes on both
ZeroMQ and gRPC transports, while keeping device runtime discovery refused.
The model, tokenizer and weight digests are worker declarations: the cache
server verifies registered layout and rotary contents but does not load or
attest the model weights. This profile does not enable multiprocess KMesh
inference.

The rotary-transfers profile retains those controls and adds independent host
snapshots, provisional rotary ownership and both transfer directions. It uses
controlled device metadata and weak references to pending reads for error,
interruption, collection and shutdown tests. That part models lifetime debt
without allocating GPU memory. Provisional state must not replace a live blend
registration. Ordered registration and canonical publication remain separate
work; this CPU profile does not qualify device transfers.

Generate the gRPC bindings from the selected LMCache checkout using its pinned
build dependency. From that checkout:

```sh
uvx --from grpcio-tools==1.78.0 python \
  lmcache/v1/multiprocess/transport/grpc_impl/_proto_gen/_generate.py
```

Run from knlp, selecting the prepared interpreter. The result directory must
be fresh and outside all source checkouts. For the original context profile:

```sh
make defconfig-kmesh-contracts-cpu
make kmesh-contracts-plan
make kmesh-contracts \
  KMESH_CONTRACTS_PY=/absolute/path/to/prepared/python \
  KMESH_CONTRACTS_SOURCE=/absolute/path/to/LMCache \
  KMESH_CONTRACTS_OUTPUT=/absolute/path/to/new-capture
make kmesh-contracts-verify \
  KMESH_CONTRACTS_OUTPUT=/absolute/path/to/new-capture
```

For the registered-state profile, select its defconfig and supply both source
paths. Keep the same configuration when verifying the capture:

```sh
make defconfig-kmesh-registered-state-cpu
make kmesh-contracts-plan
make kmesh-contracts \
  KMESH_CONTRACTS_PY=/absolute/path/to/prepared/python \
  KMESH_CONTRACTS_SOURCE=/absolute/path/to/LMCache \
  KMESH_CONTRACTS_VLLM_SOURCE=/absolute/path/to/vllm \
  KMESH_CONTRACTS_OUTPUT=/absolute/path/to/new-capture
make kmesh-contracts-verify \
  KMESH_CONTRACTS_OUTPUT=/absolute/path/to/new-capture
```

Use `make defconfig-kmesh-rotary-transfers-cpu` for the later transfer controls,
then the same plan/run/verify commands with that profile's exact source commits
and a fresh output directory. The corresponding real-device prerequisite uses
`defconfig-kmesh-rotary-transfers-cuda` and the
[native runner](../kmesh_native/README.md).

The launcher hides GPU devices, removes the explicit LMCache backend override
and requires the pinned CPU fallback. It uses two OpenMP/MKL threads and disables
optional pytest plugin loading and usage tracking. Tests allocate small synthetic
shared CPU tensors and an eight-MiB DRAM store; no model is loaded. Test-module
imports and native extensions must resolve from the selected source tree. The
paired vLLM package must also resolve from its pinned checkout. Its target is
explicitly CPU for these controls.

The capture contains JUnit XML, all collected cases and setup/call/teardown
reports, source/helper/build hashes and runtime versions. The paired source
profiles also check loaded protobuf descriptor hashes, so stale generated
bindings cannot qualify against a newer schema. `verified.json` is
written only after all checks pass. Keep captures outside the source repository;
they are not source files. Saved-file verification establishes consistency,
not independent machine or build attestation. Review future source profiles
explicitly rather than editing an old capture to name a new source. Captures
remain tied to the helper revision that produced them; verify earlier captures
with that revision.

Run the standard-library verifier controls with `make kmesh-contracts-test`.
Planning and verification do not import Torch or allocate device resources.
