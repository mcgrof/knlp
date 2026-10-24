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
| `ordered-registration` | `27e256e1aae7090c4ccfc5eea587a7fbf93fddae` | `9df1493cae0b98380a3c3ecb9b2d41776a6d8735` | 316 |
| `native-spans` | `daa0b4ddc8551d29d21ff4ac51f913eb29dfed9a` | `9df1493cae0b98380a3c3ecb9b2d41776a6d8735` | 341 |
| `interior-spans` | `e23d4acd29420a0258a9666b98d9a10fdec12b53` | `9df1493cae0b98380a3c3ecb9b2d41776a6d8735` | 341 |
| `managed-raw-service` | `d3a4b9a6dd2a8edb649d24d48cb6e6b46b10bd9e` | `9df1493cae0b98380a3c3ecb9b2d41776a6d8735` | 331 |
| `managed-raw-context` | `304df82413dbb9d9d12de145963b5451d1e9eeae` | `9df1493cae0b98380a3c3ecb9b2d41776a6d8735` | 327 |
| `owned-raw-copy` | `ca65f2820a218a84ecba4d452c47da2c1ac1c4f5` | `9df1493cae0b98380a3c3ecb9b2d41776a6d8735` | 323 |

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

The ordered-registration profile adds first-read snapshot preparation and
private canonical publication through both real request transports. It checks
normalization, independent ownership, stale and foreign clients, disappeared
shared-memory inputs and shutdown during preparation. Use
`make defconfig-kmesh-ordered-registration-cpu` with the same commands and its
exact source pair. It does not provide positive exporter teardown or enable
engine recovery. Its six-case CUDA counterpart runs separately through
`defconfig-kmesh-ordered-registration-cuda`.

Use `make defconfig-kmesh-owned-raw-copy-cpu` for the later 323-case profile.
Controlled raw mapping owners model partial import, refused cleanup, interrupted
close, failed copy drain, metadata refusal and shutdown with retained debt.
Real CPU registration checks also reject this CUDA-only operation and preserve
its registered state. The controlled device part allocates no GPU memory; the
separate four-case CUDA profile checks actual imports and transfers. Typed raw
registration, exporter release and engine admission remain subsequent gates.

Use `make defconfig-kmesh-managed-raw-context-cpu` for 327 cases, adding four
admission checks to the owned-copy profile. Missing or mixed raw owners and an
active GDS singleton refuse before import. Its nine-case CUDA counterpart tests
actual construction and checked context retirement; service release receipts and
model execution remain unqualified.

Use `make defconfig-kmesh-managed-raw-service-cpu` for the 331-case service
profile. Its four additional controls check CPU refusal, the bounded handler
inventory, supplied context ownership and an actual CPU registration held through
retirement. The wire-annotation coverage includes the dedicated managed module.
Regenerate the selected source's schemas before running. Real raw imports and
release receipts require the separate ten-case CUDA profile.

Use `make defconfig-kmesh-native-spans-cpu` for 341 cases. This includes actual
CPU L1 admission and readback, plus physical host-buffer leases that prevent
reuse after logical writer or reader ownership is released. Span requests also
reject the legacy key grammar, missing producer events, foreign sessions and
CPU execution before native GPU work. Its separate 24-case CUDA profile includes
ten service lifetime controls and fourteen actual native span tests. It does
not run a model or compare storage routes.

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

Use `make defconfig-kmesh-interior-spans-cpu` for the matching interior-source
revision. It requires the same 341 CPU controls and exact vLLM source, while
pinning the native source-gather implementation and its GPU test program.
Actual interior NHD/HND gathering is qualified by the separate CUDA profile;
CPU execution alone cannot establish native transfer correctness.

Select `make defconfig-kmesh-registered-arenas-cpu` for the eager-arena source
prerequisite. It pins the partial/fused span implementation and requires 661
CPU contract cases, including refusal of pageable, foreign or incomplete
source objects and unsupported managed-service storage configurations.
Supply the exact LMCache and vLLM trees shown by its plan. These CPU checks
allocate no GPU arena; use the paired CUDA profile to qualify actual pinned
and Vulkan/RM transfers, repacking, canonical preservation and physical drain.
