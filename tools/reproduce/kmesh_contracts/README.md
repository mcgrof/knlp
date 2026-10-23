# KMesh CPU ownership controls

This reproducer checks the ownership rules needed before KMesh can move
position-recovered key/value state through a separate cache process. It runs
the domain registry, real ZeroMQ/gRPC request transports and actual CPU cache
registrations, including retirement and replacement with different tensors.
The source profile names every required case. Skipped tests, missing phases,
unexpected expected failures, changed source or incomplete receipts cannot
qualify.

This is a CPU prerequisite. KMesh multiprocess serving admission remains closed.
The tests control completion explicitly; they do not qualify GPU events,
exporter death, model outputs, DMA-BUF transfers or performance. The shared native
extension's hash is recorded rather than restricted to a particular toolchain's
binary. These CPU tests do not qualify GPU kernels.

Use Python 3.11 or newer and a prepared LMCache environment with its shared native
extension, pytest and request-transport dependencies installed. Check out
LMCache commit `93926179d1f6e431093bd504624d3ea68f25d2c9` on a clean checkout.
Generate the gRPC bindings from that checkout using its build dependency
`grpcio-tools==1.78.0`:

```sh
python lmcache/v1/multiprocess/transport/grpc_impl/_proto_gen/_generate.py
```

Run from knlp, selecting the prepared interpreter. The result directory must
be fresh and outside both source checkouts:

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

The launcher hides CUDA devices, removes the explicit LMCache backend override
and requires the pinned CPU fallback. It uses two OpenMP/MKL threads and disables
optional pytest plugin loading and usage tracking. Tests allocate small synthetic
shared CPU tensors and an eight-MiB DRAM store; no model is loaded. Test-module
imports and native extensions must resolve from the selected source tree.

The capture contains JUnit XML, all collected cases and setup/call/teardown
reports, source/helper/build hashes and runtime versions. `verified.json` is
written only after all checks pass. Keep captures outside the source repository;
they are not source files. Saved-file verification establishes consistency,
not independent machine or build attestation. Review future source profiles
explicitly rather than editing an old capture to name a new source.

Run the standard-library verifier controls with `make kmesh-contracts-test`.
Planning and verification do not import Torch or allocate device resources.
