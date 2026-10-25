# Exact GPU SHA-256 evaluation

The restore path currently downloads each whole GPU source object and hashes
it on the CPU. This experiment keeps the same standard SHA-256 digest and
hashes the bytes on the GPU. It returns 32 bytes per object. It changes no
cache identity, padding, repair algorithm or stored digest format.

`gpu_sha256.cu` implements SHA-256 from [NIST FIPS 180-4](https://nvlpubs.nist.gov/nistpubs/FIPS/NIST.FIPS.180-4.pdf),
sections 4.2.2, 5.1.1 and 6.2.2. One thread computes each complete message;
independent messages can run concurrently. Each message's compression chain
remains sequential. This is a small experimental kernel, not a claim about
the performance of all GPU hash implementations or a validated cryptographic
module. It does not use cuPQC or require a vendor hash SDK.

## Build and check the kernel

Use a CUDA toolkit and a PyTorch CUDA environment. Choose the architecture of
the GPU that will run the test. Keep the binary and captures outside Git:

```sh
nvcc -O3 -std=c++17 -arch=sm_89 -Xcompiler=-fPIC --shared \
  tools/reproduce/kmesh_registered/gpu_sha256.cu \
  -o /path/outside/checkouts/libkmesh_sha256.so
python -m tools.reproduce.kmesh_registered.checksum_bench \
  --library /path/outside/checkouts/libkmesh_sha256.so \
  --output-dir /path/outside/checkouts/checksum-capture --trials 15
```

The default workload is four independent 3 MiB objects. This matches the
existing registered model storage profile: each object contains
256 padded tokens × 24 layers × 2 key/value planes × 2 heads × 64 elements
× 2 bytes = 3 MiB. The benchmark uses deterministic synthetic bytes.

Before timing, the benchmark checks empty messages, SHA-256 padding boundaries,
unaligned inputs, zero/random/all-one bytes, a published `abc` vector and BF16
raw bytes against the CPU implementation. It flips bytes at four positions in
one object, checks the changed digest against CPU SHA-256, checks that sibling
digests stay unchanged, and exercises an explicit CUDA producer-event handoff.
These checks do not inject device faults or establish storage-I/O visibility.

Compare three methods in rotating order: CPU hashing including the full GPU
payload download, four serial GPU calls, and one GPU call for all four objects.
Each timed call returns its digests to the CPU and checks them. GPU timings
include output allocation, kernel launch and the digest download. Warmup runs
are excluded. Batched hashing is a separate opportunity check; the existing
serial restore path hashes each source separately.

## Completion and ownership

`CudaSha256` accepts contiguous CUDA tensors and uses their raw bytes. It
launches on the current CUDA stream and waits for the digest download before
returning. The caller must first establish source readiness, including visibility
of external storage writes. An io_uring completion is not itself a CUDA event.
The helper does not make external I/O stream-ordered.

Retain physical source ownership until hashing completes. On an error, retain
the operation and exception frames, then use the caller's actual device-drain
handling before releasing memory. The dedicated KMesh service already owns
source leases and quarantines unresolved operations. Kernel correctness alone
does not qualify the restoration integration or production serving.
