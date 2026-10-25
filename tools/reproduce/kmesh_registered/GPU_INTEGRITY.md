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

## Compare complete restoration with DMA-BUF reads fixed

Use the existing [registered replay setup](README.md#matched-storage-routes),
including the independently verified historical reference and pinned source,
model, native binaries and runtime. Select the checksum configuration:

```sh
make defconfig-kmesh-registered-gpu-checksum-cuda
make kmesh-registered-plan
# Set the source, model, trace, reference, interpreter and capture paths.
export KMESH_REGISTERED_CHECKSUM_LIBRARY=/path/outside/checkouts/libkmesh_sha256.so
make kmesh-registered-storage-pair
```

This configuration sets `KMESH_REGISTERED_COMPARISON=checksum`. Both arms use
the same DMA-BUF SSD route. The default is seven matched pairs with alternating
CPU/GPU checksum order. A fresh same-source serial reference must match the
historical reference, and both arms must pass the poisoned-source-pool data
arrival check before timings. The compiled kernel and source hashes are bound
to each GPU result. Model K/V, next-token scores, generated tokens and selected
recomputation positions must match the serial reference.

The observer replaces only the dedicated reproducer's checksum call. LMCache
and vLLM stay at the storage profile's exact pins; production checksum defaults
are unchanged. Before timing children, every saved object retains its CPU
SHA-256 as an independent reference. The GPU call must match it. Each child's
restore also checks against that saved CPU digest before position adjustment.
The ordinary ownership, completion and failed-operation drain handling remains
active. Device-fault injection is not part of this checksum comparison.

Singleflight stays off. The retention fix stays enabled, and explicit eviction
before each child forces the same 12 MiB SSD read in both arms. GPU checksum
receipts must show zero payload-download bytes and four 32-byte digests per
child. Keep source checksum parity separate from timing, and report storage
read calls, checking, complete restoration and the outer model-repair interval.
Those intervals are nested and must not be added together. Full numerical
captures and duplicate reference storage remain active; this is a controlled
restore experiment, not a serving-throughput measurement.

## Measured outcome

The tested kernel preserves exact model outputs and removes the checksum's
payload download, but it makes complete restoration slower in all seven matched
pairs. CPU hashing remains the default. The [published results](https://knlp.io/kmesh-recovery.html#gpu-integrity)
report the standalone and full-model timings, runtime and limitations. This
result concerns the scalar CUDA kernel and current whole-object digest format;
it does not rule out every GPU hash implementation.
