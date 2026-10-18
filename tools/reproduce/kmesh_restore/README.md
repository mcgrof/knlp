# Exact-prefix KV restoration through DMA-BUF

This comparison asks whether direct storage transfers reduce the time to
restore an offloaded ancestor for a family of reasoning requests. It connects
the existing exact-prefix workload to the real LMCache multiprocess connector,
then compares registered host staging with a registered GPU DMA-BUF arena.
The four-pair RTX 4090 transport comparison passes. Median GPU-consumer
readiness is 9.211 ms direct versus 11.006 ms staged, 16.3% lower in the
October 17 repeat. Adapter completion itself is nearly equal, 8.023 versus
8.063 ms. Each arm has 400
measured samples; byte checks, deliberate corruption rejection, actual native
routes and drained ownership pass. This measures repeated warm SSD ranges.
See the [results and limits](https://knlp.io/kmesh.html#restore-latency).

The serving comparison enables vLLM's batch-invariant kernels in every arm,
including recomputation. This fixes the arithmetic policy while external
restoration changes scheduler batch shapes. The initial default-kernel smoke
matched restored KV digests and output tokens, but exceeded the fixed 0.02
chosen-logprob bound against full recomputation. That attempt remains a failed
gate. The deterministic retry matches all 48 bounded output-token observations
and their chosen log probabilities exactly, but fails a separate strict gate:
one child in a five-request warmup recomputes rather than reusing the ancestor.
Formal serving timing stopped at that gate. The October 17 full repeat with
INFO logging passes all strict gates: all 1,152 continuation observations,
including warmups, restore the complete ancestor and match recomputation's
tokens and chosen log probabilities exactly. No admission fix was made;
the earlier miss remains unexplained.

Four alternating pairs retain 20 families per fanout and arm per pair.
Single-child median time to first token is 23.403 ms direct versus 25.015 ms
staged, 6.4% lower, with all four paired medians favoring direct. Five-child
pooled medians are 54.912 versus 55.183 ms, but paired changes vary in sign
and batch completion does not improve. There is no stable burst benefit.
Every retained five-child family reads 240 MiB for its 48 MiB ancestor in
both paths. Those are adapter request counts, not device command traces or
proof of simultaneous GPU-memory copies. The default defconfig selects the
adapter comparison; `KMESH_RESTORE_SECTION=all` selects the full comparison.

Each serving row retains vLLM's request identifier so a cache miss can be
matched to the connector and storage logs. For an admission investigation,
set `LMCACHE_LOG_LEVEL=DEBUG` and `VLLM_LOGGING_LEVEL=DEBUG` in the queued
wrapper. These logs add overhead; use a separate run without diagnostic
logging for latency comparisons. The numerical and cached-ancestor gates
apply to both runs.

The model is Qwen/Qwen2.5-0.5B-Instruct at revision
`7ae557604adf67be50417f59c2c2f167def9a775`. The workload retains seed 20260805,
a 4,096-token ancestor and five different 512-token tails. Each 256-token KV
chunk occupies 3 MiB: two K/V components times 24 layers times 256 tokens times
two KV heads times 64 values times two bytes for BF16. Sixteen chunks make a
48 MiB ancestor. These are synthetic exact causal prefixes, with one generated
token per child. They are not task-quality or shifted-span recovery tests.

## Runtime and sources

Use the public current branches, with a clean checkout at each exact pin:

| Project | Commit | Branch |
| --- | --- | --- |
| vLLM | `1b76bf9dd9d52efbd75fb89bf2619898d8f01233` | `2026-10-16-feature-kmesh-serving-replay` |
| LMCache | `007a3388331209737ee19c095e430907fc467952` | `2026-10-16-feature-kmesh-dmabuf-restore` |

Prepare matching vLLM wheel binaries and build LMCache's CUDA extensions for
the selected GPU. The [current CUDA baseline instructions](../tokendance/README.md#current-cuda-profile)
give the pinned Python, PyTorch, wheel and package inputs. This experiment also
needs the native `lmcache_rust_raw_block_io` extension from `lmcache/rust/raw_block`
and `cupy-cuda13x==14.2.0` with `fastrlock==0.8.3` for isolated CUDA IPC. Build
and install into a dedicated environment; preserve the source and binary hashes.
Set `PYTHONPATH` to the clean LMCache checkout and the clean vLLM serving
checkout so their in-tree modules take precedence over installed packages.

GPU DMA-BUF I/O requires the patched kernel, compatible exporter and NVMe peer
route documented in the [DMA-BUF workflow](../dmabuf/README.md). A stock RunPod
container is not sufficient. The command below selects the experimental
`vulkan_rm` provider used on RTX 4090. It requires the optional LMCache
extension, built with `BUILD_WITH_VULKAN_RM=1` and the clean pinned NVIDIA
headers selected by `LMCACHE_NVIDIA_RM_HEADERS`. The build checks its header
revision. This provider has a specific software qualification; it does not
enable export of arbitrary existing Torch allocations. An unsupported provider
or registration fails rather than selecting host staging automatically.

For a dedicated runtime, the CUDA baseline preparer accepts an explicit
compiler architecture. For example, architecture 8.9 and two build jobs select
the compilation settings used for this storage test. Set `CUDA_HOME` to the
prepared CUDA toolkit and, for `vulkan_rm`, provide the clean pinned headers
required by LMCache's build profile. Vulkan development headers/libraries,
a C++20 compiler and the matching driver must already be installed. The
NVIDIA headers must be a clean checkout at
`61dcc93722ecb418bb5f2e00923f05b4b8051dd1`; the build rejects other revisions:

```sh
runtime=/absolute/path/to/a/new/restore-runtime
TORCH_CUDA_ARCH_LIST=8.9 MAX_JOBS=2 BUILD_WITH_VULKAN_RM=1 \
  LMCACHE_NVIDIA_RM_HEADERS=/absolute/path/to/pinned/nvidia/headers \
  bash tools/reproduce/tokendance/prepare_cuda.sh "$runtime"

git -C "$runtime/lmcache" fetch --depth=1 origin \
  007a3388331209737ee19c095e430907fc467952
git -C "$runtime/lmcache" diff --exit-code \
  643f8ca3962e0d02687ed9d0194e0915679b3c0b \
  007a3388331209737ee19c095e430907fc467952 -- csrc rust setup_extensions
git -C "$runtime/lmcache" checkout --detach \
  007a3388331209737ee19c095e430907fc467952
uv pip install --python "$runtime/venv/bin/python" \
  "$runtime/lmcache/rust/raw_block" cupy-cuda13x==14.2.0 fastrlock==0.8.3
ln -s "$runtime/venv/lib/python3.13/site-packages/vllm/_version.py" \
  "$runtime/vllm-serving/vllm/_version.py"
```

The tree comparison establishes that the checksum-only source change does not
alter the native code already compiled by the baseline preparer. The resulting
LMCache checkout belongs to this restore profile. Use a separate runtime for
the TokenDance profile's different source pin. Native export uses the same
procedure without the Vulkan/RM build variables. Compilation does not reserve
a GPU. Run GPU execution on an available, qualified device.

## Run the restore experiment

Load the preset and inspect its CPU-only plan:

```sh
make defconfig-kmesh-restore-cuda
make kmesh-restore-plan
```

Set the source, model and output paths through menuconfig or the
`KMESH_RESTORE_VLLM_SOURCE`, `KMESH_RESTORE_LMCACHE_SOURCE`,
`KMESH_RESTORE_MODEL_DIR` and `KMESH_RESTORE_OUTPUT` environment variables.
Select the prepared interpreter with `KMESH_RESTORE_PY`. The output parent
must be on the qualified NVMe filesystem. A shell wrapper can provide these
inputs explicitly:

```sh
#!/bin/bash
set -euo pipefail
export CUDA_HOME=/path/to/prepared/cuda
exec make kmesh-restore \
  KMESH_RESTORE_PY=/path/to/restore-runtime/venv/bin/python \
  KMESH_RESTORE_VLLM_SOURCE=/path/to/restore-runtime/vllm-serving \
  KMESH_RESTORE_LMCACHE_SOURCE=/path/to/restore-runtime/lmcache \
  KMESH_RESTORE_MODEL_DIR=/path/to/pinned/model \
  KMESH_RESTORE_OUTPUT=/path/to/qualified/nvme/fresh-results
```

Run that wrapper with `/bin/bash /path/to/wrapper.sh`.
GNU make exports its command-line variables to the launcher. The preset selects
`vulkan_rm` and `adapter`; `KMESH_RESTORE_SECTION=all` explicitly includes the
experimental strict serving qualification. A failed gate remains a failure;
the verifier does not remove recomputed requests or widen numerical bounds.

Both paths run on the same GPU. The following placeholders refer to the prepared
interpreter, clean source checkouts and authenticated local model snapshot:

```sh
/path/to/python tools/reproduce/kmesh_restore/run.py \
  --output /path/to/fresh/results \
  --model /path/to/pinned/model \
  --vllm-source /path/to/vllm \
  --lmcache-source /path/to/LMCache \
  --gpu-provider vulkan_rm --section adapter
```

Pass any required runtime variables explicitly through `/usr/bin/env` or a
shell wrapper. In particular, preserve the prepared `PYTHONPATH`, CUDA
library paths, offline model flags and sufficient process-local memlock limit.
The driver fixes two OpenMP and MKL threads. It creates only new regular files
inside the result directory; it never opens a supplied raw device or overwrites
an existing fixture. Output directories must be fresh. Child processes remain
under the driver and are drained before it exits.

Each transport must first reject a deliberately corrupted payload at the
full-byte comparison. The driver qualifies and measures the adapter path
before starting model controls. It then produces a recomputation control
without a connector and qualifies the serving path before timing that path.
Each part has four measured pairs that alternate staged/direct ordering.
Each pair retains 100 adapter samples or 20 serving families at each fanout,
after excluded warmups. A failed model gate leaves the independently verified
adapter evidence available, while keeping serving speed claims blocked.
Changing these counts is explicit through `--adapter-samples`,
`--serving-samples` and `--rounds`; the plan records the choices.

## What is timed

The adapter test submits all sixteen chunks through `RawBlockL2Adapter`, waits
for its completion event and ownership result, then copies them into a GPU
consumer and reduces every byte. `load_ms` ends at adapter completion;
`ready_ms` ends after the GPU consumer fence. Full byte comparison and SHA256
checks occur outside timing. A synthetic deterministic byte pattern gives an
independent transport-integrity check with the model's exact byte geometry.

The serving test uses one storage owner, a producer vLLM engine, then a fresh
consumer engine using `LMCacheMPConnector`. The producer persists the ancestor.
Before every consumer family, the harness clears the engine's automatic prefix
cache and drains and clears LMCache L1. It preserves the external index and SSD
data. The consumer serves one child or a five-child burst. vLLM reports each
request's first-token latency; a separate wall timer covers the complete batch.
No model startup or HTTP cache-clear call is included in those timers.

A separate untimed control captures actual ancestor block IDs, then asks the
production MP checksum endpoint to compare every layer and ancestor chunk with
the producer. The allocation hook and checksum calls are absent from measured
requests. Block IDs can be freed and reused between siblings before these
post-family hashes are read. This check does not establish simultaneous sharing;
the dedicated serving replay tests that separately. The verifier also compares
generated tokens with the recomputation
control and enforces a fixed 0.02 bound on the chosen token's log probability.
This one-token fixture is not evidence of general model quality.

Both arms use 256 MiB L1 arenas, 512 MiB engine cache pools, 4 MiB storage slots,
ordinary file `O_DIRECT`, io_uring depth eight and one adapter load/store worker.
The engine pool fits all five private ancestors and tails if external admission
does not share them. The direct L1 arena consumes GPU memory; the staged arena
consumes pinned host memory. This is equal configured cache capacity, not equal
total process memory. Direct and staged paths can split the same logical I/O
into different device commands. The raw records count backend requests,
submitted operations, completed operations and bytes; they do not trace NVMe
commands or prove the same physical command geometry.

The adapter and application caches are cold at each restore, while the tested
SSD ranges are repeatedly read and remain warm. These measurements do not
establish cold-media behavior, sustained decode throughput, cascade-attention
benefits or prefetch gains. The existing TokenDance shifted-span profile stays
CPU-backed. This harness imports no family-prefetch or external-singleflight
implementation.

## Cold external-cache arrival controls

`make defconfig-kmesh-coldburst-cuda` selects the original simultaneous
arrival versus completed-leader experiment on the current NVMe paths.
Use the same prepared runtime, source and model inputs above, then inspect
`make kmesh-restore-plan` and run `make kmesh-restore` with the prepared
interpreter on an available GPU. The preset selects `KMESH_RESTORE_SECTION=coldburst`, three retained
families after three excluded warmups per case, bursts of 2, 4 and 8 children,
and a four-child staggered control. The leader-first call completes before
the other three requests are submitted. Both host staging and GPU DMA-BUF
use fresh storage owners with identical settings.

This profile retains the current seed 20260805, 4,096-token ancestor,
512-token tails and eight-token numerical controls. It does not reproduce the
historical host-RAM tier or historical wall times.
The server arena is 512 MiB in each arm so eight outstanding private restores
are not constrained by the transport comparison's 256 MiB arena. The engine
KV pool remains 512 MiB, sufficient for eight 48 MiB ancestors and tails.
The engine prefix cache and external L1 are emptied before each family;
the owned NVMe file and its ancestor index remain. The staggered control
does not clear caches between leader completion and follower arrival.

Allocation/free hooks retain each simultaneously referenced ancestor block
table, request status and computed-token count. Reserved destinations are
reported separately from running/completed requests whose ancestors are
ready. This avoids inferring simultaneous sharing from a block ID reused
after a request finishes. Backend receipts independently count complete
chunk reads and bytes on the required native path. They are not physical
NVMe command traces or HBM-bandwidth counters. Scheduler connector calls also
retain the local-hit count and the newly required external-token count, so
backend reads need not be mistaken for transfers into new engine pages.
These instrumented eight-token
samples do not qualify serving latency or general model quality.

The same token, chosen-logprob, restored-KV, native-route and drained-ownership
checks apply. No singleflight, family prefetch or new allocator is enabled.
Recheck with `KMESH_RESTORE_SECTION=coldburst KMESH_RESTORE_OUTPUT=/path/to/results
make kmesh-restore-verify`. CPU tests in `tests/test_kmesh_coldburst.py` reject
null blocks, undrained requests and confusion between later reuse and live
sharing.

The October 17 eight-token run passes all gates: 216 continuations and 1,728
chosen-token log probabilities match recomputation exactly, including warmups.
Every retained family reads one 48 MiB ancestor per child, even in the
leader-first control. That control has only one external engine load and
three local-prefix followers. Most bursts also share one live ancestor;
one of three retained eight-child DMA-BUF families uses three copies.
Ordinary lookup starts prefetch before local-hit subtraction, while the
optional eager-prefetch flag is off. Storage reads and GPU destinations must
therefore be assessed separately. See the
[per-family results](https://knlp.io/kmesh.html#coldburst).

The larger twenty-family repeat also passes: all 828 continuations and 6,624
chosen-token log probabilities match recomputation exactly, including warmups.
Every family still reads an ancestor per child at the backend. Ready engine
copies range from one to four in concurrent bursts; both leader-first arms
share one copy in all twenty retained families. These are conditioned samples
from one engine per mode/arrival, not a general race-rate or latency result.
The page retains both sample sets and their limits. To reproduce the larger
set, invoke `run.py` in the same queued wrapper with `--section coldburst
--rounds 1 --adapter-samples 1 --serving-samples 20`, alongside the prepared
model, source, provider and fresh output arguments. The defconfig retains its
three-family default.

## Recheck the evidence

Raw provenance includes exact source commits, model artifact hashes, native
binaries, package versions, runtime and workload choices. Keep these detailed
records private when they contain local paths or device identity. The standard
library verifier independently rejects missing samples, wrong payload routes,
unfinished native operations, incorrect restored bytes, mismatched model KV,
output changes and retained key or buffer ownership.

```sh
python3 tools/reproduce/kmesh_restore/verify.py /path/to/results --section adapter
KMESH_RESTORE_OUTPUT=/path/to/results make kmesh-restore-verify
python3 -m unittest discover -s tests -p test_kmesh_restore.py -v
```

For a full-pipeline attempt that stops during serving qualification, inspect
`smoke-serving-verify.log` and repeat the CPU gate with
`verify.py /path/to/results --stage smoke --section serving`. Verification of
the complete measured pipeline requires all formal serving cells; it must
reject a capture that stopped at smoke.

Percentiles use linear interpolation at `(n - 1) * p / 100`. First-token samples
from siblings in one burst share a scheduling episode; they are not independent
families. Report sample counts and paired-round variation with any latency
claim, and keep failed attempts separate from accepted timing.

## Exact-prefix load coalescing

`make defconfig-kmesh-singleflight-cuda` selects a separate, pinned LMCache
branch with the original external-load singleflight gate. Singleflight means
one request loads an ancestor while compatible followers wait for an ordinary
vLLM local prefix-cache hit. vLLM remains at the serving-replay pin. The gate
is disabled by default outside this profile.

Set the same `KMESH_RESTORE_*` input paths as the cold-arrival profile, pointing
LMCache at `2026-10-17-feature-kmesh-external-singleflight`. Run
`make kmesh-restore-plan`, then run `make kmesh-restore` with
the prepared runtime. Use `make kmesh-restore-verify` for a GPU-free replay of the
saved evidence. The profile runs disabled and enabled controls with identical
sources, native binaries and model artifacts, twenty retained families plus
three warmups per case, burst fanout 2/4/8 and a completed-leader fanout 4.

Both controls retain all eight generated tokens and chosen-token log
probabilities against recomputation, producer/restored model-KV checksums,
native route and byte receipts, and live block ownership. The enabled arm
must show one external engine load and one ready ancestor per family, local
hits for the followers, no remaining registry state, and at least one actual
singleflight join. A run without joins does not qualify the mechanism.

This gate follows LOOKUP, which starts NVMe prefetch before subtracting local
engine hits. Backend reads are still reported independently; sharing one GPU
ancestor does not establish fewer NVMe reads or lower serving latency.
