# TokenDance-style shared-update baseline

This experiment checks whether related requests can recover cached attention
keys and values together while preserving each request's serial recovery
result. Five synthetic prompts contain the same segments at different token
positions. One request fills the cache; four siblings then recover those
segments. A matching prefix cannot explain the reuse because the shared
segments occupy different absolute positions.

The harness runs the real vLLM and LMCache implementations from the pinned
component branches. It records recovery calls, request-local selection masks,
ownership, cleanup, raw K/V tensors and logits. An independent verifier reads
the ledgers and tensor files again; a driver's self-reported success is
insufficient. A separate CPU command measures the actual Master–Mirror family
codec, which stores one full cache and exact differences for its siblings.

This is a bounded reproduction of selected TokenDance mechanisms. It does not
reproduce the paper's model sizes, agent traces, fused sparse transfer,
throughput, or full serving system. The component implementations are available
in the fork branches linked below. The raw reference capture is retained
separately and is not distributed with knlp. The CPU trace profile runs without
the component checkouts; the model experiment requires them and the prepared
native environment described below.

[CPU profile](#start-with-the-cpu-profile) ·
[Component and runtime pins](#pin-the-component-checkouts-and-native-environment) ·
[Current H100 profile](#current-cuda-profile) ·
[GPU execution](#run-the-eight-gpu-cases) ·
[Capture verification](#recheck-an-existing-capture-and-measure-encoded-bytes) ·
[Results](#reference-result-and-its-limits)

## Start with the CPU profile

From the knlp root, using Python 3.10 or newer and GNU make:

```sh
make defconfig-tokendance
make tokendance-plan
make
make tokendance-test
```

`defconfig-tokendance-cpu` selects the same CPU profile explicitly. Default
`make` generates and authenticates the exact five-request trace, including its
structural-overlap oracle. It executes no model, KV recovery or GPU work.
`result.json` records `kind: synthetic_trace_only` and
`executed_kv_reuse: false`. This is workload validation, not a recovery result.
Kconfig uses `kconfiglib` when available; the supplied defconfigs also work
through knlp's fallback that copies the preset directly.

Every execution needs a new output directory. Set `TOKENDANCE_OUTPUT` to an
absolute path to preserve multiple runs. The runner refuses to overwrite a
previous result. Failed runtime executions retain `failure.json` and logs;
they must not be described as passes.

## Current CUDA profile

The current profile ports the same eight-case experiment to newer vLLM
and LMCache branches that contain DMA-BUF and prefill/decode support.
This experiment still uses LocalCPU storage: it tests recovery correctness,
not GPU-direct storage latency. The original trace, numerical gates and
model revision remain unchanged. Each case runs in a fresh process on one
H100 with BF16, eager execution and Triton attention.

Both component branches are named
`2026-10-16-feature-kmesh-current-validation`. Exact source commits and
Git trees are in `source_pairs.current` in [profile.json](profile.json).
Its `current_cuda` section pins the image, Python and CUDA packages,
GPU geometry and compiled vLLM extension digests. LMCache builds from the
pinned source; its native extension digests are recorded and must agree
across all cases. The Python packages and official vLLM wheel are fixed in
[requirements-cuda.lock](requirements-cuda.lock).

Inside the pinned image, install Git, a C++ compiler, libnuma development
headers, `uv`, and the CUDA 13.0 compiler, runtime development headers and
CCCL headers. The runtime image alone does not include these build tools.
Set `CUDA_HOME` to that toolkit and put its `bin` directory on `PATH`.
Then prepare a fresh directory:

```sh
bash tools/reproduce/tokendance/prepare_cuda.sh /workspace/kmesh-runtime
. /workspace/kmesh-runtime/run.env
make defconfig-tokendance-cuda
make tokendance-plan
TOKENDANCE_OUTPUT=/workspace/tokendance-h100 make tokendance-gpu
```

Run the GPU command on an available GPU with the prepared runtime and
pinned image declaration emitted by `run.env`. Set the source and model
paths explicitly. The wrapper does not rent compute. The GPU and runtime
must match the profile; another device requires separate qualification.

Preparation clones the public branches, checks their exact trees, builds
LMCache's CUDA extension, links only matching wheel binaries into ignored
vLLM source paths, and downloads the pinned model. It also prepares a
separate vLLM checkout for the allocator check described in
[the allocator README](../kmesh_forks/README.md). The generated environment
contains paths, not credentials. The model snapshot occupies its canonical
path under `/scratch/hf`; the harness authenticates every required file.
The compiler defaults to architecture 9.0, four build jobs and one NVCC thread.
`TORCH_CUDA_ARCH_LIST`, `MAX_JOBS` and `NVCC_THREADS` can override these build
inputs for a separate hardware qualification. They do not change the H100
runtime profile's acceptance requirements.

The native run writes the same raw tensors, semantic ledgers, independent
`verified.json` and offline `codec/codec.json` as the historical profile.
It runs CPU contracts and resolved-config checks before loading the model.
Offline model flags prevent model downloads during execution; this native
launcher does not provide Docker's network isolation or read-only mounts.
Its synchronized capture timings are not performance measurements.

For a captured current run, select `TOKENDANCE_SOURCE_PAIR=current` when
using `make tokendance-verify`. Historical captures keep their original
source and runtime checks below.

## Pin the component checkouts and native environment

These pins describe the historical ROCm profile. Use the current CUDA
profile above for the H100 retest.

Both published component branches are named
`2026-10-16-feature-tokendance-reproduction`. A checkout must be clean and match
both the commit and tree in [profile.json](profile.json):

| Component | Dated commit | Native base |
| --- | --- | --- |
| [vLLM branch](https://github.com/mcgrof/vllm/tree/2026-10-16-feature-tokendance-reproduction) | [`18f90213f45783bad6b7acd7bc43cbba957a69f8`](https://github.com/mcgrof/vllm/commit/18f90213f45783bad6b7acd7bc43cbba957a69f8) | `454ea5b52611c933e00581723e2db56f0144cea7` |
| [LMCache branch](https://github.com/mcgrof/LMCache/tree/2026-10-16-feature-tokendance-reproduction) | [`aa4984b57f78b9d1dce2e8d870ce7aa4716cd341`](https://github.com/mcgrof/LMCache/commit/aa4984b57f78b9d1dce2e8d870ce7aa4716cd341) | `499c830b3fda473388a2b7d6bb002dd69f5d7867` |

Clone only the intended branches, then detach at the committed pins so later
branch updates cannot silently change the experiment:

```sh
git clone --single-branch --branch 2026-10-16-feature-tokendance-reproduction \
  https://github.com/mcgrof/vllm.git ../vllm-tokendance
git clone --single-branch --branch 2026-10-16-feature-tokendance-reproduction \
  https://github.com/mcgrof/LMCache.git ../lmcache-tokendance
git -C ../vllm-tokendance checkout --detach 18f90213f45783bad6b7acd7bc43cbba957a69f8
git -C ../lmcache-tokendance checkout --detach aa4984b57f78b9d1dce2e8d870ce7aa4716cd341
```

Set `TOKENDANCE_VLLM_SOURCE` and `TOKENDANCE_LMCACHE_SOURCE` to the absolute
paths of those checkouts. These are experimental fork branches; publication
does not imply an upstream merge or qualification of another runtime.

The native environment is an existing Docker volume mounted read-only at
`/opt/kmesh-target`. It must contain `/opt/kmesh-target/bin/python` and the
following exact packages. A volume name alone does not establish compatibility.
The resolved-profile preflight also authenticates the two compiled vLLM
extensions against their SHA-256 digests in
[run_online_serial_pic_gate.py](harness/run_online_serial_pic_gate.py).

| Input | Required value |
| --- | --- |
| Container | `rocm/vllm-dev@sha256:9e531214b46ebabcfc0ee0888324533de0d90c28a828d5803f7b381179d8457c` |
| Python | `3.12.13` |
| PyTorch / HIP | `2.11.0+gitd0c8b1f` / `7.2.53211` |
| vLLM wheel | `0.26.1rc1.dev221+g454ea5b52.rocm723` |
| Transformers / tokenizers | `5.14.1` / `0.23.0rc0` |
| safetensors | `0.8.0` |
| sortedcontainers / aiofile / caio | `2.4.0` / `3.9.0` / `0.9.25` |
| Model | `Qwen/Qwen2.5-0.5B-Instruct` |
| Model revision | `7ae557604adf67be50417f59c2c2f167def9a775` |
| GPU profile | One visible `gfx1100` device, 48,301,604,864 bytes |
| Inference | BF16, eager, `TRITON_ATTN`, prefix cache off, 32-token blocks |
| Recovery | LocalCPU storage, check layer 1, recompute ratio 0.15 |

The component Python sources overlay the matching compiled wheel; the wrapper
does not build or install packages. LMCache uses its pure-PyTorch device-ops
shim for this LocalCPU experiment. The separate optional native LMCache
extension is outside the profile. Docker and the prepared volume are required
prerequisites, not artifacts downloaded by `make`. A different native build,
CUDA runtime, model or GPU needs a separately qualified profile rather than
removing the identity checks. This harness does not exercise DMA-BUF or storage
prefill/decode disaggregation.

Provide a local model snapshot containing the seven files authenticated by
`profile.json`. Its host directory is arbitrary: the wrapper resolves each
file's symlink and mounts the authenticated files at the fixed model path
expected inside the container. External snapshot-blob links therefore work,
and extra files cannot change the model inputs. Model files, component sources,
Git object dependencies and the runtime volume are mounted read-only. Network
access is disabled during execution.

## Run the CPU contract checks

These checks import the actual component implementations in the prepared
container and exercise admission, malformed metadata, lifecycle and verifier
rejection cases. They do not load model weights or expose GPU devices.

```sh
TOKENDANCE_VLLM_SOURCE=/path/to/vllm \
TOKENDANCE_LMCACHE_SOURCE=/path/to/lmcache \
TOKENDANCE_RUNTIME_VOLUME=prepared-runtime-volume \
TOKENDANCE_DOCKER='docker' \
TOKENDANCE_OUTPUT=/path/to/fresh-contract-results \
make tokendance-contracts
```

Use `TOKENDANCE_DOCKER='sudo -n docker'` only where that is the configured
container invocation. `contracts.json` contains the individual contract checks;
it is CPU evidence, not qualification of a new GPU/model run.

## Run the eight GPU cases

Select the ROCm profile, inspect the plan, and run the command on a
qualified GPU host. Pass the source, model and output paths explicitly.

```sh
make defconfig-tokendance-rocm
make tokendance-plan
/usr/bin/env \
  TOKENDANCE_VLLM_SOURCE=/path/to/vllm \
  TOKENDANCE_LMCACHE_SOURCE=/path/to/lmcache \
  TOKENDANCE_RUNTIME_VOLUME=prepared-runtime-volume \
  TOKENDANCE_MODEL_DIR=/path/to/pinned-model-snapshot \
  TOKENDANCE_DOCKER='docker' \
  TOKENDANCE_OUTPUT=/path/to/fresh-gpu-results \
  make tokendance-gpu TOKENDANCE_PY=/usr/bin/python3
```

The launcher keeps a unique container name
for each step and removes that container on interruption; cancellation of the
new GPU wrapper has not yet been qualified under a live GPU run.

The cases run in separate containers: ordinary full recomputation, serial
recovery, collective recovery in forward and reversed request order, cold miss,
zero sharing, incompatible partition, and injected sibling failure. The last
four check fallback and isolation. Selection masks, attention and destination
slots stay request-local even when rotary and difference calls batch together.
The independent numerical comparison uses ordinary, serial and forward
collective captures; reversed order is checked by semantic ledgers and output.

Outputs include `plan.json`, `sources.json`, resolved `preflight.json`,
per-case `arms/*/artifacts/`, `verified.json`, `SHA256SUMS`, and `report.json`.
`SHA256SUMS` authenticates `verified.json`; that summary records the result and
ledger digests that bind the captured input artifacts. The report is accepted
only after the complete independent gate passes. Capture instrumentation
synchronizes the device and writes tensors, so its timings are unsuitable for
performance comparisons.

## Recheck an existing capture and measure encoded bytes

Use a Python environment with PyTorch and safetensors to recheck all eight
cases on CPU. `TOKENDANCE_BUNDLE` names the capture directory containing
`arms/`; it is not a Git bundle. The output directory must be outside it.

```sh
TOKENDANCE_BUNDLE=/path/to/eight-case-capture \
TOKENDANCE_SOURCE_PAIR=reference \
TOKENDANCE_OUTPUT=/path/to/fresh-verification \
make tokendance-verify TOKENDANCE_PY=/path/to/torch-env/bin/python
cd /path/to/fresh-verification
sha256sum -c SHA256SUMS
```

Use `reference` only for the retained capture made with vLLM `2f8d7fc1dce4`
and LMCache `d416824a6f89`. Use `dated` for a new capture on the dated pins.
Both accepted pairs are explicit in `profile.json`; a different source pair
is rejected. Verification reloads raw tensor files, checks their digests,
compares K/V bits and logits, and checks masks, order, failure and cleanup
ledgers. The raw reference capture is retained separately and is not included
in knlp.

For actual serialized-byte accounting, return to the knlp root and run:

```sh
TOKENDANCE_VLLM_SOURCE=/path/to/vllm \
TOKENDANCE_LMCACHE_SOURCE=/path/to/lmcache \
TOKENDANCE_RUNTIME_VOLUME=prepared-runtime-volume \
TOKENDANCE_DOCKER='docker' \
TOKENDANCE_BUNDLE=/path/to/eight-case-capture \
TOKENDANCE_VERIFICATION=/path/to/fresh-verification/verified.json \
TOKENDANCE_OUTPUT=/path/to/fresh-codec-results \
make tokendance-codec
```

This CPU command invokes the dated LMCache `encode_family` and `decode_family`
on the four captured caches, with content-aligned shared spans. It writes
`family.bin`, reads it back, requires every BF16 bit to match, and checks that
a corrupted payload is rejected. `codec.json` records actual encoded bytes
including codec metadata, dense bytes, and source digests. It measures offline
storage representation, not live GPU page sharing or transfer speed.

## Reference result and its limits

On 2026-10-16 the portable verifier rechecked the retained 2026-08-03 GPU
capture on CPU. All eight cases passed, including fallback, order stability,
request-local mask agreement and cleanup. For the four recovered siblings,
serial and collective K/V were bitwise identical and their maximum logit
difference was zero. Each prompt had 512 tokens; 416 were retrieved, so the
realized shared fraction was 81.25% after segment rounding.

Ordinary full recomputation and serial recovery had a maximum absolute logit
difference of 0.55859375. The ordinary/recovery numerical diagnostic failed;
it is explicitly excluded from the acceptance claim. The bounded output-token
check passed, but one generated token is not a model-quality evaluation.
The successful result is collective equivalence to serial recovery, not
equivalence to full recomputation.

The dated LMCache codec encoded those four captured caches into 17,506,917
bytes versus 25,165,824 dense bytes: 69.566% of dense size, or 30.434% fewer
bytes. All four restored exactly and the corruption control passed. The dense
size is four members × two K/V planes × 512 tokens × 24 layers × 128 features
× two BF16 bytes. CPU measurement used the pinned container with two OpenMP
threads and no GPU devices. It establishes bytes and exact restoration only.

The dated source pair passed the CPU contract suite. No fresh GPU capture on
those dated branches was run as part of this integration. No latency,
throughput, end-to-end storage offload, cascade-attention benefit, GPU memory
capacity, agent-task quality or paper-scale performance result follows from
these checks. See [the KMesh page](../../../docs/kmesh.html).

The current CUDA profile was freshly retested on one H100 on 2026-10-16,
using the rebased branches pinned in `source_pairs.current`. All eight cases
and the independent verifier pass. Serial and collective K/V match bit for
bit, with zero maximum logit difference. The ordinary/serial diagnostic still
fails, with maximum absolute logit difference 0.5625. The offline codec again
uses 17,506,917 bytes, restores all four members exactly and rejects corruption.
This qualifies the current LocalCPU recovery baseline; it establishes no
DMA-BUF or prefill/decode transfer result. The sanitized
H100 retest record binds these
findings to their source commits and verifier digest.
