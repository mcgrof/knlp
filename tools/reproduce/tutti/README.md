# Tutti storage evaluation

This knlp workflow builds a pinned public [Tutti](https://github.com/xPU-IO/Tutti)
runtime and a small GPU/NVMe round-trip benchmark. It is an initial storage
evaluation, with a separate synthetic compute/IO overlap stage. It does not
implement an LMCache connector or reproduce the paper's serving results.

The default source pin is `5e48c2ab6deea2ec9972a5f1beb56854f62bf149`.
The public runtime uses GPU-issued NVMe commands, prebuilt PRPs and the SNVMe
driver. Tutti's phrase "GPU io_uring" describes its GPU submission scheme;
it is not the Linux io_uring API. No Linux io_uring DMA-BUF patches are needed
to build this arm. SNVMe and GPU peer-memory support are still required.

The [paper](https://arxiv.org/abs/2605.03375v1) and the evolving public runtime
are different artifacts: the upstream [roadmap](https://github.com/xPU-IO/Tutti/issues/16)
lists SGL and Green Context work separately. Pin and qualify each revision
before claiming a particular paper feature is present.

## Build and configure

Use Linux, Python 3.9+, CMake 3.21+, a C++17 compiler, CUDA 12.6+ and the
upstream gRPC, protobuf, yaml-cpp, uuid and NUMA development dependencies.
PyYAML reads runtime YAML; JSON runtime files work without it. CUDA and the
GPU driver must already be installed. Building SNVMe also needs headers for
the running kernel and a supported GPU peer-memory interface. The pinned
upstream tree has kernel implementations for 5.4-tlinux4, 5.10, 5.15 and 6.8;
compilation on a newer kernel is a qualification task, not an assumed feature.

```sh
make defconfig-tutti-build
make tutti-plan
make tutti-doctor
make tutti-fetch
make tutti-build
```

After selecting a Tutti defconfig, plain `make` runs doctor, fetch and build
sequentially, including under `make -j`. `tutti-build` itself requires a fetched
source. The source workspace defaults to ignored `work/tutti/`. Existing
checkouts must be clean and match the exact configured SHA; the workflow
does not reset local work. Select a new `CONFIG_TUTTI_WORK_DIR` when changing
pins. Building neither loads modules nor starts a daemon. Disabling
`CONFIG_TUTTI_BUILD_MODULE` only skips module compilation, not its runtime need.

The `.kdevops` fragment `knlp-tutti.config` installs Debian-family build
dependencies through both knlp and devconfig roles. Copy it into the usual
kdevops fragment search path and select it with your base configuration,
as with KVTide. Set `WORKFLOW_KNLP_GIT_REF` to the desired public branch/commit.
It deliberately leaves CUDA, driver, kernel headers and device setup to the
host owner. No provisioning or hardware run occurs in `tutti-check`.

## Device setup and smoke run

An operator must first qualify and deploy the pinned upstream
[SNVMe modules and daemon](https://github.com/xPU-IO/Tutti/blob/5e48c2ab6deea2ec9972a5f1beb56854f62bf149/doc/getting-started.md).
Its daemon config is `config/local_nvme_config.yaml`; the application config
is a different schema. Explicitly assign an owned test NVMe, accelerator,
queue grant and filesystem view. Do not run generic hardware CTests on an
unreviewed device. This workflow never infers a PCI device, formats storage,
rebinds a driver, or stops another user's workload.

Copy `runtime.example.yaml` outside the tracked tree, then set its accelerator
ID, daemon endpoint and explicit device ID to match the deployment. Start
with the upstream `ext4-local-nvme` contract on a qualified filesystem; do
not assume XFS equivalence without a separate validation. Keep the daemon
and client GPU enumeration consistent. The directory must be an existing
disposable directory within the **daemon-returned accelerator view**.

```sh
make defconfig-tutti-smoke
make menuconfig
# Set TUTTI_RUNTIME_CONFIG, TUTTI_DIRECTORY and TUTTI_ALLOW_FILE_WRITES.
# Set CUDA architecture and build workspace to match the built artifact.
make tutti-smoke
```

`tutti-smoke` always uses 64 KiB requests, batch depths 1 and 8, a 16 MiB
span, no warmups and one measured round trip per cell. It requires a matching
build receipt. Selecting the smoke defconfig and running plain `make` builds
and runs its equivalent small matrix. A missing write opt-in or runtime
directory stops the pipeline before a hardware run.

The transfer probe creates an exclusive file in a unique run subdirectory,
materializes and fsyncs its extents, registers an aligned GPU allocation,
writes a deterministic pattern, poisons the GPU buffer, reads back and
compares **every byte** on the host. Every repetition, including warmup, is
verified. Payload, registration, initialization and verification are outside
the timed window. Writes measure completion, not durable persistence; there
is no measured NVMe FLUSH. Reads immediately follow writes and can hit the
SSD/controller cache. A small-span run is not steady-state media bandwidth.

Failed or partially accepted I/O, unknown completion, mismatched bytes and
cleanup failures cannot produce a PASS result. Failed payload files remain
for inspection. After a timeout or process death, establish device quiescence
through the deployment owner before reusing files or buffers. Process exit
alone is not proof that all DMA has stopped. Locks serialize this workflow's
own workspace and directory; they do not exclude unrelated workloads.

## Matrix and receipts

```sh
make defconfig-tutti-bench
make menuconfig
# Restore the local runtime, directory, write opt-in and build settings.
make tutti-bench
make tutti-report RUN=/absolute/path/to/a/completed/run
```

Defaults: 64/256/1024/4096 KiB, application batch depths 1/8/32, a 256 MiB
span, one warmup, five measured repetitions and seed 42. The seed shuffles
cell order. `CONFIG_TUTTI_RANDOM=y` additionally permutes object offsets
without replacement. Each cell uses one registered GPU allocation for its
whole span, plus approximately twice that span in host verification buffers.
Choose capacity explicitly before increasing it.

The probe submits a batch, waits for terminal completion and drains it before
submitting the next batch. **Application batch depth is not observed device
queue depth.** The transfer timing includes descriptor preparation, public
submit/wait/release calls and final CUDA stream synchronization. The p50/p99
values describe batch completion times, not individual I/O latencies.
CPU seconds use `RUSAGE_SELF`: the process and its threads. Daemon, IRQ and
system CPU need separate measurement in an A/B experiment.

Each run under ignored `results/tutti/` contains the exact commands, full
configuration, source/binary/build hashes, hardware inventory, raw stdout and
stderr, byte-verified measurements, and a PASS/FAIL manifest. Reports verify
the receipt hashes and complete matrix before summarizing median/min/max
MiB/s and CPU microseconds per operation. Runtime diagnostics on stdout are
retained; only `KNLP_TUTTI_JSON` records enter the measurements. No fabricated
fallback or unverified row is accepted. Receipts include host/device identity
and local paths: review them before publishing; do not commit runs here.

## Extend toward an A/B comparison

First compare matched storage probes: same GPU and NVMe, request sizes,
span, offset order, queue ownership, repetitions, CPU affinity, clock policy,
filesystem and cache state. Add a submit-and-drain mode to the CPU comparator
to match this probe; also retain its continuously refilled mode as a separate
best-achievable comparison. Record actual outstanding commands and Tutti's
queue/worker grants. Never label these schedules equal just because both
have a parameter named "depth". Random A/B tests should replay one serialized
offset trace; the C++ RNG is not a portable cross-language trace contract.

Use full verification as an acceptance gate in both arms. Measure registration
cost separately from amortized transfers, and collect system CPU and GPU
compute interference in addition to process CPU. Use alternating paired
blocks and confidence intervals; five within-process repetitions are only
an initial diagnostic. A drive-saturating throughput tie can still hide
different CPU and SM costs. A storage result alone does not predict TTFT.

`make tutti-overlap` runs the pinned upstream layerwise example with small
configurable dimensions, 50% prefix hits and **fixed** compute duration
(1000 microseconds/layer by default). It requires the same runtime opt-in.
Do not use storage-dependent compute auto-calibration across arms. Its
verification samples K-tensor bytes; it is explicitly labeled as such in
the receipt and is not accepted by `tutti-report`. Extend full K/V checking
before treating it as a correctness or serving-equivalence gate.

Next add an equivalent LMCache storage adapter, exact tensor geometry and
lifetime tests, then fixed-model single-instance serving with identical
cache budgets and hit traces. PD is a separate workload dimension. Multi-drive
striping, real model overlap, scheduler ablations, SGL and Green Context are
later steps requiring separate feature qualification and receipts.

## Hardware-free checks

```sh
make tutti-check
```

These test configuration validation, partial/malformed/tampered receipt
rejection and stage ordering. They cannot establish CUDA compilation, SNVMe
compatibility, GPU/NVMe correctness, or performance. A HOST-header syntax
check is useful when CUDA is unavailable; the benchmark refuses HOST execution.
