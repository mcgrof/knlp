# io_uring DMA-BUF enablement and validation

This profile prepares the public kernel and application revisions used by
the DMA-BUF review. It builds the kernel and exposes separate CPU, host
storage, MP registration and GPU ordering checks. It does not turn a
software admission change into hardware support. Test both payload
directions and preserve the independently checked results.

The older `pd-smoke`, `pd-lmcache-*` and `pd-nixl-*` defconfigs target
the historical reproduction lane. They
do not prepare this kernel or select these current MP tests. Use:

```sh
make defconfig-io-uring-dmabuf
make                       # print the plan, without network or hardware I/O
make dmabuf-fetch           # fetch and check all four exact source commits
```

`sources.json` pins Linux, LMCache, native NIXL IO_URING and vLLM. Source
checkouts are detached and isolated under `.cache/io-uring-dmabuf/src`.
The application pins include the contiguous CUDA staging registration,
repeated receiver-close and GCC15 configuration-test corrections.
Existing checkouts must match the pin and have no tracked edits. A fetch
does not update the user's other trees. Edit `.config` or use menuconfig
to change the work directory, build jobs, base config and scratch path.

## Kernel preparation

Install the usual kernel build dependencies through your distribution:
a matching compiler, make, binutils, bc, bison, flex, OpenSSL/ELF/DWARF
development headers, gawk, pahole and compression tools. The host-memory
selftest also needs the system Linux UAPI headers. Python stages use the
standard library; application tests need the dependencies below.

```sh
make dmabuf-kernel-config
make dmabuf-kernel-build
```

Configuration starts from `/boot/config-$(uname -r)` or `/proc/config.gz`,
or `CONFIG_KNLP_DMABUF_BASE_CONFIG` when supplied. It retains platform
drivers, adds the required DMA-BUF/io_uring, NVMe and XFS/ext4 options,
clears missing distribution certificate filenames and uses the separate
`-knlp-dmabuf` release suffix. Review the resulting config, especially
NIC/root-storage drivers, tracing and your module/image signing policy.
Existing build configs are not overwritten; use a separate work directory
for a new configuration. No ACS/IOMMU or peer-policy override is applied.

Installation is a separate privileged action:

```sh
make dmabuf-kernel-install
# Select this test kernel in the bootloader and reboot manually.
make dmabuf-doctor
```

The install target runs `modules_install install` with stripped modules.
Bootloader/initramfs hooks are distribution-specific and may update the
boot menu; verify the selected default before rebooting and retain your
working rescue kernel. The runner issues no reboot or boot-selection
commands. When a local build release exists, doctor
requires the running release to match it. Otherwise doctor records that
source identity was not checked. A positive queue capability is only a
prerequisite; it does not prove an exporter or a working GPU/NVMe route.

## Host-memory storage checks

Set `CONFIG_KNLP_DMABUF_SCRATCH` to an existing owned directory on physical
NVMe-backed XFS/ext4. The workflow accepts directories, not raw devices.
Tests create and initialize new files and retain them with their logs.
The default system-heap exporter needs access to `/dev/dma_heap/system`;
`CONFIG_KNLP_DMABUF_EXPORTER="udmabuf"` instead needs `/dev/udmabuf`.
Load the configured modules and grant appropriate access on your system.

```sh
make dmabuf-selftest-build
make dmabuf-storage-test
```

The pinned kernel's real C selftest checks writes, reads and independent
host readback at 4 KiB and 8 MiB. These are **host-memory DMA-BUF** results.
They do not prove GPU placement or serving. Logical request size is not
NVMe command size: use `nvme_setup_cmd` tracing as described in the kernel
selftest to verify splitting and MDTS. This workflow does not claim that
an 8 MiB operation reached storage as one command.

Configure `CONFIG_KNLP_DMABUF_GPU` and `CONFIG_KNLP_DMABUF_NVME` with exact
PCI addresses to run `make dmabuf-peer-check`. This is a policy diagnostic,
not a transfer test. The source-pinned model cannot discover an edited
kernel admission table. It runs with the optional NVIDIA metadata query
disabled; inspect unknown/rejected outcomes rather than treating exit zero
as pair qualification. The kernel guide documents additional diagnostics.

## LMCache CPU, MP and GPU checks

Create a dedicated Python environment with the Torch version and CUDA/HIP
toolchain required by the pinned LMCache source. Follow that source's build
requirements; provide Rust/Cargo for the raw-block extension. Use this
environment as `PYTHON` in make, or activate it before running make.
Install pytest and the pinned source's test dependencies as well.

```sh
make dmabuf-lmcache-install PYTHON=/path/to/venv/bin/python
make dmabuf-cpu-test PYTHON=/path/to/venv/bin/python
```

Installation builds LMCache from its pinned editable tree and separately
installs `rust/raw_block`. The test preflight requires the extension to
import and LMCache to resolve from the configured source tree. CPU tests
exercise MP shared-file publication, adoption, ACK/extent lifetime and HTTP
controls. They need neither a GPU nor the experimental kernel. They are
not model-serving P/D or GPU transport qualification.

On the booted custom kernel, after preparing the NVMe scratch directory:

```sh
make dmabuf-mp-host-test PYTHON=/path/to/venv/bin/python
make dmabuf-gpu-test PYTHON=/path/to/venv/bin/python
```

MP host tests exercise real registered host-memory I/O and ownership. They
require `/dev/udmabuf`. GPU tests explicitly enable the pinned V2/V3
connector/exporter ordering and padding checks on fresh files. Coordinate
GPU ownership through the host's reservation mechanism before that target.
Native export is the default. Select
`CONFIG_KNLP_DMABUF_GPU_PROVIDER="vulkan_rm"` only after building LMCache's
optional helper with its matching public NVIDIA SDK and build flags, as
described in its `raw_block_dmabuf.rst` guide. This is explicit provider
selection, not automatic fallback. It does not repair platform routing.

JUnit reports reject skipped, empty or failed selections as qualification.
Commands and output remain under the configured `results` directory.
The workflow does not time out and kill owners with potentially live DMA.
Inspect retained state before retrying a failed hardware test.

## Current prefill/decode test targets

The `pd-storage` defconfig selects the same pinned sources and plan-only
workflow as `io-uring-dmabuf`. It does not select the historical `pd-run`
smoke runner. After fetching and preparing the application environments:

```sh
make defconfig-pd-storage
make dmabuf-fetch
make pd-lmcache-test PYTHON=/path/to/lmcache-venv/bin/python
make pd-mp-test PYTHON=/path/to/lmcache-venv/bin/python
make pd-vllm-test PYTHON=/path/to/vllm-venv/bin/python
```

`pd-lmcache-test` checks publication barriers, generation and manifest
identity, ACK (acknowledgement) and extent lifetime, restore coverage,
notification ordering, launcher admission and causal-trace validation.
`pd-mp-test` runs real shared-file MP publication/adoption and HTTP
controls. `pd-vllm-test` runs the NIXL storage connector's CPU coordinator,
configuration, descriptor planning and pack/scatter contracts. Its NIXL
transfer uses test doubles; it does not run a NIXL storage plugin. The
selection explicitly deselects exporter and device-transfer cases. It
loads the connector unit-test fixtures without the global model fixtures.
All three hide GPUs and reject unexpected skips. They need application
builds and test dependencies, but do not need the custom kernel or NVMe.

`make pd-cpu-test` runs those three selections in order when one Python
environment can import both pinned projects. Each project must resolve
from its configured checkout; an unrelated installed version is rejected.
Use the separate targets with separate environments otherwise. Reports
and command logs use distinct target names under the work directory.
`make pd-test-plan` prints source pins and stages without running them.

Hardware checks remain explicit: `dmabuf-mp-host-test` checks MP host
registration and `dmabuf-gpu-test` checks GPU connector ordering. These
checks do not run model-serving P/D. Reserve the GPU before the latter.
The public LMCache `docs/source/mp/l2_storage/raw_block.rst` and
`examples/disagg_prefill/1p1d_raw_block/README.md` describe the distinct MP
and in-process serving configurations. Full writer/reader model launch,
recompute-oracle comparison, failed-load rejection and pressure runs are
not automated by these targets.

For an existing **single-host in-process** serving trace, invoke the real
pinned LMCache checker:

```sh
make pd-timeline-check PD_TIMELINE=/path/to/serving.jsonl
# A deliberately host-staged trace must name its actual payload path:
make pd-timeline-check PD_TIMELINE=/path/to/serving.jsonl \
    PD_REQUIRED_PATH=host_fixed
```

The default requires `dmabuf_fixed` payload provenance. The checker rejects
missing/empty receipts, broken causal order and the wrong payload path.
Its JSON report is retained alongside the command log. This validates the
trace, not model quality, and does not adapt the checker to MP HTTP or
cross-host clocks. Never pass a synthetic contract fixture as a serving
qualification result.

## Scope still requiring separate work

NIXL and vLLM sources are pinned and fetched; this profile does not automate
their builds or coordinate full model-serving MP P/D. Use their source
build guides and LMCache's public MP raw-block guide for writer/reader
configuration and publication controls. Retain an existing NIXL baseline.
Cross-host shared storage, reset/crash recovery, broader model quality and
performance remain separate gates. No defconfig makes an unsupported
exporter, PCIe route or block driver supported.
