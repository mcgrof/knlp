# KVTide — KV-cache storage I/O bench harness

KVTide is a Kconfig-reproducible harness for portable, vendor-neutral
tiered KV-cache storage I/O work. It has two separate benchmark modes.
The original mode stands up a software NVMe key-value target and compares
NVMe-KV commands over loopback TCP. The physical-PCIe mode drives ordinary
NVM reads on unused real NVMe namespaces. Keeping the modes separate avoids
confusing an NVMe command set with the block reads used by AISIO, SPDK, and
the Linux premapped-buffer path.

The software NVMe-KV mode drives the SPDK `kvmalloc` KV bdev through:

- **spdk** — `spdk_nvme_perf` in KV mode, userspace kernel-bypass,
  connecting straight to the target.
- **xnvme** — `xnvme_kv_perf` (vendored in `src/`), io_uring passthrough
  commands through the kernel nvme-tcp host stack and a `/dev/ngXnY`
  KV namespace char device.

The physical-PCIe mode compares current upstream implementations through:

- **upcie** — xNVMe's minimal userspace PCIe/NVMe driver with host buffers.
- **upcie-cuda** — the same host-initiated command path with read buffers in
  CUDA memory, so the NVMe controller transfers payloads directly to the GPU.
- **spdk** — current standalone `spdk_nvme_perf` using its userspace driver.
- **linux** — current xNVMe `xnvmeperf` through the kernel NVMe driver and
  `io_uring_cmd`.
- **fixed** — the same kernel command interface with an ordinary registered
  userspace buffer.
- **premap** — a kernel-owned `blk_iobuf` pool buffer DMA-mapped once, then
  reused with `IORING_URING_CMD_FIXED`; it never pins userspace memory.

Beating SPDK across small and large reads is the engineering goal. It is not
an empirical claim. The report says that premap won only for cells where the
recorded median exceeds SPDK, and keeps every cell where SPDK wins.
The current `spdk` and `premap` arms use different benchmark programs, so
their ratio compares complete software paths rather than isolating one
backend function. The `upcie` and `linux` arms both use `xnvmeperf` and provide
the closest same-tool comparison.

No KV hardware is required for the software mode. Physical-PCIe results need
real disposable NVMe namespaces; QEMU is useful only for functional checks.

## Quick start

For the software target:

```
make defconfig-kvtide-ab
make
```

Plain `make` runs doctor → fetch → build → target-up → bench → report.
The target stays up afterwards so cells can be re-run; tear it down with
`make kvtide-target-down`. A single cell:

```
tools/reproduce/kvtide/bench.sh xnvme store 64 4096
```

All knobs live in the "KVTide storage bench harness" Kconfig menu:
source pins, target address/NQN/core mask, hugepages, and the bench
matrix (ops, value sizes, queue depths, seconds per cell, initiator CPU
pin). Host-specific overrides append to `.config`:

```
echo 'CONFIG_KVTIDE_SRC_DIR="/home/me/kvtide"' >> .config
```

For the current physical-PCIe comparison, first boot a kernel carrying the
premap work and a translating input-output memory management unit (IOMMU).
Then name only unused NVMe controllers and explicitly allow their drivers to
be rebound:

```
make defconfig-kvtide-pcie
echo 'CONFIG_KVTIDE_PCIE_BDFS="0000:41:00.0 0000:42:00.0 0000:43:00.0 0000:44:00.0"' >> .config
echo 'CONFIG_KVTIDE_PCIE_ALLOW_REBIND=y' >> .config
make
```

`pcie-bind.sh` checks that each PCI function is an NVMe controller and refuses
to detach it when its block devices are mounted, used as swap, or held by a
stacked block device or process. The benchmark is read-only and returns every
listed controller to the kernel `nvme` driver on exit. It also restores the original
hugepage count. A restore failure is printed as a critical error. These checks
do not make an OS disk disposable; verify the PCI addresses yourself before
setting the opt-in.

On a translated-IOMMU boot, `pcie-bind.sh` tries to load
`vfio_iommu_type1` explicitly so it is available when the kernel does not
expose vfio device cdevs through iommufd. Loading `vfio-pci` alone does not
make the legacy type1 transport available to xNVMe.

`KVTIDE_PCIE_XNVME_VFIO_MODE` selects `auto`, `iommufd`, or `type1` and is
recorded in `run.meta`. Keep `auto` unless the host exposes device cdevs but
cannot initialize xNVMe's iommufd path. Select `type1` explicitly on that host
instead of presenting a fallback run as an iommufd result.

`defconfig-kvtide-pcie-linux-baremetal` additionally builds and installs the
public `blk-iobuf-pool-v5-premap-iova` kernel. It writes the required IOMMU,
non-multipath, pool-order, and pool-size settings into a GRUB fragment and
stops for a reboot before benchmarking. This defconfig deliberately leaves
PCI rebinding disabled until the operator supplies the two lines above.

The premap allocation can fall back to an ordinary registered buffer when the
kernel cannot retain an I/O virtual address. KVTide does not accept that silent
fallback as evidence. Before recording a premap cell, it submits a read one
logical block larger than every namespace's ordinary
`queue/max_hw_sectors_kb` limit. The run stops unless that read succeeds. The
validation command, ordinary limit, probe size, and output remain beside the
CSV. Crossing the ordinary limit proves that the request used the retained
mapping and the separate premapped request limit on this kernel.

The two workload profiles answer different questions:

- `aisio` runs random reads at queue depth 128 with 512-byte, 4 KiB, and
  8 KiB commands while increasing the submitting threads from one to four.
  This reproduces the published AISIO synthetic command-rate controls. The
  512-byte cell is meaningful only on a namespace formatted with 512-byte
  logical blocks; KVTide records and enforces the actual logical block size.
  Matching the published platform additionally requires its 16 Samsung
  PM1753 SSDs, H100 system and PCIe topology. The profile cannot manufacture
  that hardware parity, so compare absolute IOPS only after confirming it.
- `kv` runs random reads from 4 KiB through 2 MiB at queue depths 1, 16, 64,
  and 128. These are synthetic size and concurrency crossover points, not a
  recorded production KV-cache distribution. They show where command rate,
  the ordinary DMA-mapping limit, device MDTS, and bandwidth become the
  active constraint. Add trace-derived profiles after preserving their size,
  offset, ordering, pacing, and concurrency fidelity.

The physical mode performs one warm-up and five recorded ten-second runs by
default. It records the median rather than selecting the best repetition.
Global busy CPU time is reported as utilized cores; it includes system work,
so use it to expose a full-stack cost rather than to attribute cycles to one
function. None of the current physical tools reports a compatible latency
distribution. Do not infer tail latency from this harness. Keep one queue per
controller when comparing `fixed` or `premap`; their companion tool does not
yet create multiple queues per controller. Configure at least four controllers
for the default one-to-four-thread AISIO profile. The run rejects a profile
that asks for more threads than its configured CPUs or queues. Each benchmark
program generates its own random offsets. The cells match the random
distribution and controls, not an identical ordered request stream.

## What gets fetched

| Mode | Component | Source selection |
|---|---|---|
| Software KV | SPDK KV target stack | SPDK Gerrit `refs/changes/07/28307/12` |
| Software KV | xNVMe | commit `a5bf2a65` |
| Software KV | NIXL + XNVME_KV plugin | branch `20260717-xnvme-kv` |
| Physical PCIe | xNVMe | resolve `main` at fetch time |
| Physical PCIe | SPDK | resolve `master` at fetch time |

The physical defaults intentionally name upstream branches. Every fetch
refreshes a clean harness-owned checkout to the then-current tip, and refuses
to overwrite tracked local changes. Every run records the resulting commit IDs
in `run.meta`, making the result reproducible even after upstream moves. Pin a
commit in Kconfig when a campaign must keep the same source across later runs.

The SPDK KV command set support (bdev/kvmalloc, nvmf KV namespaces,
`spdk_nvme_perf` KV mode) is in review on SPDK Gerrit; one change ref
carries the whole stack plus its base, so no full clone is needed.
`patches/spdk-kv-stack-local-fixes.patch` fixes a GCC 15/AVX10 build
probe and an rpc_autogen decoder collision; it is applied by default
(`CONFIG_KVTIDE_SPDK_LOCAL_FIXES`).

The target script never runs SPDK `scripts/setup.sh` — no PCI
rebinding, no vfio/uio; local NVMe stays kernel-owned. Only hugepages
are reserved (2 GiB by default).

## Kernel under test (kvtide-ab-linux)

`make defconfig-kvtide-ab-linux` adds a kernel dimension to the A/B: a
`kvtide-linux` stage fetches a Linux tree (shallow, single branch),
bases its config on the running kernel's, enables
`CONFIG_BLK_IOBUF_POOL`, builds and installs it, then **gates** the
pipeline on actually running it — the first `make` stops after the
install and asks for one reboot; the second `make` passes the gate and
runs the bench on the kernel under test. Each bench CSV gets a
`.meta` sidecar recording `uname -r`, so runs on different kernels
stay distinguishable.

The tree and branch default to the blk_iobuf_pool v3 series on
kernel.org and can be overridden as environment variables at defconfig
time (requires python3 kconfiglib; the defconfig deliberately omits
the two symbols so olddefconfig can fill them):

```
# default: the blk_iobuf_pool v3 series
make defconfig-kvtide-ab-linux

# any other tree/branch
make defconfig-kvtide-ab-linux \
    LINUX_TREE=https://git.kernel.org/pub/scm/linux/kernel/git/mcgrof/linux.git \
    LINUX_BRANCH=blk-iobuf-pool-v3
```

This is meant for disposable bench hosts (kdevops QEMU guests) — the
stage installs a kernel and updates the bootloader. The kernel-side
tooling this pairs with is public in the kdevops project:
[blk_iobuf_pool RFC](https://github.com/linux-kdevops/kdevops/blob/main/docs/rfc-20260630-v1-blk-iobuf-pool.html),
[`defconfig-iobuf-nvme`](https://github.com/linux-kdevops/kdevops/blob/main/defconfigs/iobuf-nvme)
and the
[`iobuf_bench` scripts](https://github.com/linux-kdevops/kdevops/tree/main/scripts/workflows/iobuf_bench);
the kernel branch is
[mcgrof/linux `blk-iobuf-pool-v3`](https://git.kernel.org/pub/scm/linux/kernel/git/mcgrof/linux.git/log/?h=blk-iobuf-pool-v3).

## The NIXL stage (optional)

`make defconfig-kvtide-nixl` additionally builds NIXL with the XNVME_KV
storage plugin — an NVMe-KV backend over xNVMe io_uring_cmd — then
`make kvtide-nixl` runs its mock-device unit tests plus the full-agent
integration test against the same KV target device. This validates a
complete transfer-library data path on top of the target the perf
initiators use.

## Running under kdevops (QEMU guest)

knlp ships as a kdevops plugin, and the plugin can provision every
KVTide build dependency. On the kdevops side:

```
make kdevops-plugin-add URL=https://github.com/mcgrof/knlp
# kdevops resolves defconfig fragments from
# ~/.config/kdevops/defconfigs/configs, so stage the plugin's fragment:
mkdir -p ~/.config/kdevops/defconfigs/configs
cp ~/.config/kdevops/plugins/knlp/defconfigs/configs/knlp-kvtide.config \
   ~/.config/kdevops/defconfigs/configs/
make defconfig-<base>+knlp-kvtide
make
make bringup
```

Then on the guest, knlp is already cloned at the configured data path:

```
ssh <guest>
cd /data/knlp
make defconfig-kvtide-ab && make
```

Guest sizing: 2 vCPUs for the target core mask plus one initiator core
(the defaults assume ≥ 5 cores; shrink `KVTIDE_TARGET_CORE_MASK` and
`KVTIDE_BENCH_INIT_CORE` for smaller guests), ~2 GiB hugepages, ~8 GiB
disk for sources and builds.

kdevops also carries blk_iobuf_pool validation tooling (its
`defconfig-iobuf-nvme` and the `iobuf_bench` scripts), so the same guest
workflow can boot a kernel of choice and A/B the kernel-side block-layer
path underneath the xNVMe initiator.

QEMU cannot reproduce PCIe or GPU-direct throughput. Use a declared bare-metal
host for the physical mode. Stage the physical fragment instead and select the
uPCIe CUDA dependency option only on a node where the NVIDIA driver and CUDA
toolkit are already installed:

```
cp ~/.config/kdevops/plugins/knlp/defconfigs/configs/knlp-kvtide-pcie.config \
   ~/.config/kdevops/defconfigs/configs/
make defconfig-<base>+knlp-kvtide-pcie \
    DECLARED_HOSTS="storage-node"
make && make bringup
```

The plugin installs the current uPCIe v0.8.0 `dmabuf-import` and
`iommu-map-pa` DKMS packages when requested. That version is not a performance
pin: xNVMe and SPDK still resolve their configured upstream branches on the
node, and their exact commits go into each result. The plugin does not install
or license CUDA.

## Bare metal

Two paths, both optional:

**Via kdevops declared hosts** — kdevops' own `DECLARED_HOSTS`
mechanism flips any configuration from guest bringup to existing
machines at defconfig time (it selects `SKIP_BRINGUP` +
`KDEVOPS_USE_DECLARED_HOSTS`; the knlp plugin's inventory hook
respects declared hosts):

```
make defconfig-<base>+knlp-kvtide DECLARED_HOSTS="metal1 metal2"
make
make bringup      # provisions the declared hosts over ssh
```

**Direct** — clone knlp on the machine and run the harness; the perf
harness (target + bench) has no VM assumptions beyond enough cores to
keep the target mask and the initiator pin disjoint:

```
make defconfig-kvtide-ab && make
```

The **kernel stage** is the exception: it installs a kernel and
updates the bootloader, so it refuses to run when
`systemd-detect-virt` reports no virtualization. Bare-metal kernel
testing is an explicit opt-in:

```
make defconfig-kvtide-ab-linux-baremetal
```

(equivalently `CONFIG_KVTIDE_LINUX_ALLOW_BAREMETAL=y`), taking the
same `LINUX_TREE` / `LINUX_BRANCH` overrides. kdevops' own
`defconfig-iobuf-baremetal` is the precedent for this kind of
bare-metal block-layer testing.

## Results

CSV columns: `initiator,op,qd,vsize,iops,MBps,lat_us,p99_us,
init_cpu_cores`. The latency column is p50 for xnvme but the arithmetic
mean for spdk (`spdk_nvme_perf` reports no percentiles on its Total
line); p99 is xnvme-only. Compare iops/MBps/cpu across initiators;
latency shape only within one initiator.

Results land in `$KVTIDE_SRC_DIR/results/` (outside the repo);
`kvtide-bench-latest.csv` symlinks the newest run and
`make kvtide-report` renders the table plus per-cell A/B IOPS ratios.

Physical results use a separate timestamped directory and
`kvtide-pcie-latest` link. `results.csv` records profile, arm, benchmark tool,
driver, request size, queue depth, submitting threads, repetition, IOPS,
MiB/s, failed commands, wall time, and system CPU cores. `run.meta` records the
kernel, command line, selected source refs, resolved source commits, and the
goal statement. The directory also keeps PCIe and NUMA topology, CPU
governors, interrupt counters, identify data, and before-and-after NVMe health
logs so frequency, placement, errors, and thermal state remain auditable.
`make kvtide-pcie-report` writes `summary.txt` and compares premap with SPDK
only where both have valid recorded repetitions.

## Status

The physical-PCIe mode has passed configuration, compilation, and static
validation locally. It has not yet produced a real-device performance result.
Treat its first bare-metal campaign as harness validation as well as data: keep
failed commands at zero, check every recorded command and source commit, and
repeat any surprising crossover before making a performance statement.

The software NVMe-KV mode has the following validation history.

Validated in a QEMU guest (Debian, kernel 6.12, loopback TCP,
including the NIXL plugin unit + integration tests) **and on bare
metal** (Latitude m4-metal-small, Ubuntu 24.04, 12 cores, kernel
under test 7.2.0-rc1+ built from `blk-iobuf-pool-v3` via
`defconfig-kvtide-ab-linux-baremetal` — the bare-metal interlock and
the one-reboot gate both exercised on real hardware; full 24-cell
A/B clean, spdk `init_cpu_cores ≈ 1.0` everywhere confirming the
single-worker `-c [N]` pinning).

The earlier guest CSV predates that pinning fix: its driver passed
the SPDK core selection as a bare number, which SPDK parses as a hex
mask, so its spdk cells ran three workers (measured ≈1.9 cores).
Per-core comparisons against it remain honest via `init_cpu_cores`.

Known behavior: large-value stores at high queue depth are sensitive
to the kernel nvme-tcp solicited-write (R2T) path when writes exceed
the target's in-capsule data size — a knob,
`KVTIDE_TARGET_INCAPSULE`; the userspace initiator does not take
that path. The guest's severe 64K-store collapse did not reproduce
on bare metal (33.4k vs 38 IOPS at qd16) — kernel and hardware both
differ, which is exactly the A/B the harness now supports. Open
observation from the bare-metal run: xnvme qd=1 cells latch to ~1 ms
per op on the 7.2-rc kernel (fine at qd ≥ 16); unexplained,
recorded in the archived results.

Kernel-stage caveat seen on Ubuntu server images: a distro dkms
module (bnxt_en) can fail to build against an rc kernel and abort
`make install` mid-way; the stage's install then needs a manual
`update-initramfs -c -k <rel>` + `update-grub`. Check `dkms status`
and your NIC driver before rebooting a remote machine.
