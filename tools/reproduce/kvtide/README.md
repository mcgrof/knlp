# KVTide — NVMe-KV target + initiator bench harness

KVTide is a Kconfig-reproducible harness for portable, vendor-neutral
tiered KV-cache I/O work. It stands up a software NVMe key-value target
(the SPDK `kvmalloc` KV bdev exported over NVMe-oF/TCP on loopback) and
drives it with one or two initiators over the exact same matrix:

- **spdk** — `spdk_nvme_perf` in KV mode, userspace kernel-bypass,
  connecting straight to the target.
- **xnvme** — `xnvme_kv_perf` (vendored in `src/`), io_uring passthrough
  commands through the kernel nvme-tcp host stack and a `/dev/ngXnY`
  KV namespace char device.

Identical workload, identical CPU accounting (busy jiffies on all cores
except the target's busy-polling reactor cores), one CSV — so an A/B
comparison between the userspace and kernel data paths is one sort away.

No KV hardware is required; everything runs inside a QEMU guest.

## Quick start

```
make defconfig-kvtide-ab      # or -spdk / -xnvme / -nixl
make
```

Plain `make` runs doctor → fetch → build → target-up → bench → report.
The target stays up afterwards so cells can be re-run; tear it down with
`make kvtide-target-down`. A single cell:

```
tools/reproduce/kvtide/bench.sh xnvme store 64 4096
```

All knobs live in the "KVTide NVMe-KV bench harness" Kconfig menu:
source pins, target address/NQN/core mask, hugepages, and the bench
matrix (ops, value sizes, queue depths, seconds per cell, initiator CPU
pin). Host-specific overrides append to `.config`:

```
echo 'CONFIG_KVTIDE_SRC_DIR="/home/me/kvtide"' >> .config
```

## What gets fetched

| Component | Source | Pin |
|---|---|---|
| SPDK KV target stack | SPDK Gerrit | `refs/changes/07/28307/12` |
| xNVMe | github.com/xnvme/xnvme | `a5bf2a65` |
| NIXL + XNVME_KV plugin (optional) | github.com/mcgrof/nixl | `20260717-xnvme-kv` |

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

Bare metal: kdevops `DECLARED_HOSTS` flips any config to existing
hosts; the harness itself has no guest assumptions beyond core count.

## Results

CSV columns: `initiator,op,qd,vsize,iops,MBps,lat_us,p99_us,
init_cpu_cores`. The latency column is p50 for xnvme but the arithmetic
mean for spdk (`spdk_nvme_perf` reports no percentiles on its Total
line); p99 is xnvme-only. Compare iops/MBps/cpu across initiators;
latency shape only within one initiator.

Results land in `$KVTIDE_SRC_DIR/results/` (outside the repo);
`kvtide-bench-latest.csv` symlinks the newest run and
`make kvtide-report` renders the table plus per-cell A/B IOPS ratios.

## Status

Validated: the full A/B matrix inside a QEMU guest (Debian, kernel
6.12, loopback TCP), including the NIXL plugin unit + integration
tests. That recorded run predates one harness fix worth knowing when
comparing against it: its driver passed the SPDK core selection as a
bare number, which SPDK parses as a hex mask, so the spdk cells ran
three workers (measured ≈1.9 cores in the CSV) instead of one; the
harness now pins spdk with the exact `-c [N]` core-list syntax.
Per-core comparisons against the recorded CSV remain honest via its
`init_cpu_cores` column. Known behavior: large-value stores at high
queue depth are sensitive to the kernel nvme-tcp solicited-write (R2T)
path when writes exceed the target's in-capsule data size — now a
knob, `KVTIDE_TARGET_INCAPSULE`; the userspace initiator does not take
that path. Provisional: bare-metal runs.
