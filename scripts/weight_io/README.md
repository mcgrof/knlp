# Weight-loading I/O characterization

These scripts measure what reaching the weights of an open checkpoint
costs at the storage device. A reader asks for some bytes of a
safetensors shard; the file system, the page cache and the block layer
turn that into NVMe commands; the device moves some number of bytes. The
gap between the three is the subject. Each traced run reports:

- **W**, the bytes the selected unit needs (a layer, an embedding row,
  a set of INT4 neurons), computed from the shard headers;
- **F**, the bytes the reader asked the kernel for, as the reader
  reports it (`F declared`) and, with intent tracing, as the kernel saw
  it: read-family calls and memory mappings on the target file, demand
  faults and page-cache fills (`F observed`);
- **D**, the bytes the device moved for the target file's sectors inside
  the trial window, with the command count, the command size
  histogram and completion latency percentiles.

Everything is attributed by file extents and time window, so readahead,
neighbouring tensors and other tenants of the device are counted
separately and never hidden. The method and its limits are in the
docstrings of `traced_replay.py` and `replay_ranges.py`.

## Running it

The Kconfig workflow runs the whole thing from a defconfig; bulk data
(checkpoints, traces, run directories) lives under
`CONFIG_KNLP_WEIGHT_IO_ROOT`, outside Git, on the NVMe you want to
characterize.

```
make defconfig-weight-io-inventory && make   # shard headers only, no device
make defconfig-weight-io-controls  && make   # three controls replayed on the NVMe
make defconfig-weight-io-full      && make   # plus compact ranges and the GPU stage
```

The stages are `weight-io-doctor` (checks the toolchain, the device and
the interpreters), `weight-io-inventory` (pins, shard headers, byte
layout, unit selection, page-footprint projections), `weight-io-fetch`
(control checkpoints in full, verified against the hub's LFS hashes;
optionally the selected ranges of the large checkpoints),
`weight-io-tracers` (clones and builds `nvme_tp_monitor`,
`syscall_monitor` and `mmap_readamp` from the
[ebpf-syscall](https://github.com/SamsungDS/ebpf-syscall) repository),
`weight-io-matrix` (the idle-exit probe and the replay matrix),
`weight-io-gpu` (native load and decode, one BF16 and one AWQ layer
streamed through the GPU) and `weight-io-report` (metrics, tables,
projections, per-run manifests). Each is a make target and each is
resumable. The tracers attach BPF programs, so the replay stages run
them with `sudo`; the readers themselves run as you.

The replay groups: `layers` reads the first, middle and last decoder
layer of every control under `pread`, `mmap` and `O_DIRECT`, cold and
warm; `rows` reads single embedding rows of the smallest control;
`chunks` re-reads one dense layer as contiguous `O_DIRECT` chunks of
4 KiB to 1 MiB at queue depths 1 to 32; `awq` reads clustered and
scattered neuron slices of the INT4 checkpoint through a naive reader,
a per-range direct reader and a page-merging direct reader. Which
checkpoint plays which role is derived from the inventory: the
smallest embedding row, the largest unquantized layer and the
checkpoint whose config names `awq`.

## Intent tracing

With `CONFIG_KNLP_WEIGHT_IO_INTENT=y` every reader is started stopped,
`syscall_monitor` and `mmap_readamp` attach to its process id, and it
is resumed once both report attachment. The run's `intent.jsonl` is the
reader's file calls (schema `syscall-capture-v2`, one joined record per
call), `faults.jsonl` the per-file fault and page-cache tables, and
`summary.json` carries, per trial window, the read calls and bytes and
the mappings on the target file, and per run the demand-fault and
page-cache-fill bytes. For a reader these scripts wrote, observed F
equals declared F; for a third-party loader, observed F is the only F
there is. A capture can be replayed against a copy of the checkpoint
with `syscall_replayer` from the same tracer tree.

## What to publish

A reproducibility claim needs the device's geometry, not its identity:
`stageA/device.json` records the logical and physical block size, the
maximum command size, the segment limit, the readahead window and the
partition start. Nothing in these scripts reads a serial number or a
model string, and nothing should be added that does.
