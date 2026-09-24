# SPDX-License-Identifier: MIT
"""Refuse to run the P/D ladder on a host that cannot produce a valid result.

Every check here is one line of output and was, at some point, an hour lost.
The expensive ones are not the obvious "is it installed" questions but the
ones where something is present and still wrong: a compiled extension that
exists but will not import, a driver that reports success while handing the
NVMe addresses nothing mapped, a device that advertises a dma-buf ceiling and
then refuses the registration.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys

OK, WARN, FAIL = "ok", "warn", "FAIL"


class Report:
    def __init__(self) -> None:
        self.rows: list[tuple[str, str, str]] = []

    def add(self, name: str, state: str, detail: str = "") -> None:
        self.rows.append((name, state, detail))

    def render(self) -> int:
        width = max(len(n) for n, _, _ in self.rows)
        for name, state, detail in self.rows:
            print(f"  {name:<{width}}  {state:<4}  {detail}")
        failed = [n for n, s, _ in self.rows if s == FAIL]
        if failed:
            print(f"\npd-doctor: {len(failed)} blocking problem(s): {', '.join(failed)}")
            return 1
        print("\npd-doctor: this host can produce a valid result")
        return 0


def _read(path: str) -> str | None:
    try:
        with open(path) as f:
            return f.read().strip()
    except OSError:
        return None


def check_kernel(r: Report) -> None:
    """The dma-buf series and the clamp are what make large commands possible."""
    found = None
    for entry in sorted(os.listdir("/sys/block")):
        if not entry.startswith("nvme"):
            continue
        if _read(f"/sys/block/{entry}/queue/max_hw_dmabuf_sectors_kb"):
            found = entry
            break
    r.add("kernel dma-buf ceiling", OK if found else FAIL,
          f"max_hw_dmabuf_sectors_kb on {found}" if found
          else "no namespace reports it: kernel lacks the dma-buf series")

    cmdline = _read("/proc/cmdline") or ""
    lifted = "nvme.lift_dma_opt_clamp=1" in cmdline
    r.add("nvme clamp lifted", OK if lifted else WARN,
          "nvme.lift_dma_opt_clamp=1" if lifted
          else "not on the command line: commands stay at the DMA-optimal clamp")


def check_driver(r: Report) -> None:
    """595 on a 7.3 kernel reports success for a mapping it never made."""
    ver = _read("/sys/module/nvidia/version")
    if not ver:
        r.add("nvidia driver", WARN, "not loaded (only the GPU rungs need it)")
        return
    try:
        major = int(ver.split(".")[0])
    except ValueError:
        major = 0
    r.add("nvidia driver", OK if major >= 610 else FAIL,
          f"{ver}" + ("" if major >= 610
                      else " -- needs >= 610.57.04; 595 hands the NVMe unmapped"
                           " bus addresses and every command dies in a DMAR fault"))


def check_extensions(r: Report) -> None:
    """Importable, not merely present: LMCache warns and falls back."""
    for mod, why in (
        ("lmcache.cuda_ops", "CUDA kernels; without it correctness claims are void"),
        ("lmcache.lmcache_native", "the native storage ops"),
        ("lmcache_rust_raw_block_io", "io_uring/O_DIRECT/dma-buf device layer"),
    ):
        code = f"import {mod}"
        p = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
        detail = why if p.returncode == 0 else p.stderr.strip().splitlines()[-1][:88]
        r.add(f"import {mod}", OK if p.returncode == 0 else FAIL, detail)


def check_device(r: Report, serial: str) -> None:
    """A wrong device here is data loss, so every step is checked."""
    if not serial:
        r.add("target namespace", WARN, "no serial configured: device rungs will skip")
        return
    link = None
    by_id = "/dev/disk/by-id"
    if os.path.isdir(by_id):
        for entry in os.listdir(by_id):
            if entry.endswith(serial) and "-part" not in entry:
                link = os.path.join(by_id, entry)
                break
    if not link:
        r.add("target namespace", FAIL, f"no /dev/disk/by-id entry ends with {serial}")
        return
    real = os.path.realpath(link)
    name = os.path.basename(real)
    back = _read(f"/sys/block/{name}/device/serial")
    if (back or "").strip() != serial:
        r.add("target namespace", FAIL,
              f"{real} reads back serial {back!r}, not {serial!r}")
        return
    r.add("target namespace", OK, f"{serial} -> {real}, serial verified")

    # A multipath head or a dm target inherits the dma-buf ceiling through
    # blk_stack_limits and then refuses the registration.
    dev_link = os.path.realpath(f"/sys/block/{name}/device")
    head = "nvme-subsys" in dev_link
    r.add("not a multipath head", FAIL if head else OK,
          "this namespace is a subsystem head: registration returns EOPNOTSUPP"
          if head else dev_link.rsplit("/", 1)[-1])

    mounts = _read("/proc/mounts") or ""
    mounted = any(line.split()[0].startswith(real) for line in mounts.splitlines())
    r.add("target unmounted", FAIL if mounted else OK,
          "something is mounted on it" if mounted else "nothing mounted")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--serial", default=os.environ.get("KNLP_PD_DEVICE_SERIAL", ""))
    ap.add_argument("--rungs", default=os.environ.get("KNLP_PD_RUNGS", "smoke"))
    args = ap.parse_args(argv)

    print(f"pd-doctor: rungs={args.rungs}")
    r = Report()
    check_kernel(r)
    check_extensions(r)
    if args.rungs != "smoke":
        check_driver(r)
        check_device(r, args.serial)
    else:
        r.add("device checks", OK, "skipped: smoke runs on CPU and files only")
    return r.render()


if __name__ == "__main__":
    raise SystemExit(main())
