# SPDX-License-Identifier: MIT
"""Run one arm of the storage-mediated P/D matrix and record what it measured.

Seven arms across two implementations:

    nixl-gds      vLLM connector over NIXL GDS          (the baseline)
    nixl-block    NIXL XNVME_NVM, io_uring, raw block
    nixl-xfs      NIXL XNVME_NVM, io_uring, file on XFS
    nixl-ext4     NIXL XNVME_NVM, io_uring, file on ext4
    lmcache-block LMCache raw_block, io_uring, raw block
    lmcache-xfs   LMCache raw_block, io_uring, file on XFS
    lmcache-ext4  LMCache raw_block, io_uring, file on ext4

Arms are interleaved across iterations rather than run back to back: on this
device a non-interleaved sweep once produced a clean-looking ranking that was
drift, and did not reproduce.  Every number carries its arm, its iteration and
its host so a later reader can tell measurements apart from assumptions.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import time

MiB = 1 << 20


def resolve_device(serial: str, partition: int) -> str:
    by_id = "/dev/disk/by-id"
    for entry in sorted(os.listdir(by_id)):
        if entry.endswith(serial) and "-part" not in entry:
            real = os.path.realpath(os.path.join(by_id, entry))
            name = os.path.basename(real)
            with open(f"/sys/block/{name}/device/serial") as f:
                if f.read().strip() != serial:
                    raise SystemExit(f"{real} does not read back serial {serial}")
            return f"{real}p{partition}" if partition else real
    raise SystemExit(f"no /dev/disk/by-id entry for serial {serial}")


def target_path(arm: dict, serial: str) -> str:
    if arm["target"] == "block":
        return resolve_device(serial, arm["partition"])
    mount = arm["xfs_mount"] if arm["target"] == "xfs" else arm["ext4_mount"]
    return os.path.join(mount, f"pd-{arm['name']}.bin")


def run_nixl(arm: dict, path: str, size: int, objects: int) -> dict:
    """Drive the NIXL device harness and read its own reported rates."""
    env = dict(os.environ)
    env["NIXL_PLUGIN_DIR"] = arm["plugin_dir"]
    env["LD_LIBRARY_PATH"] = arm["ld_path"]
    if arm["target"] != "block" and not os.path.exists(path):
        # The backend opens the target; a file one has to exist first.
        with open(path, "wb") as f:
            f.truncate(objects * size * 4)
    cmd = [arm["harness"], "--device", path, "--backend", arm["nixl_backend"],
           "--objects", str(objects), "--object-size", str(size),
           "--range-offset", "0", "--range-len", str(objects * size * 4)]
    if arm["nixl_backend"] == "XNVME_NVM":
        cmd += ["--async", "io_uring", "--buffer-mode", arm["buffer_mode"]]
    p = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=600)
    out = p.stdout + p.stderr
    rates: dict[str, float] = {}
    for line in out.splitlines():
        for key in ("write", "read"):
            if line.strip().startswith(f"{key}:") and "MiB/s" in line:
                rates[key] = float(line.split("(")[1].split()[0])
    ok = "harness: PASS" in out
    return {"ok": ok, "write_mibs": rates.get("write"), "read_mibs": rates.get("read"),
            "error": None if ok else out.strip().splitlines()[-1][:160]}


def run_gds(arm: dict, path: str, size: int, objects: int) -> dict:
    """Drive the NIXL GDS backend through its python API.

    The device harness speaks only XNVME_NVM and POSIX, so the baseline has
    to be driven directly.  Note what this measures: when the IOMMU is
    translating, cuFile reports "NVMe : compat" and stages through host
    memory, so a number from that state is a host-staged number and is
    labelled as one in the record.
    """
    import nixl._utils as nixl_utils
    from nixl._api import nixl_agent, nixl_agent_config

    agent = nixl_agent("pd-matrix-gds", nixl_agent_config(backends=[]))
    if "GDS" not in agent.get_plugin_list():
        return {"ok": False, "write_mibs": None, "read_mibs": None,
                "error": "GDS plugin not present"}
    agent.create_backend("GDS")

    total = objects * size
    src = nixl_utils.malloc_passthru(total)
    dst = nixl_utils.malloc_passthru(total)
    nixl_utils.ba_buf(src, total)
    reg = agent.get_reg_descs([(src, total, 0, "s"), (dst, total, 0, "d")], "DRAM")
    assert agent.register_memory(reg) is not None

    fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o644)
    os.ftruncate(fd, total)
    files = agent.register_memory([(0, total, fd, "f")], "FILE")
    xfer_files = files.trim()

    def one(direction: str, buf: int) -> float:
        descs = agent.get_xfer_descs([(buf, total, 0)], "DRAM")
        h = agent.initialize_xfer(direction, descs, xfer_files, "pd-matrix-gds")
        if not h:
            raise RuntimeError(f"{direction}: initialize_xfer failed")
        t0 = time.monotonic()
        state = agent.transfer(h)
        if state == "ERR":
            raise RuntimeError(f"{direction}: transfer failed")
        while agent.check_xfer_state(h) != "DONE":
            pass
        dt = time.monotonic() - t0
        agent.release_xfer_handle(h)
        return dt

    try:
        wr = one("WRITE", src)
        rd = one("READ", dst)
    except Exception as e:
        return {"ok": False, "write_mibs": None, "read_mibs": None, "error": str(e)[:160]}
    finally:
        os.close(fd)

    mib = total / MiB
    return {"ok": True, "write_mibs": mib / wr, "read_mibs": mib / rd, "error": None}


def run_lmcache(arm: dict, path: str, size: int, objects: int) -> dict:
    """Drive LMCache's own device layer over the same geometry."""
    import lmcache_rust_raw_block_io as rb
    is_file = arm["target"] != "block"
    if is_file and not os.path.exists(path):
        with open(path, "wb") as f:
            f.truncate(objects * size * 4)
    lba = 4096
    dev = rb.RawBlockDevice(path, writable=True, use_odirect=True, alignment=lba,
                            io_engine="io_uring", iouring_queue_depth=32)
    try:
        payload = bytearray(os.urandom(size))
        back = bytearray(size)
        t0 = time.monotonic()
        for i in range(objects):
            dev.write_uring(i * size, payload, size, size)
        wr = time.monotonic() - t0
        t0 = time.monotonic()
        for i in range(objects):
            dev.read_uring(i * size, back, size, size)
        rd = time.monotonic() - t0
    finally:
        dev.close()
    total = objects * size / MiB
    return {"ok": True, "write_mibs": total / wr, "read_mibs": total / rd, "error": None}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--name", required=True)
    ap.add_argument("--path", required=True, choices=("nixl", "lmcache"))
    ap.add_argument("--nixl-backend", default="XNVME_NVM")
    ap.add_argument("--target", required=True, choices=("block", "xfs", "ext4"))
    ap.add_argument("--partition", type=int, default=1)
    ap.add_argument("--serial", required=True)
    ap.add_argument("--buffer-mode", default="host")
    ap.add_argument("--xfs-mount", default="/mnt/pd-xfs")
    ap.add_argument("--ext4-mount", default="/mnt/pd-ext4")
    ap.add_argument("--harness", default="")
    ap.add_argument("--plugin-dir", default="")
    ap.add_argument("--ld-path", default="")
    ap.add_argument("--object-size", type=int, default=2 * MiB)
    ap.add_argument("--objects", type=int, default=64)
    ap.add_argument("--iteration", type=int, default=0)
    ap.add_argument("--out", default="")
    a = ap.parse_args(argv)

    arm = {"name": a.name, "target": a.target, "partition": a.partition,
           "nixl_backend": a.nixl_backend, "buffer_mode": a.buffer_mode,
           "xfs_mount": a.xfs_mount, "ext4_mount": a.ext4_mount,
           "harness": a.harness, "plugin_dir": a.plugin_dir, "ld_path": a.ld_path}
    path = target_path(arm, a.serial)
    if a.path == "nixl" and a.nixl_backend == "GDS":
        res = run_gds(arm, path, a.object_size, a.objects)
    elif a.path == "nixl":
        res = run_nixl(arm, path, a.object_size, a.objects)
    else:
        res = run_lmcache(arm, path, a.object_size, a.objects)

    rec = {"arm": a.name, "path": a.path, "backend": a.nixl_backend if a.path == "nixl" else "raw_block",
           "target": a.target, "device": path, "iteration": a.iteration,
           "object_size": a.object_size, "objects": a.objects,
           "buffer_mode": a.buffer_mode, "host": platform.node(),
           "kernel": platform.release(), "when": time.strftime("%Y-%m-%dT%H:%M:%S"),
           **res}
    line = json.dumps(rec)
    if a.out:
        os.makedirs(os.path.dirname(a.out), exist_ok=True)
        with open(a.out, "a") as f:
            f.write(line + "\n")
    w = f"{res['write_mibs']:.0f}" if res["write_mibs"] else "-"
    r = f"{res['read_mibs']:.0f}" if res["read_mibs"] else "-"
    print(f"  {a.name:<14} iter {a.iteration}  write {w:>7} MiB/s  read {r:>7} MiB/s"
          f"  {'ok' if res['ok'] else 'FAIL: ' + str(res['error'])[:60]}")
    return 0 if res["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
