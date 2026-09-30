#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0
"""Qualify whether cache reads are observably served by a block device.

This CPU-only harness writes real, aligned bytes and records both per-process
``/proc/self/io`` deltas and backing-device sector deltas.  It deliberately
separates first validation, warm reuse, post-validation cold reuse, and an
O_DIRECT reuse pass.  It does not infer NVMe service from O_DIRECT alone.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import mmap
import os
from pathlib import Path
import subprocess
import time

BLOCK = 4096
CHUNK = 4 * 1024 * 1024


def command(*argv):
    normalized = [os.fspath(value) for value in argv]
    try:
        result = subprocess.run(
            normalized, capture_output=True, text=True, timeout=30
        )
    except (OSError, subprocess.SubprocessError) as error:
        return {
            "argv": normalized,
            "returncode": None,
            "stdout": "",
            "stderr": f"unavailable: {error}",
        }
    return {
        "argv": normalized,
        "returncode": result.returncode,
        "stdout": result.stdout.strip(),
        "stderr": result.stderr.strip(),
    }


def process_io():
    values = {}
    with open("/proc/self/io") as source:
        for line in source:
            key, value = line.split(":", 1)
            values[key] = int(value)
    return values


def block_stats(device):
    fields = Path(f"/sys/class/block/{device}/stat").read_text().split()
    return {
        "reads_completed": int(fields[0]),
        "sectors_read": int(fields[2]),
        "read_bytes": int(fields[2]) * 512,
        "writes_completed": int(fields[4]),
        "sectors_written": int(fields[6]),
        "written_bytes": int(fields[6]) * 512,
    }


def delta(after, before):
    return {key: after[key] - before[key] for key in before}


def backing_device(root):
    mount = command("findmnt", "-n", "-o", "SOURCE", "-T", os.fspath(root))
    source = mount["stdout"].splitlines()[0]
    parent = command("lsblk", "-ndo", "PKNAME", source)["stdout"]
    return (parent or os.path.basename(source)).strip(), source


def aligned_mapping():
    mapping = mmap.mmap(-1, CHUNK)
    view = memoryview(mapping)
    view[:] = bytes(range(256)) * (CHUNK // 256)
    return mapping, view


def prepare(path, size):
    if size <= 0 or size % BLOCK:
        raise ValueError("each file size must be positive and 4096-byte aligned")
    mapping, view = aligned_mapping()
    digest = hashlib.sha256()
    flags = os.O_CREAT | os.O_TRUNC | os.O_RDWR | os.O_DIRECT
    fd = os.open(path, flags, 0o600)
    started = time.perf_counter()
    try:
        offset = 0
        while offset < size:
            length = min(CHUNK, size - offset)
            written = os.pwritev(fd, [view[:length]], offset)
            if written != length:
                raise OSError(f"short direct write: {written} of {length}")
            digest.update(view[:length])
            offset += written
        os.fsync(fd)
    finally:
        os.close(fd)
        view.release()
        mapping.close()
    status = os.stat(path)
    return {
        "logical_bytes": status.st_size,
        "allocated_bytes": status.st_blocks * 512,
        "sha256": digest.hexdigest(),
        "seconds": time.perf_counter() - started,
        "write_mode": "O_DIRECT_pwritev_fsync",
    }


def evict(path):
    fd = os.open(path, os.O_RDONLY)
    try:
        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
    finally:
        os.close(fd)


def read_pass(path, *, direct, verify, device, expected_sha256, label):
    mapping = mmap.mmap(-1, CHUNK)
    view = memoryview(mapping)
    flags = os.O_RDONLY | (os.O_DIRECT if direct else 0)
    fd = os.open(path, flags)
    process_before = process_io()
    device_before = block_stats(device)
    started = time.perf_counter()
    digest = hashlib.sha256() if verify else None
    requested = 0
    try:
        size = os.fstat(fd).st_size
        while requested < size:
            length = min(CHUNK, size - requested)
            if direct and length % BLOCK:
                raise ValueError("direct read tail is not block aligned")
            got = os.preadv(fd, [view[:length]], requested)
            if got != length:
                raise OSError(f"short read: {got} of {length}")
            if digest is not None:
                digest.update(view[:got])
            requested += got
    finally:
        os.close(fd)
    elapsed = time.perf_counter() - started
    device_after = block_stats(device)
    process_after = process_io()
    actual_sha256 = digest.hexdigest() if digest is not None else None
    view.release()
    mapping.close()
    return {
        "label": label,
        "direct": direct,
        "integrity": "sha256" if verify else "prevalidated_identity_contract",
        "requested_bytes": requested,
        "seconds": elapsed,
        "gib_per_second": requested / elapsed / 2**30,
        "sha256": actual_sha256,
        "sha256_matches": (
            actual_sha256 == expected_sha256 if actual_sha256 is not None else None
        ),
        "process_delta": delta(process_after, process_before),
        "backing_device_delta": delta(device_after, device_before),
    }


def parse_file(value):
    name, size = value.split("=", 1)
    return name, int(size)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--file", action="append", type=parse_file, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.root.mkdir(parents=True, exist_ok=True)
    device, mount_source = backing_device(args.root)
    receipt = {
        "schema": "kv_storage_service_qualification_v1",
        "root": str(args.root.resolve()),
        "mount_source": mount_source,
        "backing_device": device,
        "system": {
            "findmnt": command(
                "findmnt", "-n", "-o", "SOURCE,FSTYPE,OPTIONS", "-T", args.root
            ),
            "lsblk": command(
                "lsblk", "-o", "NAME,PKNAME,TYPE,SIZE,ROTA,TRAN,MOUNTPOINTS"
            ),
        },
        "files": [],
    }
    for name, size in args.file:
        path = args.root / name
        prepared = prepare(path, size)
        os.sync()
        evict(path)
        passes = [
            read_pass(
                path,
                direct=False,
                verify=True,
                device=device,
                expected_sha256=prepared["sha256"],
                label="first_validation_cold",
            ),
            read_pass(
                path,
                direct=False,
                verify=False,
                device=device,
                expected_sha256=prepared["sha256"],
                label="warm_prevalidated_reuse",
            ),
        ]
        evict(path)
        passes.append(
            read_pass(
                path,
                direct=False,
                verify=False,
                device=device,
                expected_sha256=prepared["sha256"],
                label="post_validation_cold_reuse",
            )
        )
        passes.append(
            read_pass(
                path,
                direct=True,
                verify=False,
                device=device,
                expected_sha256=prepared["sha256"],
                label="direct_prevalidated_reuse",
            )
        )
        receipt["files"].append(
            {
                "name": name,
                "path": str(path),
                "preparation": prepared,
                "alignment_bytes": BLOCK,
                "filefrag": command("filefrag", "-v", path),
                "passes": passes,
            }
        )
    args.output.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
