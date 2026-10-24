# SPDX-License-Identifier: MIT
"""Observe exact cold-source admission and the registered storage data path.

The experiment uses a fresh regular file and the existing raw-block adapter.
L2 lookup proves current visibility, not power-loss durability. Evicting every
source before a child is a controlled cold read, not a cache-policy result.
"""

from __future__ import annotations

import os
from pathlib import Path
import platform
import resource
import subprocess
import time


def validate_settings(settings: dict) -> None:
    """Bind storage object bytes to the frozen model's actual KV geometry."""
    geometry = {
        "chunk_tokens": 256,
        "model_layers": 24,
        "logical_kv_planes": 2,
        "kv_heads": 2,
        "head_size": 64,
        "dtype_bytes": 2,
        "source_objects": 4,
        "children": 4,
    }
    if any(
        type(settings.get(name)) is not int or settings[name] != value
        for name, value in geometry.items()
    ):
        raise ValueError("storage profile differs from the frozen model geometry")
    if settings.get("prefetch_policy") != "retain":
        raise ValueError("both storage arms require explicit external L1 retention")
    payload = 1
    for name in (
        "chunk_tokens",
        "model_layers",
        "logical_kv_planes",
        "kv_heads",
        "head_size",
        "dtype_bytes",
    ):
        payload *= settings[name]
    if (
        settings["chunk_payload_bytes"] != payload
        or type(settings["arena_bytes"]) is not int
        or not payload * settings["source_objects"]
        <= settings["arena_bytes"]
        <= 1 << 30
        or settings["arena_bytes"] % 4096
        # Two checkpoint banks, each with a header and at least one data block.
        or settings["metadata_bytes"] < 2 * 2 * 4096
        or settings["metadata_bytes"] % 4096
        or settings["max_transfer_bytes"] <= 0
        or settings["max_transfer_bytes"] % 4096
    ):
        raise ValueError(
            "storage capacity or alignment differs from its canonical objects"
        )


def require_host_memlock(settings: dict) -> None:
    """Refuse an impossible host registration before starting the model pair."""
    soft, _hard = resource.getrlimit(resource.RLIMIT_MEMLOCK)
    if soft != resource.RLIM_INFINITY and soft < settings["arena_bytes"]:
        raise ValueError(
            f"RLIMIT_MEMLOCK soft limit {soft} bytes cannot register the "
            f"{settings['arena_bytes']}-byte host arena; raise the process's "
            "locked-memory limit, allowing additional worker and metadata registrations"
        )


def describe_location(path: Path) -> dict:
    """Record the actual filesystem and NVMe namespace used by both arms."""
    probe = path if path.exists() else path.parent
    st = probe.stat()
    devt = f"{os.major(st.st_dev)}:{os.minor(st.st_dev)}"
    device = Path("/sys/dev/block", devt).resolve(strict=True)
    fs = subprocess.check_output(
        ["findmnt", "-n", "-o", "FSTYPE", "-T", str(probe)], text=True
    ).strip()
    if fs not in ("xfs", "ext4") or not device.name.startswith("nvme"):
        raise ValueError("storage comparison requires XFS or ext4 on an NVMe namespace")
    return {
        "filesystem": fs,
        "device": device.name,
        "device_number": devt,
        "kernel": platform.release(),
        "memlock_bytes": dict(
            zip(("soft", "hard"), resource.getrlimit(resource.RLIMIT_MEMLOCK))
        ),
    }


def initialize_backing_file(path: Path, file_bytes: int) -> None:
    """Create initialized private extents; preallocation alone stays unwritten."""
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    try:
        os.posix_fallocate(fd, 0, file_bytes)
        zeros = bytes(min(file_bytes, 1 << 20))
        remaining = file_bytes
        while remaining:
            chunk = memoryview(zeros)[: min(remaining, len(zeros))]
            while chunk:
                written = os.write(fd, chunk)
                if written <= 0:
                    raise OSError("backing-file initialization made no progress")
                chunk = chunk[written:]
                remaining -= written
        os.fsync(fd)
    finally:
        os.close(fd)


def create_config(route: str, path: Path, settings: dict):
    """Create one owned file and configure matched registered I/O arenas."""
    from lmcache.v1.distributed.config import (
        DRAML1ManagerConfig,
        EvictionConfig,
        GPUL1ManagerConfig,
        L1MemoryManagerConfig,
        StorageManagerConfig,
    )
    from lmcache.v1.distributed.l2_adapters.config import L2AdaptersConfig
    from lmcache.v1.distributed.l2_adapters.raw_block_l2_adapter import (
        RawBlockL2AdapterConfig,
    )

    if route not in ("staged", "dma-buf"):
        raise ValueError("select staged or dma-buf storage")
    validate_settings(settings)
    if route == "staged":
        require_host_memlock(settings)
    describe_location(path)
    block_align = 4096
    payload = settings["chunk_payload_bytes"]
    if payload <= 0 or payload % block_align:
        raise ValueError("canonical payload must contain complete storage blocks")
    slot = payload + block_align
    # One immutable source set plus an equally sized reserve. No payload is
    # overwritten or reclaimed during this five-request replay.
    file_bytes = settings["metadata_bytes"] + 2 * settings["source_objects"] * slot
    memory = L1MemoryManagerConfig(
        settings["arena_bytes"],
        False,
        shm_name="",
        buffer_registration="fixed" if route == "staged" else "dmabuf",
    )
    l1 = (
        DRAML1ManagerConfig(memory)
        if route == "staged"
        else GPUL1ManagerConfig(memory, device="cuda:0", buffer_provider="vulkan_rm")
    )
    adapter = RawBlockL2AdapterConfig(
        device_path=str(path),
        slot_bytes=slot,
        capacity_bytes=file_bytes,
        block_align=block_align,
        header_bytes=block_align,
        meta_total_bytes=settings["metadata_bytes"],
        meta_enable_periodic=False,
        load_checkpoint_on_init=False,
        use_odirect=True,
        enable_zero_copy=True,
        io_engine="io_uring",
        iouring_queue_depth=settings["queue_depth"],
        max_data_transfer_size=settings["max_transfer_bytes"],
        num_store_workers=1,
        num_lookup_workers=1,
        num_load_workers=1,
    )
    config = StorageManagerConfig(
        l1,
        EvictionConfig("LRU"),
        L2AdaptersConfig([adapter]),
        prefetch_policy=settings["prefetch_policy"],
    )
    initialize_backing_file(path, file_bytes)
    return config


def await_exact_l2_keys(adapter, keys, layouts, deadline: float) -> None:
    """Consume only our lookup receipts and unlock every acquired key.

    A missing receipt never permits L1 eviction. Unknown lookup ownership
    belongs to the caller's still-running service until actual process exit.
    """
    while time.monotonic() < deadline:
        task = adapter.submit_lookup_and_lock_task(keys, layouts)
        result = None
        while result is None and time.monotonic() < deadline:
            result = adapter.query_lookup_and_lock_result(task)
            if result is None:
                time.sleep(0.002)
        if result is None:
            raise RuntimeError("source L2 lookup completion is unknown")
        found = [key for i, key in enumerate(keys) if result.test(i)]
        if found:
            adapter.submit_unlock(found)
        if len(found) == len(keys):
            return
        time.sleep(0.002)
    raise RuntimeError("exact source keys did not enter L2 before the deadline")


def evict_settled_l1_keys(manager, keys, deadline: float) -> int:
    """Wait for actual store leases to release without force-deleting a source."""
    pending = list(keys)
    deleted_count = 0
    while pending and time.monotonic() < deadline:
        locked = []
        for key in pending:
            deleted, skipped = manager.delete_l1_keys([key])
            if (deleted, skipped) == (1, 0):
                deleted_count += 1
            elif (deleted, skipped) == (0, 1):
                locked.append(key)
            else:
                raise RuntimeError(
                    "retained source L1 key disappeared or changed owners"
                )
        pending = locked
        if pending:
            time.sleep(0.002)
    if pending:
        raise RuntimeError("source L1 copies still have outstanding physical leases")
    return deleted_count
