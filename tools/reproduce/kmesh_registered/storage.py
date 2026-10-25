# SPDX-License-Identifier: MIT
"""Observe exact cold-source admission and the registered storage data path.

The experiment uses a fresh regular file and the existing raw-block adapter.
L2 lookup proves current visibility, not power-loss durability. Evicting every
source before a child is a controlled cold read, not a cache-policy result.
Qualification additionally poisons the freed L1 arena before loading it again;
timing runs omit this conditioning after both real model controls pass.
"""

from __future__ import annotations

from functools import wraps
import os
from pathlib import Path
import platform
import resource
import subprocess
import threading
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


class StorageObserver:
    """Own bounded observation hooks in the importing experiment process."""

    def __init__(
        self,
        manager,
        settings: dict,
        *,
        poison_arena: bool = False,
        checksum_library=None,
    ):
        adapters = manager.l2_adapters()
        if len(adapters) != 1:
            raise ValueError("cold comparison requires one actual L2 adapter")
        self.manager, self.adapter = manager, adapters[0][1]
        self.settings = settings
        self.location = describe_location(
            Path(self.adapter.report_status()["core"]["device_path"])
        )
        self.lock = threading.Lock()
        self.reserved = {}
        self.admitted = {}
        self.phase = "source"
        self.rows = []
        self.originals = []
        self.poison_arena = poison_arena
        self.conditioning = False
        self.conditioning_work = None
        self.gpu_checksum = None
        if checksum_library is not None:
            from gpu_integrity import CudaSha256

            self.gpu_checksum = CudaSha256(checksum_library)

    def install(self) -> None:
        """Install process-local admission, I/O and timing observations."""
        from lmcache.v1.distributed.error import L1Error
        from lmcache.v1.distributed.storage_manager import StorageManager
        from lmcache.v1.multiprocess import kmesh_span_store as spans
        from lmcache.v1.storage_backend.raw_block import RawBlockCore, decode_object_key

        reserve = StorageManager.reserve_write
        admit = StorageManager.finish_write_with_status
        checksum = spans.checksum_span_object
        restore = spans.KMeshSpanStore.restore
        load = RawBlockCore.load_many_into

        @wraps(reserve)
        def reserve_source(manager, keys, layout_desc):
            if manager is self.manager and not self.conditioning:
                size = sum(
                    shape.numel() * dtype.itemsize
                    for shape, dtype in zip(
                        layout_desc.shapes, layout_desc.dtypes, strict=True
                    )
                )
                if size != self.settings["chunk_payload_bytes"]:
                    raise RuntimeError(
                        "actual source layout differs from the storage profile"
                    )
            result = reserve(manager, keys, layout_desc)
            if manager is self.manager and not self.conditioning:
                with self.lock:
                    for key in result:
                        self.reserved[key] = layout_desc
            return result

        @wraps(load)
        def load_source(core, encoded_keys, objs, *, io_context=None):
            if io_context is None:
                raise RuntimeError(
                    "storage timing requires actual native I/O attribution"
                )
            start = time.perf_counter_ns()
            result = load(core, encoded_keys, objs, io_context=io_context)
            elapsed = time.perf_counter_ns() - start
            self.rows.append(
                {
                    "kind": "storage_read",
                    "phase": self.phase,
                    "elapsed_ns": elapsed,
                    "io_request_id": io_context.request_id,
                    "source_key_hashes": [
                        decode_object_key(key).chunk_hash.hex() for key in encoded_keys
                    ],
                    "payload_bytes": sum(obj.get_size() for obj in objs),
                    "success": len(result) == len(encoded_keys) and all(result),
                }
            )
            return result

        @wraps(admit)
        def admit_source(manager, completion):
            result = admit(manager, completion)
            if manager is self.manager:
                with self.lock:
                    for key, outcome in result.items():
                        if outcome == L1Error.SUCCESS:
                            self.admitted[key] = self.reserved[key]
            return result

        @wraps(checksum)
        def hash_source(obj):
            start = time.perf_counter_ns()
            tensor = obj.get_tensor(0)
            use_gpu = self.gpu_checksum is not None and self.phase != "source"
            if use_gpu:
                result = self.gpu_checksum.hexdigest(tensor)
            else:
                result = checksum(obj)
                if self.gpu_checksum is not None:
                    if self.gpu_checksum.hexdigest(tensor) != result:
                        raise RuntimeError(
                            "GPU hash differs from the CPU source digest"
                        )
            self.rows.append(
                {
                    "kind": "checksum",
                    "phase": self.phase,
                    "elapsed_ns": time.perf_counter_ns() - start,
                    "payload_bytes": obj.get_size(),
                    "host_payload_copy_bytes": (
                        obj.get_size() if tensor.is_cuda and not use_gpu else 0
                    ),
                    "host_digest_copy_bytes": 32 if use_gpu else 0,
                    "checksum_backend": "cuda-sha256" if use_gpu else "cpu",
                }
            )
            return result

        @wraps(restore)
        def restore_source(
            executor, fence, request_id, generation, span, *args, **kwargs
        ):
            start = time.perf_counter_ns()
            result = restore(
                executor, fence, request_id, generation, span, *args, **kwargs
            )
            self.rows.append(
                {
                    "kind": "restore",
                    "phase": self.phase,
                    "elapsed_ns": time.perf_counter_ns() - start,
                    "request_id": request_id,
                    "span_id": span.span_id,
                    "state": result.state,
                }
            )
            return result

        self.originals = [
            (StorageManager, "reserve_write", reserve),
            (StorageManager, "finish_write_with_status", admit),
            (spans, "checksum_span_object", checksum),
            (spans.KMeshSpanStore, "restore", restore),
            (RawBlockCore, "load_many_into", load),
        ]
        StorageManager.reserve_write = reserve_source
        StorageManager.finish_write_with_status = admit_source
        spans.checksum_span_object = hash_source
        spans.KMeshSpanStore.restore = restore_source
        RawBlockCore.load_many_into = load_source

    def cold(self, request_id: str) -> dict:
        """Verify actual L2 admission, release lookup locks, then evict L1."""
        with self.lock:
            keys = list(self.admitted)
            layouts = {key.object_group_id: self.admitted[key] for key in keys}
        if len(keys) != self.settings["source_objects"] or set(layouts) != {0}:
            raise RuntimeError("cold comparison lacks its exact canonical source set")
        start = time.perf_counter_ns()
        deadline = time.monotonic() + 60
        await_exact_l2_keys(self.adapter, keys, layouts, deadline)
        deleted = evict_settled_l1_keys(self.manager, keys, deadline)
        poison = self._poison_free_arena() if self.poison_arena else None
        self.phase = request_id
        result = {
            "request_id": request_id,
            "source_key_hashes": [key.chunk_hash.hex() for key in keys],
            "confirmed_l2_keys": len(keys),
            "deleted_l1_keys": deleted,
            "poison": poison,
            "elapsed_ns": time.perf_counter_ns() - start,
            "before": self.adapter.report_status(),
        }
        self.rows.append({"kind": "cold", **result})
        return result

    def measure(self, request_id: str) -> dict:
        """Record post-model storage completion for the actual request."""
        row = {
            "kind": "after",
            "request_id": request_id,
            "adapter": self.adapter.report_status(),
        }
        self.rows.append(row)
        return row

    def snapshot(self) -> dict:
        """Capture observed operations and actual storage ownership."""
        return {
            "rows": list(self.rows),
            "adapter": self.adapter.report_status(),
            "storage_manager": self.manager.report_status(),
            "location": self.location,
            "poisoned_l1_arena": self.poison_arena,
            "checksum_backend": "cuda-sha256" if self.gpu_checksum else "cpu",
        }

    def close(self) -> None:
        """Restore the hooks saved by this process-local observer."""
        for owner, name, original in reversed(self.originals):
            setattr(owner, name, original)
        self.originals.clear()

    def _poison_free_arena(self) -> dict:
        """Condition an exclusive free arena, retaining any failed GPU work."""
        import hashlib
        import uuid

        import torch
        from lmcache import torch_dev
        from lmcache.v1.distributed.api import MemoryLayoutDesc, ObjectKey
        from lmcache.v1.platform.stream import synchronize_device

        if self.conditioning_work is not None:
            raise RuntimeError("earlier arena conditioning still owns physical work")
        key = ObjectKey(
            hashlib.sha256(uuid.uuid4().bytes).digest(), "arena-conditioning", 0
        )
        size = self.settings["arena_bytes"]
        layout = MemoryLayoutDesc([torch.Size((size // 2,))], [torch.bfloat16])
        work = {}
        self.conditioning_work = work
        self.conditioning = True
        try:
            work["objects"] = self.manager.reserve_write([key], layout)
            if set(work["objects"]) != {key}:
                raise RuntimeError(
                    "canonical arena is not entirely free for qualification"
                )
            work["completion"] = self.manager.prepare_write_completion(work["objects"])
            work["leases"] = self.manager.acquire_write_transfer_leases(
                work["objects"], all_buffers=True
            )
            buffer = work["objects"][key].get_tensor(0)
            if (
                buffer is None
                or buffer.nbytes != size
                or not buffer.is_contiguous()
                or buffer.device.type not in ("cpu", "cuda")
            ):
                raise RuntimeError(
                    "arena qualification requires one complete contiguous arena"
                )
            work["buffers"] = (buffer, buffer.view(torch.uint8))
            if buffer.device.type == "cuda":
                with torch_dev.device(buffer.device):
                    work["buffers"][1].fill_(255)
                    synchronize_device(buffer.device)
            else:
                work["buffers"][1].fill_(255)
            receipt = {
                "bytes": buffer.nbytes,
                "device": str(buffer.device),
                "drained": True,
            }
            self.manager.discard_write_by_owner(work["completion"])
            self.manager.release_transfer_leases(work["leases"])
            _, skipped = self.manager.delete_l1_keys([key])
            if skipped:
                raise RuntimeError(
                    "arena qualification reservation remains physically leased"
                )
            self.conditioning_work = None
            return receipt
        finally:
            self.conditioning = False
