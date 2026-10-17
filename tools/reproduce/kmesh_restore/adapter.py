# SPDX-License-Identifier: MIT
"""Measure real LMCache adapter loads through registered host or GPU arenas."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import select
from pathlib import Path
import time

from restore_common import (
    ARENA_BYTES,
    CAPACITY_BYTES,
    CHUNK_BYTES,
    CHUNKS,
    SLOT_BYTES,
    compact_core_pair,
    create_media,
    sha256,
    write_json,
)

# A failed constructor or unanswered native operation supplies no proof that
# its borrowed arena can be released. Keep that owner through process exit.
_RETAINED_RESOURCES: list[list[object]] = []


def wait_event(fd: int) -> None:
    """Wait for a real adapter completion notification with a bounded deadline."""
    poll = select.poll()
    poll.register(fd, select.POLLIN)
    if not poll.poll(60000):
        raise TimeoutError("LMCache adapter completion timed out")
    os.eventfd_read(fd)


def run(
    config: dict, mode: str, output: Path, samples: int, corrupt: bool = False
) -> None:
    """Restore sixteen model-sized chunks and fence their first GPU consumer."""
    import torch

    from lmcache.v1.distributed.api import MemoryLayoutDesc, ObjectKey
    from lmcache.v1.distributed.config import GPUL1ManagerConfig, L1MemoryManagerConfig
    from lmcache.v1.distributed.l2_adapters.raw_block_l2_adapter import (
        RawBlockL2Adapter,
        RawBlockL2AdapterConfig,
    )
    from lmcache.v1.distributed.memory_manager.gpu_l1_memory_manager import (
        GPUL1MemoryManager,
    )
    from lmcache.v1.distributed.memory_manager.l1_memory_manager import L1MemoryManager

    output.mkdir()
    media = output / "owned-storage.bin"
    create_media(media)
    memory_config = L1MemoryManagerConfig(
        ARENA_BYTES,
        False,
        shm_name="",
        buffer_registration="fixed" if mode == "staged" else "none",
    )
    manager = (
        GPUL1MemoryManager(
            GPUL1ManagerConfig(memory_config, buffer_provider=config["gpu_provider"])
        )
        if mode == "direct"
        else L1MemoryManager(memory_config)
    )
    keepalive: list[object] = [manager]
    _RETAINED_RESOURCES.append(keepalive)
    adapter = RawBlockL2Adapter(
        RawBlockL2AdapterConfig(
            device_path=str(media),
            slot_bytes=SLOT_BYTES,
            capacity_bytes=CAPACITY_BYTES,
            meta_total_bytes=1 << 20,
            meta_enable_periodic=False,
            load_checkpoint_on_init=False,
            io_engine="io_uring",
            use_odirect=True,
            iouring_queue_depth=8,
            num_store_workers=1,
            num_load_workers=1,
        ),
        manager.get_l1_memory_desc(),
    )
    keepalive.append(adapter)
    layout = MemoryLayoutDesc([torch.Size([CHUNK_BYTES])], [torch.uint8])
    _, objects = manager.allocate(layout, CHUNKS)
    keepalive.append(objects)
    if len(objects) != CHUNKS:
        raise MemoryError("could not allocate the full ancestor")
    keys = [
        ObjectKey(ObjectKey.IntHash2Bytes(i), "exact-prefix-fixture", 0)
        for i in range(CHUNKS)
    ]
    expected = []
    for i, obj in enumerate(objects):
        pattern = (
            (
                torch.arange(CHUNK_BYTES, device=obj.tensor.device, dtype=torch.int32)
                * 17
                + i * 29
            )
            % 251
        ).to(torch.uint8)
        obj.tensor.copy_(pattern)
        expected.append(obj.tensor.cpu().numpy().tobytes())
    torch.cuda.synchronize()
    store_id = adapter.submit_store_task(keys, objects)
    wait_event(adapter.get_store_event_fd())
    store = adapter.pop_completed_store_outcomes()[store_id]
    if not store.buffers_reusable or not store.result.is_successful():
        raise RuntimeError("initial ancestor store did not finish safely")
    if corrupt:
        # Flip one payload byte after the completed store. Slot metadata is
        # untouched, so only a complete consumer-byte check can reject it.
        fd = os.open(media, os.O_RDWR)
        try:
            offset = (1 << 20) + 4096
            byte = os.pread(fd, 1, offset)
            os.pwrite(fd, bytes([byte[0] ^ 0xFF]), offset)
            os.fsync(fd)
        finally:
            os.close(fd)
    task = adapter.submit_lookup_and_lock_task(keys, {0: layout})
    wait_event(adapter.get_lookup_and_lock_event_fd())
    if adapter.query_lookup_and_lock_result(task).get_indices_list() != list(
        range(CHUNKS)
    ):
        raise RuntimeError("ancestor lookup is incomplete")
    consumer = torch.empty(CHUNKS * CHUNK_BYTES, device="cuda", dtype=torch.uint8)
    records = []
    try:
        for sample in range(samples + 5):
            for obj in objects:
                obj.tensor.fill_(0xCD)
            consumer.fill_(0xEF)
            torch.cuda.synchronize()
            before = adapter.report_status()
            started = time.perf_counter_ns()
            task = adapter.submit_load_task(keys, objects)
            wait_event(adapter.get_load_event_fd())
            outcome = adapter.query_load_outcome(task)
            if outcome is None or not outcome.buffers_reusable:
                raise RuntimeError("restore did not return destination ownership")
            if outcome.bitmap.get_indices_list() != list(range(CHUNKS)):
                raise RuntimeError("restore has missing chunks")
            loaded = time.perf_counter_ns()
            for i, obj in enumerate(objects):
                consumer[i * CHUNK_BYTES : (i + 1) * CHUNK_BYTES].copy_(
                    obj.tensor, non_blocking=True
                )
            # The GPU consumer covers every payload byte; copying back for its
            # independent exact comparison happens after the timed fence.
            checksum = consumer.sum(dtype=torch.int64)
            torch.cuda.synchronize()
            ready = time.perf_counter_ns()
            actual = consumer.cpu().numpy().tobytes()
            expected_bytes = b"".join(expected)
            if actual != expected_bytes:
                raise AssertionError("restored GPU consumer bytes differ")
            after = adapter.report_status()
            before["core"], after["core"] = compact_core_pair(
                before["core"], after["core"]
            )
            records.append(
                {
                    "sample": sample - 5,
                    "warmup": sample < 5,
                    "load_ms": (loaded - started) / 1e6,
                    "ready_ms": (ready - started) / 1e6,
                    "gpu_checksum": int(checksum.item()),
                    "payload_sha256": hashlib.sha256(actual).hexdigest(),
                    "before": before,
                    "after": after,
                }
            )
    finally:
        write_json(output / "samples.json", records)
    adapter.submit_unlock(keys)
    terminal = adapter.report_status()
    close = adapter.close_with_outcome()
    if not close.buffers_reusable:
        raise RuntimeError("adapter shutdown retained unresolved registrations")
    manager.free(objects)
    manager.close()
    _RETAINED_RESOURCES.remove(keepalive)
    write_json(
        output / "result.json",
        {
            "mode": mode,
            "samples": samples,
            "terminal": terminal,
            "media_sha256": sha256(media),
            "exact_payloads": True,
        },
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("mode", choices=("direct", "staged"))
    parser.add_argument("output", type=Path)
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--corrupt", action="store_true")
    args = parser.parse_args()
    run(
        json.loads(args.config.read_text()),
        args.mode,
        args.output,
        args.samples,
        args.corrupt,
    )
