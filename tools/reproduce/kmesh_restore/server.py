# SPDX-License-Identifier: MIT
"""Serve the fixed offloaded ancestor through production LMCache MP APIs."""

from __future__ import annotations

import json
from pathlib import Path
import sys

from restore_common import ARENA_BYTES, CAPACITY_BYTES, CHUNK, SLOT_BYTES


def main(config: dict) -> None:
    """Start one loopback storage owner with identical budgets in both arms."""
    from lmcache.v1.distributed.config import (
        EvictionConfig,
        GPUL1ManagerConfig,
        L1ManagerConfig,
        L1MemoryManagerConfig,
        StorageManagerConfig,
    )
    from lmcache.v1.distributed.l2_adapters.config import L2AdaptersConfig
    from lmcache.v1.distributed.l2_adapters.raw_block_l2_adapter import (
        RawBlockL2AdapterConfig,
    )
    from lmcache.v1.mp_observability.config import ObservabilityConfig
    from lmcache.v1.multiprocess.config import (
        DEFAULT_COORDINATOR_CONFIG,
        HTTPFrontendConfig,
        MPServerConfig,
    )
    from lmcache.v1.multiprocess.http_server import run_http_server

    memory = L1MemoryManagerConfig(
        ARENA_BYTES,
        False,
        shm_name="",
        buffer_registration="fixed" if config["mode"] == "staged" else "none",
    )
    if config["mode"] == "direct":
        l1 = GPUL1ManagerConfig(
            memory, tag="gpu", buffer_provider=config["gpu_provider"]
        )
    else:
        l1 = L1ManagerConfig(memory)
    adapter = RawBlockL2AdapterConfig(
        device_path=config["media"],
        capacity_bytes=CAPACITY_BYTES,
        slot_bytes=SLOT_BYTES,
        meta_total_bytes=1 << 20,
        io_engine="io_uring",
        use_odirect=True,
        iouring_queue_depth=8,
        num_store_workers=1,
        num_load_workers=1,
        meta_enable_periodic=False,
    )
    if config["mode"] == "direct":
        adapter.affinity_tag = "gpu"
    run_http_server(
        HTTPFrontendConfig(http_host="127.0.0.1", http_port=config["http_port"]),
        MPServerConfig(
            host="127.0.0.1",
            port=config["mq_port"],
            chunk_size=CHUNK,
            max_gpu_workers=1,
            max_cpu_workers=4,
            isolated_ipc=True,
            worker_reap_timeout_seconds=30,
            worker_registration_grace_seconds=30,
        ),
        StorageManagerConfig(l1, EvictionConfig("noop"), L2AdaptersConfig([adapter])),
        ObservabilityConfig(enabled=False),
        DEFAULT_COORDINATOR_CONFIG,
    )


if __name__ == "__main__":
    main(json.loads(Path(sys.argv[1]).read_text()))
