# SPDX-License-Identifier: MIT
"""Define the fixed exact-prefix restoration workload and its measurements."""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import random
import subprocess
import time
import urllib.request

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
REVISION = "7ae557604adf67be50417f59c2c2f167def9a775"
VLLM_COMMIT = "1b76bf9dd9d52efbd75fb89bf2619898d8f01233"
LMCACHE_COMMIT = "007a3388331209737ee19c095e430907fc467952"
ANCESTOR = 4096
TAIL = 512
CHUNK = 256
# BF16 K/V, 24 layers, two KV heads and 64 values per head.
CHUNK_BYTES = 2 * 24 * CHUNK * 2 * 64 * 2
CHUNKS = ANCESTOR // CHUNK
ARENA_BYTES = 256 << 20
SLOT_BYTES = 4 << 20
CAPACITY_BYTES = (1 << 20) + 64 * SLOT_BYTES


def write_json(path: Path, data: object) -> None:
    """Replace a JSON record atomically, retaining actual sample values."""
    temporary = path.with_suffix(path.suffix + ".new")
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def sha256(path: Path) -> str:
    """Hash one artifact without loading it all into memory."""
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            result.update(block)
    return result.hexdigest()


def percentile(values: list[float], q: float) -> float:
    """Return a linearly interpolated percentile of a nonempty finite sample."""
    if not values or any(not math.isfinite(x) or x < 0 for x in values):
        raise ValueError("latency samples must be nonempty, finite and nonnegative")
    ordered = sorted(values)
    position = (len(ordered) - 1) * q / 100
    lower = math.floor(position)
    upper = math.ceil(position)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def distribution(values: list[float]) -> dict:
    """Report the sample count and explicitly defined latency percentiles."""
    return {"count": len(values)} | {
        f"p{q}_ms": percentile(values, q) for q in (50, 95, 99)
    }


def compact_core_pair(before: dict, after: dict) -> tuple[dict, dict]:
    """Retain this sample's native operations and cumulative error counters."""
    old = {
        (op["device_instance_id"], op["operation_id"], op["batch_id"])
        for op in before["io_by_request"]["operations"]
    }
    new = [
        op
        for op in after["io_by_request"]["operations"]
        if (op["device_instance_id"], op["operation_id"], op["batch_id"]) not in old
    ]

    def compact(core: dict, operations: list) -> dict:
        journal = {
            key: value
            for key, value in core["io_by_request"].items()
            if key not in ("operations", "rows", "contexts")
        }
        return core | {"io_by_request": journal | {"operations": operations}}

    return compact(before, []), compact(after, new)


def prompts(fanout: int = 5) -> tuple[list[int], list[list[int]]]:
    """Generate the replay's synthetic ancestor and a prefix-stable tail list."""
    rng = random.Random(20260805)
    ancestor = [rng.randrange(4, 50000) for _ in range(ANCESTOR)]
    return ancestor, [
        ancestor + [rng.randrange(4, 50000) for _ in range(TAIL)] for _ in range(fanout)
    ]


def http(port: int, route: str, body: dict | None = None) -> dict:
    """Call a loopback control endpoint outside the model timing interval."""
    request = urllib.request.Request(
        f"http://127.0.0.1:{port}/{route}",
        data=None if body is None else json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=300) as response:
        return json.load(response)


def cold_l1(port: int) -> dict:
    """Drain external operations, then clear only the server's resident L1."""
    deadline = time.monotonic() + 60
    while time.monotonic() < deadline:
        status = http(port, "status")["storage_manager"]
        l1 = status["l1_manager"]
        stores = status["store_controller"]
        if (
            l1["external_io_lease_count"]
            == stores["pending_keys_count"]
            == stores["in_flight_task_count"]
            == 0
        ):
            http(port, "cache/clear", {"force": False})
            status = http(port, "status")["storage_manager"]
            if status["l1_manager"]["total_object_count"] == 0:
                return status
        time.sleep(0.01)
    raise TimeoutError("L1 did not become empty with all external I/O drained")


def create_media(path: Path) -> None:
    """Initialize a new owned regular file; never open an existing file or device."""
    with path.open("xb") as stream:
        for _ in range(CAPACITY_BYTES // (1 << 20)):
            stream.write(bytes(1 << 20))
        stream.flush()
        os.fsync(stream.fileno())


def source_identity(path: Path, expected: str) -> dict:
    """Authenticate the exact public source commit and reject tracked edits."""
    command = ["git", "-C", str(path)]
    revision = subprocess.check_output(
        command + ["rev-parse", "HEAD"], text=True
    ).strip()
    dirty = subprocess.check_output(
        command + ["status", "--porcelain", "--untracked-files=no"], text=True
    )
    if revision != expected or dirty:
        raise RuntimeError(f"source identity mismatch: {path}")
    return {"commit": revision, "clean": True}
