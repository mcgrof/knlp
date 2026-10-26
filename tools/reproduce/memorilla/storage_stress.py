# SPDX-License-Identifier: MIT
"""Concurrent real-bank replay against original mmap and LMCache storage.

This measures a complete CPU bank fetch plus a validating byte consumer.
It does not label mmap view creation as completed I/O or claim decoder latency.
Use the checkpoint stress command separately for generation comparisons.
"""

import argparse
import ctypes
import gc
import hashlib
import json
import math
import mmap
import os
import resource
import sys
import threading
import time
from concurrent.futures import CancelledError, ThreadPoolExecutor, as_completed
from pathlib import Path
from importlib import metadata

from .environment import dependency_problems, verify_package
from .run import LMCACHE, configuration, source_check, write
from .workloads import percentile, schedule


def tensor_record(tensor):
    """Consume every byte, including pages not faulted by mmap view creation."""
    return {
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype),
        "sha256": hashlib.sha256(tensor.contiguous().numpy().tobytes()).hexdigest(),
    }


def file_residency(paths):
    """Inspect Linux file-page residency without reading payload pages."""
    libc = ctypes.CDLL(None, use_errno=True)
    function = libc.mincore
    function.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p]
    function.restype = ctypes.c_int
    resident = pages = 0
    for path in paths:
        size = path.stat().st_size
        if not size:
            continue
        count = (size + mmap.PAGESIZE - 1) // mmap.PAGESIZE
        with path.open("rb") as stream, mmap.mmap(
            stream.fileno(), 0, access=mmap.ACCESS_COPY
        ) as mapping:
            address = ctypes.addressof(ctypes.c_char.from_buffer(mapping))
            bitmap = (ctypes.c_ubyte * count)()
            if function(address, size, bitmap):
                raise OSError(ctypes.get_errno(), "mincore failed")
            resident += sum(bool(x & 1) for x in bitmap)
            pages += count
    return {
        "resident_pages": resident,
        "pages": pages,
        "resident_fraction": resident / pages if pages else None,
    }


def prepare_file_cache(paths, policy, owned_root):
    """Control only caller-owned files, never global page-cache state."""
    if policy == "cold":
        for path in paths:
            if not path.resolve().is_relative_to(owned_root.resolve()):
                raise ValueError("cache eviction requires files inside the work root")
        for path in paths:
            with path.open("rb") as stream:
                os.posix_fadvise(stream.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)
    elif policy == "hot":
        buffer = bytearray(1 << 20)
        for path in paths:
            with path.open("rb", buffering=0) as stream:
                while stream.readinto(buffer):
                    pass
    elif policy != "unchanged":
        raise ValueError("unknown file-cache policy")
    return file_residency(paths)


def process_usage():
    current = resource.getrusage(resource.RUSAGE_SELF)
    result = {
        "max_rss_kib": current.ru_maxrss,
        "minor_faults": current.ru_minflt,
        "major_faults": current.ru_majflt,
    }
    result["io"] = {
        key: int(value)
        for key, value in (
            line.split(":", 1)
            for line in Path("/proc/self/io").read_text().splitlines()
        )
    }
    cgroup = Path("/sys/fs/cgroup") / Path(
        "/proc/self/cgroup"
    ).read_text().strip().split("::", 1)[1].lstrip("/")
    result["cgroup"] = {}
    for name in [
        "memory.current",
        "memory.peak",
        "memory.max",
        "memory.stat",
        "memory.events",
        "memory.pressure",
    ]:
        path = cgroup / name
        if path.exists():
            result["cgroup"][name] = path.read_text()
    return result


def concurrent_replay(load, records, order, clients, cancel_every=0, timeout=120):
    """Replay immutable identities and drain accepted work after cancellation.

    Cancellation uses real queued futures. A running operation cannot be
    forcibly cancelled: it must finish and release its lease before return.
    Capacity exhaustion receives bounded backpressure, recorded in latency.
    """
    gate = threading.Event()
    submitted = time.monotonic()

    def request(index, collection):
        gate.wait()
        start = time.monotonic()
        retries = 0
        while True:
            try:
                tick = time.monotonic()
                tensor = load(collection)
                loaded = time.monotonic()
                observed = tensor_record(tensor)
                break
            except MemoryError:
                retries += 1
                if time.monotonic() - start > timeout:
                    raise TimeoutError("bank admission exceeded its deadline")
                time.sleep(0.001)
        end = time.monotonic()
        if observed != records[collection]:
            raise ValueError(f"wrong collection bytes or geometry: {collection}")
        return {
            "request_id": index,
            "collection_id": collection,
            "status": "completed",
            "observed": observed,
            "queue_seconds": start - submitted,
            "lookup_seconds": loaded - tick,
            "consumer_seconds": end - loaded,
            "service_seconds": end - start,
            "response_seconds": end - submitted,
            "capacity_retries": retries,
        }

    rows = []
    with ThreadPoolExecutor(max_workers=clients) as executor:
        futures = {
            executor.submit(request, i, collection): (i, collection)
            for i, collection in enumerate(order)
        }
        cancellation = {}
        if cancel_every:
            for future, (index, _) in futures.items():
                if index % cancel_every == 0:
                    cancellation[index] = future.cancel()
        gate.set()
        for future in as_completed(futures):
            index, collection = futures[future]
            try:
                row = future.result()
            except CancelledError:
                row = {
                    "request_id": index,
                    "collection_id": collection,
                    "status": "cancelled",
                }
            row["cancellation_requested"] = index in cancellation
            row["cancellation_accepted"] = cancellation.get(index, False)
            rows.append(row)
    return sorted(rows, key=lambda row: row["request_id"])


def execute(opts):
    if opts.output.exists():
        raise FileExistsError("choose a new storage-stress output directory")
    cfg = configuration(opts.config)
    root = Path(cfg["root"])
    inputs = json.loads((root / "inputs.json").read_text())
    source = Path(inputs["source"])
    source_check(source)
    sys.path.insert(0, str(source))
    from memorilla.embeddings import EmbeddingStore

    from .worker import ArtifactEmbeddingStore

    setup_start = time.monotonic()
    native = EmbeddingStore(
        opts.benchmark, "test", root=inputs["assets"]["data"]["path"]
    )
    collections = native.collection_ids.tolist()
    if opts.banks:
        # Spread the working set across the published index rather than
        # selecting only the smallest or earliest documents.
        collections = collections[:: max(1, len(collections) // opts.banks)][
            : opts.banks
        ]
    expected = {
        collection: tensor_record(native.documents(collection))
        for collection in collections
    }
    store = None
    installation = None
    if cfg["profile"] == "lmcache":
        import lmcache

        installation = verify_package(lmcache, root / "source/lmcache", LMCACHE)
        problems = dependency_problems()
        if problems:
            raise ValueError(problems)
        cfg["source_revision"] = inputs["source_revision"]
        store = ArtifactEmbeddingStore(
            native,
            cfg,
            Path(inputs["assets"]["reader"]["path"]) / "stage3",
            opts.output,
        )
    # Drop the setup's mmap handles before requesting a cold file-cache
    # state. No native source code or shared/global cache setting changes.
    native = EmbeddingStore(
        opts.benchmark, "test", root=inputs["assets"]["data"]["path"]
    )
    if store:
        store.native = native
        paths = sorted((root / "artifacts" / opts.benchmark).glob("*.safetensors"))
        load = store.documents
    else:
        paths = sorted(native.directory.glob("documents-*.safetensors"))
        local = threading.local()

        def load(collection):
            if not hasattr(local, "store"):
                local.store = EmbeddingStore(
                    opts.benchmark, "test", root=inputs["assets"]["data"]["path"]
                )
            return local.store.documents(collection)

    gc.collect()
    setup_seconds = time.monotonic() - setup_start
    before_cache = prepare_file_cache(paths, opts.file_cache, root)
    before = process_usage()
    order = [
        collections[i]
        for i in schedule(len(collections), opts.requests, opts.seed, opts.pattern)
    ]
    start = time.monotonic()
    try:
        rows = concurrent_replay(load, expected, order, opts.clients, opts.cancel_every)
        elapsed = time.monotonic() - start
        after = process_usage()
        after_cache = file_residency(paths)
        storage = store.stats() if store else None
    finally:
        if store:
            store.close()
    opts.output.mkdir(parents=True, exist_ok=True)
    with (opts.output / "requests.jsonl").open("w") as stream:
        for row in rows:
            stream.write(json.dumps(row) + "\n")
    completed = [r for r in rows if r["status"] == "completed"]
    assert (
        len(rows) == opts.requests
        and len({r["request_id"] for r in rows}) == opts.requests
    )
    result = {
        "backend": cfg["profile"],
        "benchmark": opts.benchmark,
        "requests": len(rows),
        "completed": len(completed),
        "cancelled": len(rows) - len(completed),
        "clients": opts.clients,
        "working_set_banks": len(collections),
        "working_set_tensor_bytes": sum(
            math.prod(r["shape"]) * 2 for r in expected.values()
        ),
        "pool_bytes": cfg["pool_bytes"] if store else None,
        "setup_seconds": setup_seconds,
        "elapsed_seconds": elapsed,
        "requests_per_second": len(completed) / elapsed,
        "service_seconds": {
            str(p): percentile([r["service_seconds"] for r in completed], p)
            for p in [0.5, 0.95, 0.99]
        },
        "lookup_seconds": {
            str(p): percentile([r["lookup_seconds"] for r in completed], p)
            for p in [0.5, 0.95, 0.99]
        },
        "response_seconds": {
            str(p): percentile([r["response_seconds"] for r in completed], p)
            for p in [0.5, 0.95, 0.99]
        },
        "capacity_retries": sum(r["capacity_retries"] for r in completed),
        "file_cache_policy": opts.file_cache,
        "cache_before": before_cache,
        "cache_after": after_cache,
        "usage_before": before,
        "usage_after": after,
        "storage": storage,
        "lmcache_installation": installation,
        "correct": True,
        "scope": "CPU bank fetch plus full-byte validating consumer; real published banks; not decoder or individual online token latency. Setup/conversion excluded from request timing and reported separately.",
        "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "author_source": inputs["source_revision"],
        "asset_revisions": {k: v["revision"] for k, v in inputs["assets"].items()},
        "runtime": {
            name: metadata.version(name)
            for name in ["torch", "lmcache", "vllm", "transformers", "safetensors"]
        },
    }
    write(opts.output / "summary.json", result)
    print(json.dumps(result, indent=2))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument(
        "--benchmark", choices=["pubmedqa", "narrativeqa"], default="pubmedqa"
    )
    parser.add_argument("--requests", type=int, default=1024)
    parser.add_argument("--clients", type=int, default=8)
    parser.add_argument("--banks", type=int, default=0)
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument(
        "--pattern",
        choices=["shuffle", "hot", "random", "sequential"],
        default="shuffle",
    )
    parser.add_argument("--cancel-every", type=int, default=0)
    parser.add_argument(
        "--file-cache", choices=["hot", "cold", "unchanged"], default="unchanged"
    )
    parser.add_argument("--output", required=True, type=Path)
    opts = parser.parse_args()
    if min(opts.requests, opts.clients) < 1 or min(opts.banks, opts.cancel_every) < 0:
        parser.error("invalid workload bounds")
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    execute(opts)


if __name__ == "__main__":
    main()
