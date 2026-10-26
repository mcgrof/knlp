# SPDX-License-Identifier: MIT
"""Exercise real allocator/catalog failures and process recovery on owned files."""

import argparse
import hashlib
import json
import os
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .run import configuration, write


def open_store(root, directory):
    from lmcache.v1.context_artifacts import ArtifactStore
    from lmcache.v1.memory_allocators.host_memory_allocator import HostMemoryAllocator

    from .worker import catalog_class

    catalog = catalog_class(root)(directory / "catalog.sqlite3")
    allocator = HostMemoryAllocator(65536)
    return ArtifactStore(catalog, allocator), catalog, allocator


def child(root, directory, interrupted):
    store, catalog, allocator = open_store(root, directory)
    if interrupted:
        import lmcache.v1.context_artifacts as implementation

        original = implementation.read_file_range_into

        def wait_after_allocation(*args, **kwargs):
            (directory / "allocated.marker").write_text(str(os.getpid()))
            threading.Event().wait(120)
            return original(*args, **kwargs)

        implementation.read_file_range_into = wait_after_allocation
    with store.lease("bank-a") as obj:
        observed = hashlib.sha256(memoryview(obj.byte_array)).hexdigest()
    assert observed == store.describe("bank-a")["sha256"]
    store.close()
    catalog.close()
    assert allocator.memcheck()
    print(json.dumps({"recovered": True, "sha256": observed}), flush=True)


def expect(exception, operation):
    try:
        operation()
    except exception:
        return
    raise AssertionError(f"expected {exception.__name__}")


def run(root, directory):
    directory.mkdir(parents=True, exist_ok=False)
    payload = bytes(range(256)) * 160
    wanted = hashlib.sha256(payload).hexdigest()
    path = directory / "bank.bin"
    path.write_bytes(payload)
    store, catalog, allocator = open_store(root, directory)
    for key in ["bank-a", "bank-b"]:
        store.publish(
            key, {"kind": "synthetic-fault-fixture"}, path, wanted, len(payload)
        )
    results = []

    def read(key="bank-a"):
        with store.lease(key) as obj:
            assert bytes(obj.byte_array) == payload

    ready = threading.Barrier(17)
    release = threading.Event()

    def shared_read():
        with store.lease("bank-a") as obj:
            ready.wait(timeout=30)
            release.wait(timeout=30)
            return hashlib.sha256(memoryview(obj.byte_array)).hexdigest()

    with ThreadPoolExecutor(max_workers=16) as executor:
        futures = [executor.submit(shared_read) for _ in range(16)]
        try:
            ready.wait(timeout=30)
            assert store.read_count == 1 and store.cache_hits == 15
            assert not store.evict("bank-a")
            expect(RuntimeError, store.close)
            expect(MemoryError, lambda: read("bank-b"))
        finally:
            release.set()
        assert all(f.result() == wanted for f in futures)
    results.extend(
        [
            "concurrent-miss-coalescing",
            "active-lease-eviction-rejected",
            "active-close-rejected",
            "active-capacity-backpressure",
        ]
    )
    assert store.evict("bank-a")
    assert allocator.memcheck()
    path.write_bytes(bytes([payload[0] ^ 1]) + payload[1:])
    expect(ValueError, read)
    results.append("corrupt-bytes-rejected")
    path.write_bytes(payload[:100])
    expect(EOFError, read)
    results.append("short-read-rejected")
    path.unlink()
    expect(FileNotFoundError, read)
    results.append("missing-file-rejected")
    path.write_bytes(payload)
    expect(KeyError, lambda: read("missing-identity"))
    results.append("missing-identity-rejected")
    large = directory / "oversize.bin"
    large.write_bytes(payload * 2)
    store.publish(
        "oversize", {}, large, hashlib.sha256(payload * 2).hexdigest(), len(payload) * 2
    )
    expect(MemoryError, lambda: read("oversize"))
    results.append("oversize-bank-rejected")
    read()
    assert store.evict("bank-a") and allocator.memcheck()
    store.close()
    catalog.close()
    command = [
        sys.executable,
        "-m",
        __name__,
        "--work-root",
        str(root),
        "--directory",
        str(directory),
        "--recover",
    ]
    # __name__ is __main__ when invoked as a module; use the importable name.
    command[2] = "tools.reproduce.memorilla.storage_faults"
    interrupted = subprocess.Popen(
        command + ["--interrupt"], stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    deadline = time.monotonic() + 45
    try:
        while not (directory / "allocated.marker").exists():
            if interrupted.poll() is not None:
                raise RuntimeError(interrupted.communicate()[1].decode())
            if time.monotonic() > deadline:
                raise TimeoutError("interrupted reader did not reach allocated state")
            time.sleep(0.02)
    finally:
        if interrupted.poll() is None:
            interrupted.kill()
        stdout, stderr = interrupted.communicate(timeout=15)
        (directory / "interrupted.stdout").write_bytes(stdout)
        (directory / "interrupted.stderr").write_bytes(stderr)
    assert interrupted.returncode == -9
    recovered = subprocess.run(command, capture_output=True, text=True, timeout=45)
    (directory / "recovery.stdout").write_text(recovered.stdout)
    (directory / "recovery.stderr").write_text(recovered.stderr)
    assert recovered.returncode == 0, recovered.stderr
    assert json.loads(recovered.stdout.splitlines()[-1])["sha256"] == wanted
    results.append("killed-inflight-reader-restart")
    result = {
        "passed": results,
        "count": len(results),
        "correct": True,
        "scope": "Real host allocator, catalog, bounded leases and file reads; synthetic owned payloads; process killed after allocation before read completion. No GPU or power-loss guarantee.",
    }
    write(directory / "summary.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--work-root", type=Path)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--recover", action="store_true")
    parser.add_argument("--interrupt", action="store_true")
    opts = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    root = opts.work_root or Path(configuration(opts.config)["root"])
    if opts.recover:
        child(root, opts.directory, opts.interrupt)
    else:
        print(json.dumps(run(root, opts.directory), indent=2))


if __name__ == "__main__":
    main()
