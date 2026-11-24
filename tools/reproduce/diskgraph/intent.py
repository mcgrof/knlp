# SPDX-License-Identifier: MIT
"""Inspect or replay the pre-page-cache requests from a successful DiskGraph run."""

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
import ctypes
import hashlib
import json
import os
from pathlib import Path
import random
import stat
import threading
import time

from device import load_extents, select

PAGE = 4096
BUF_LIMIT = 1 << 30


def read_trace(path):
    files = {}
    demand = []
    waves = []
    ended = False
    with path.open() as source:
        if json.loads(next(source)) != {
            "event": "meta",
            "version": 1,
            "clock": "CLOCK_MONOTONIC",
        }:
            raise ValueError("Unsupported intent format")
        for line in source:
            row = json.loads(line)
            kind = row["event"]
            if ended:
                raise ValueError("Unexpected event after end of intent capture")
            if (
                kind == "file"
                and row["file"] not in files
                and 0 < row["file"]
                and 0 <= row["size"]
            ):
                files[row["file"]] = row["size"]
            elif kind == "demand":
                size = files[row["file"]]
                if not all(
                    0 <= start < size and 0 < length <= size - start
                    for start, length in row["ranges"]
                ):
                    raise ValueError("Planner range exceeds file size")
                demand.append(row)
            elif kind == "read_batch":
                if not 0 < len(row["reads"]) <= row["qd"] <= 4096:
                    raise ValueError("Read batch does not fit the recorded queue")
                for file, offset, length in row["reads"]:
                    if not (
                        0 <= offset < files[file]
                        and offset % PAGE == 0
                        and 0 < length <= 1 << 20
                        and length % PAGE == 0
                    ):
                        raise ValueError("Invalid aligned read request")
                waves.append(row)
            elif kind == "end":
                ended = True
            else:
                raise ValueError(f"Unexpected intent event {kind}")
    if not ended or not demand or not waves:
        raise ValueError(
            "Expected a complete capture with planner and Buffered requests"
        )
    return files, demand, waves


def distinct_bytes(ranges):
    result = 0
    for intervals in ranges.values():
        previous = 0
        for start, end in sorted(intervals):
            if end > previous:
                result += end - max(start, previous)
                previous = end
    return result


def shape(files, demand, waves):
    planner_sizes = Counter()
    intervals = defaultdict(list)
    for event in demand:
        planner_sizes.update(length for _, length in event["ranges"])
        intervals[event["file"]].extend(
            (offset, offset + size) for offset, size in event["ranges"]
        )
    requested_sizes = Counter()
    eof_bytes = 0
    unique = defaultdict(set)
    for wave in waves:
        for file, offset, length in wave["reads"]:
            requested_sizes[length] += 1
            expected = min(length, files[file] - offset)
            eof_bytes += expected
            unique[file].update(
                range(offset // PAGE, (offset + expected - 1) // PAGE + 1)
            )
    return {
        "planner_ranges": planner_sizes.total(),
        "planner_requested_bytes_with_reuse": sum(
            n * c for n, c in planner_sizes.items()
        ),
        "planner_distinct_bytes": distinct_bytes(intervals),
        "planner_range_size_histogram": dict(sorted(planner_sizes.items())),
        "planner_ranges_lt_4k": sum(c for n, c in planner_sizes.items() if n < PAGE),
        "io_uring_requests": requested_sizes.total(),
        "io_uring_requested_bytes_including_eof": sum(
            n * c for n, c in requested_sizes.items()
        ),
        "io_uring_completed_bytes_expected": eof_bytes,
        "io_uring_request_size_histogram": dict(sorted(requested_sizes.items())),
        "distinct_requested_pages": sum(map(len, unique.values())),
        "lanes": len({w["lane"] for w in waves}),
        "waves": len(waves),
    }


def report(args, files, demand, waves):
    live = json.loads(args.live.read_text())
    if live["backend"] != "buffered":
        raise ValueError("The intent sidecar is only captured for Buffered")
    result = {
        "dataset": live["dataset"],
        "kernel": live["kernel"],
        "seeds_per_pass": live["seeds_per_pass"],
        "passes": [],
        "trace_sha256": hashlib.file_digest(
            args.trace.open("rb"), "sha256"
        ).hexdigest(),
    }
    device_dir = args.live.parent
    capture_path = device_dir / "nvme.jsonl"
    offsets = device_dir / "fiemap.txt"
    extents = None
    if capture_path.is_file():
        extents = load_extents(
            offsets, int((device_dir / "partition-start-sectors").read_text())
        )
    for run in live["passes"]:
        start, end = run["start_ns"], run["end_ns"]
        summary = {k: value for k, value in run.items() if k != "batches"}
        summary.update(
            shape(
                files,
                [d for d in demand if start <= d["at_ns"] < end],
                [w for w in waves if start <= w["at_ns"] < end],
            )
        )
        if extents:
            with capture_path.open() as data:
                _, physical = select(
                    map(json.loads, data),
                    start,
                    end,
                    extents,
                    allow_empty=True,
                    keep_events=False,
                    histograms=True,
                )
            summary["nvme"] = physical
            if summary["planner_distinct_bytes"]:
                summary["nvme_edge_bytes_per_distinct_planner_byte"] = (
                    physical.get("payload_bytes", 0) / summary["planner_distinct_bytes"]
                )
        result["passes"].append(summary)
    if not all(run["io_uring_requests"] for run in result["passes"]):
        raise ValueError("Each live pass must contain recorded Buffered requests")
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


class Request(ctypes.Structure):
    _fields_ = [
        ("offset", ctypes.c_uint64),
        ("length", ctypes.c_uint32),
        ("expected", ctypes.c_uint32),
        ("fd", ctypes.c_int),
    ]


def open_dummy(path, file, size, cold):
    file_path = path / f"{file}.bin"
    mode = os.O_RDONLY | os.O_NOFOLLOW
    if not file_path.exists():
        pattern = random.Random(file).randbytes(1 << 20)
        with file_path.open("xb") as target:
            for offset in range(0, size, len(pattern)):
                target.write(pattern[: min(len(pattern), size - offset)])
            target.flush()
            os.fsync(target.fileno())
    fd = os.open(file_path, mode)
    meta = os.fstat(fd)
    if not stat.S_ISREG(meta.st_mode) or meta.st_size != size:
        os.close(fd)
        raise ValueError(f"Dummy file shape differs for {file}")
    if cold:
        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
    os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_RANDOM)
    return fd


def replay(args, files, _, waves):
    used = sorted({file for wave in waves for file, _, _ in wave["reads"]})
    size = sum(files[f] for f in used)
    if size > args.max_bytes:
        raise ValueError(
            f"Replay requires {size} file bytes. Increase --max-bytes to approve it."
        )
    root = (
        args.directory
        / hashlib.file_digest(args.trace.open("rb"), "sha256").hexdigest()
    )
    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    descriptors = {}
    try:
        for file in used:
            descriptors[file] = open_dummy(root, file, files[file], args.cold)
        latest = defaultdict(list)
        for wave in waves:
            latest[wave["lane"]].append(wave)
        jobs = []
        for lane in latest.values():
            reads = [r for wave in lane for r in wave["reads"]]
            sizes = (ctypes.c_uint32 * len(lane))(*(len(w["reads"]) for w in lane))
            items = (Request * len(reads))(
                *(
                    Request(
                        offset,
                        length,
                        min(length, files[file] - offset),
                        descriptors[file],
                    )
                    for file, offset, length in reads
                )
            )
            jobs.append(
                (
                    items,
                    sizes,
                    len(sizes),
                    max(w["qd"] for w in lane),
                    max(r[2] for r in reads),
                )
            )
        if (
            len(jobs) * args.clients > 512
            or sum(job[3] * job[4] for job in jobs) * args.clients > BUF_LIMIT
        ):
            raise ValueError("Replay concurrency exceeds the buffer budget")

        native = ctypes.CDLL(str(args.library.resolve())).diskgraph_replay
        native.argtypes = [
            ctypes.POINTER(Request),
            ctypes.POINTER(ctypes.c_uint32),
            ctypes.c_size_t,
            ctypes.c_uint,
            ctypes.c_uint32,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_uint64),
            ctypes.POINTER(ctypes.c_uint64),
        ]
        native.restype = ctypes.c_int
        jobs *= args.clients
        results = []
        for pass_id in range(args.passes):
            barrier = threading.Barrier(len(jobs) + 1, timeout=15)

            def worker(job):
                actual, digest = ctypes.c_uint64(), ctypes.c_uint64()
                barrier.wait()
                ret = native(
                    *job,
                    args.mode == "uring",
                    not args.no_digest,
                    ctypes.byref(actual),
                    ctypes.byref(digest),
                )
                if ret:
                    raise OSError(-ret, os.strerror(-ret))
                return actual.value, None if args.no_digest else f"{digest.value:016x}"

            with ThreadPoolExecutor(max_workers=len(jobs)) as pool:
                pending = [pool.submit(worker, job) for job in jobs]
                start = time.monotonic_ns()
                cpu = time.process_time()
                barrier.wait()
                complete = [job.result() for job in pending]
                elapsed = (time.monotonic_ns() - start) / 1e9
                results.append(
                    {
                        "pass": pass_id + 1,
                        "wall_s": elapsed,
                        "cpu_s": time.process_time() - cpu,
                        "completed_bytes": sum(x[0] for x in complete),
                        "per_lane_digests": [x[1] for x in complete],
                    }
                )
    finally:
        for fd in descriptors.values():
            os.close(fd)
    expected = (
        sum(
            min(length, files[file] - offset)
            for wave in waves
            for file, offset, length in wave["reads"]
        )
        * args.clients
    )
    if any(
        r["completed_bytes"] != expected
        or r["per_lane_digests"] != results[0]["per_lane_digests"]
        for r in results
    ):
        raise RuntimeError("Replay completed bytes or payload stamps changed")
    print(
        json.dumps(
            {
                "mode": args.mode,
                "kernel": os.uname().release,
                "lanes": len(jobs),
                "dummy_file_bytes": size,
                "expected_completed_bytes_per_pass": expected,
                "requests_per_pass": sum(len(w["reads"]) for w in waves) * args.clients,
                "passes": results,
            },
            indent=2,
        )
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest="action", required=True)
    p = subs.add_parser("report")
    p.add_argument("trace", type=Path)
    p.add_argument("--live", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p = subs.add_parser("replay")
    p.add_argument("trace", type=Path)
    p.add_argument("--directory", type=Path, required=True)
    p.add_argument("--library", type=Path, required=True)
    p.add_argument("--mode", choices=("pread", "uring"), default="pread")
    p.add_argument("--clients", type=int, default=1)
    p.add_argument("--passes", type=int, default=2)
    p.add_argument("--max-bytes", type=int, default=4 << 30)
    p.add_argument("--cold", action="store_true")
    p.add_argument("--no-digest", action="store_true")
    args = parser.parse_args()
    if args.action == "report":
        report(args, *read_trace(args.trace))
    else:
        if min(args.clients, args.passes, args.max_bytes) < 1:
            parser.error("Positive replay limits are required")
        replay(args, *read_trace(args.trace))


if __name__ == "__main__":
    main()
