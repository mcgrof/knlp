# SPDX-License-Identifier: MIT
"""Select completed DiskGraph reads by phase and XFS file extents."""

from bisect import bisect_right
from collections import Counter
import json
import re

FILE = re.compile(r"^File size of (.+) is ")
EXTENT = re.compile(r"^\s*\d+:\s*\d+(?:\.\.\s*\d+)?:\s*(\d+)(?:\.\.\s*\d+)?:\s*(\d+):")


def load_extents(path, partition_start=0):
    extents = []
    filename = None
    for line in path.read_text().splitlines():
        header = FILE.match(line)
        row = EXTENT.match(line)
        if header:
            filename = header[1]
        elif row:
            if filename is None or any(
                flag in line
                for flag in ("unwritten", "delalloc", "unknown_loc", "shared")
            ):
                raise ValueError(f"Extent cannot be attributed: {line}")
            offset, length = map(int, row.groups())
            offset += partition_start
            extents.append((offset * 512, (offset + length) * 512, filename))
    if not extents:
        raise ValueError(f"No 512-byte-unit extents found in {path}")
    extents.sort()
    if any(left[1] > right[0] for left, right in zip(extents, extents[1:])):
        raise ValueError("Input file extents overlap")
    return extents


def phase_bucket(ts, start, end, count):
    return (ts - start) * count // (end - start)


def select(
    events,
    start,
    end,
    extents,
    allow_empty=False,
    keep_events=True,
    histograms=False,
    payload_buckets=None,
):
    headers = []
    drops = []
    unit = None
    pending = {}
    selected = []
    reads = Counter()
    latency_us = Counter()
    depth = Counter()
    extent_starts = [item[0] for item in extents]
    for event in events:
        kind = event["event_type"]
        if kind in ("capture_meta", "clock_anchor"):
            headers.append(event)
            if kind == "capture_meta":
                unit = event["lba_bytes"]
        if kind == "drops":
            drops.append(event)
        if kind not in ("nvme_cmd", "nvme_cmp"):
            continue
        key = event["hwq"], event["cid"]
        if kind == "nvme_cmd":
            if key in pending:
                raise ValueError(f"Command identity reused in flight: {key}")
            pending[key] = event, len(pending) + 1
            continue
        item = pending.pop(key, None)
        if item is None:
            continue
        command, command_depth = item
        if event["ts"] - command["ts"] != event["lat_ns"]:
            raise ValueError(f"Command completion latency does not reconcile: {key}")
        if not start <= command["ts"] < end or command["op_name"] != "read":
            continue
        if event["status"] != 0:
            raise ValueError(f"In-window read failed: {key}")
        size = command["bytes"]
        offset = command["slba"] * unit
        reads["all_reads"] += 1
        reads["all_bytes"] += size
        i = bisect_right(extent_starts, offset) - 1
        if i < 0 or offset + size > extents[i][1]:
            reads["unattributed_reads"] += 1
            reads["unattributed_bytes"] += size
            if (i >= 0 and offset < extents[i][1]) or (
                i + 1 < len(extent_starts) and extent_starts[i + 1] < offset + size
            ):
                reads["unattributed_overlaps_index_reads"] += 1
                reads["unattributed_overlaps_index_bytes"] += size
            continue
        if keep_events:
            selected.extend((command, event))
        category = "payload" if extents[i][2].endswith("/edges.bin") else "other_index"
        reads[f"{category}_reads"] += 1
        reads[f"{category}_bytes"] += size
        reads[f"size_{size}"] += 1
        if payload_buckets is not None and category == "payload":
            bucket = phase_bucket(command["ts"], start, end, len(payload_buckets))
            payload_buckets[bucket] += size
        if histograms:
            latency_us[(event["lat_ns"] + 999) // 1000] += 1
            depth[command_depth] += 1
    if any(
        start <= c["ts"] < end and c["op_name"] == "read" for c, _ in pending.values()
    ):
        raise ValueError("Source read missing completion at the window boundary")
    if unit is None or len(drops) != 1 or drops[0]["dropped"] != 0:
        raise ValueError("Expected a complete source capture")
    if (
        not reads["payload_reads"]
        and not reads["other_index_reads"]
        and not allow_empty
    ):
        raise ValueError("No fully attributed reads")
    selected.sort(key=lambda e: e["ts"])
    stats = dict(reads)
    stats.setdefault("unattributed_overlaps_index_bytes", 0)
    if histograms:
        stats.update(
            {
                "latency_us_ceiling_histogram": dict(sorted(latency_us.items())),
                "observed_namespace_depth_at_submit_histogram": dict(
                    sorted(depth.items())
                ),
            }
        )
    return headers + selected + drops, stats
