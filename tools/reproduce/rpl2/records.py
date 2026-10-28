# SPDX-License-Identifier: MIT
"""Request records: one JSON object per line, a header first.

The header names the source, runtime, filesystem type and requested and
actual I/O mode. Each request record covers one whole request: open, read
and close of one immutable object. Digest verification runs after the
timed region and its result is recorded per request. A summary is derived
only from complete request records; nothing is subtracted or estimated.
"""

import json
import math
from pathlib import Path

SCHEMA = "knlp.rpl2.mounted-fs.v1"

HEADER_FIELDS = {
    "schema": str,
    "type": str,
    "status": str,
    "source": dict,
    "runtime": dict,
    "filesystem": dict,
    "io_mode_requested": str,
    "io_mode_actual": str,
    "object_bytes": int,
    "objects": int,
    "requests": int,
    "seed": int,
    "cache_state": str,
    "started_utc": str,
}

REQUEST_FIELDS = {
    "schema": str,
    "type": str,
    "request": int,
    "key": str,
    "generation": int,
    "object_bytes": int,
    "bytes_read": int,
    "io_mode": str,
    "start_ns": int,
    "end_ns": int,
    "latency_ns": int,
    "validation": str,
}

STATUSES = ("ok", "skipped")
VALIDATIONS = ("match", "mismatch", "short_read", "error")
IO_MODES = ("buffered", "direct")


class RecordError(ValueError):
    """A record does not follow the schema."""


def _check_fields(record, fields, where):
    for name, kind in fields.items():
        if name not in record:
            raise RecordError(f"{where}: missing field {name!r}")
        value = record[name]
        if kind is int and (isinstance(value, bool) or not isinstance(value, int)):
            raise RecordError(f"{where}: field {name!r} must be an integer")
        if kind is not int and not isinstance(value, kind):
            raise RecordError(f"{where}: field {name!r} must be {kind.__name__}")
    if record["schema"] != SCHEMA:
        raise RecordError(f"{where}: unknown schema {record['schema']!r}")


def validate_header(record, where="header"):
    _check_fields(record, HEADER_FIELDS, where)
    if record["type"] != "header":
        raise RecordError(f"{where}: first record must be the header")
    if record["status"] not in STATUSES:
        raise RecordError(f"{where}: unknown status {record['status']!r}")
    if record["io_mode_requested"] not in IO_MODES:
        raise RecordError(f"{where}: unknown requested I/O mode")
    if record["status"] == "ok" and record["io_mode_actual"] not in IO_MODES:
        raise RecordError(f"{where}: unknown actual I/O mode")
    if record["status"] == "skipped" and not record.get("reason"):
        raise RecordError(f"{where}: a skipped run must state its reason")


def validate_request(record, header, where):
    _check_fields(record, REQUEST_FIELDS, where)
    if record["type"] != "request":
        raise RecordError(f"{where}: expected a request record")
    if record["validation"] not in VALIDATIONS:
        raise RecordError(f"{where}: unknown validation {record['validation']!r}")
    if record["io_mode"] != header["io_mode_actual"]:
        raise RecordError(f"{where}: I/O mode differs from the header")
    if record["object_bytes"] != header["object_bytes"]:
        raise RecordError(f"{where}: object size differs from the header")
    if record["end_ns"] < record["start_ns"]:
        raise RecordError(f"{where}: end precedes start")
    if record["latency_ns"] != record["end_ns"] - record["start_ns"]:
        raise RecordError(f"{where}: latency is not end minus start")
    if record["validation"] == "match" and (
        record["bytes_read"] != record["object_bytes"]
    ):
        raise RecordError(f"{where}: a match must read the whole object")


def load(path):
    """Return (header, requests) after validating every line."""
    lines = Path(path).read_text().splitlines()
    if not lines:
        raise RecordError(f"{path}: no records")
    records = []
    for number, line in enumerate(lines, 1):
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise RecordError(f"{path}:{number}: not JSON: {exc}") from exc
    header = records[0]
    validate_header(header, f"{path}:1")
    requests = records[1:]
    if header["status"] == "skipped" and requests:
        raise RecordError(f"{path}: a skipped run has request records")
    for number, record in enumerate(requests, 2):
        validate_request(record, header, f"{path}:{number}")
    indexes = [r["request"] for r in requests]
    if indexes != list(range(len(requests))):
        raise RecordError(f"{path}: request indexes are not 0..n-1 in order")
    if header["status"] == "ok" and len(requests) != header["requests"]:
        raise RecordError(
            f"{path}: header promises {header['requests']} requests, "
            f"found {len(requests)}"
        )
    return header, requests


def percentile(values, fraction):
    """Nearest-rank percentile of complete observations."""
    if not values:
        raise ValueError("no observations")
    ordered = sorted(values)
    rank = max(1, math.ceil(fraction * len(ordered)))
    return ordered[rank - 1]


def summarize(header, requests):
    summary = {
        "schema": SCHEMA,
        "type": "summary",
        "status": header["status"],
        "io_mode_requested": header["io_mode_requested"],
        "io_mode_actual": header["io_mode_actual"],
        "filesystem": header["filesystem"].get("type", "unknown"),
        "object_bytes": header["object_bytes"],
        "requests": len(requests),
    }
    if header["status"] == "skipped":
        summary["reason"] = header["reason"]
        return summary
    counts = {v: 0 for v in VALIDATIONS}
    for record in requests:
        counts[record["validation"]] += 1
    summary["validation"] = counts
    good = [r["latency_ns"] for r in requests if r["validation"] == "match"]
    summary["verified_requests"] = len(good)
    if good:
        median = percentile(good, 0.50)
        summary["latency_ms"] = {
            "p50": median / 1e6,
            "p90": percentile(good, 0.90) / 1e6,
            "p99": percentile(good, 0.99) / 1e6,
            "max": max(good) / 1e6,
        }
        summary["median_request_mib_s"] = (
            header["object_bytes"] / (1024 * 1024) / (median / 1e9)
        )
    return summary
