# SPDX-License-Identifier: MIT
"""Schema checks for request records, using synthetic records only."""

import json
import tempfile
import unittest
from pathlib import Path

from tools.reproduce.rpl2 import records


def header(**changes):
    value = {
        "schema": records.SCHEMA,
        "type": "header",
        "status": "ok",
        "source": {"repository": "knlp", "commit": "synthetic"},
        "runtime": {"python": "3.x"},
        "filesystem": {"type": "synthetic"},
        "io_mode_requested": "buffered",
        "io_mode_actual": "buffered",
        "object_bytes": 4096,
        "objects": 2,
        "requests": 2,
        "seed": 1,
        "cache_state": "uncontrolled",
        "started_utc": "2000-01-01T00:00:00+00:00",
    }
    return value | changes


def request(index, **changes):
    value = {
        "schema": records.SCHEMA,
        "type": "request",
        "request": index,
        "key": f"obj-{index:06d}",
        "generation": 1,
        "object_bytes": 4096,
        "bytes_read": 4096,
        "io_mode": "buffered",
        "start_ns": 1000 * index,
        "end_ns": 1000 * index + 500,
        "latency_ns": 500,
        "validation": "match",
    }
    return value | changes


class RecordTests(unittest.TestCase):
    def write(self, lines):
        directory = Path(tempfile.mkdtemp())
        self.addCleanup(lambda: [p.unlink() for p in directory.iterdir()])
        path = directory / "records.jsonl"
        path.write_text("".join(json.dumps(line) + "\n" for line in lines))
        return path

    def test_valid_records_load_and_summarize(self):
        path = self.write([header(), request(0), request(1)])
        head, requests = records.load(path)
        summary = records.summarize(head, requests)
        self.assertEqual(summary["verified_requests"], 2)
        self.assertEqual(summary["validation"]["match"], 2)
        self.assertAlmostEqual(summary["latency_ms"]["p50"], 0.0005)

    def test_missing_field_is_rejected(self):
        bad = request(0)
        del bad["latency_ns"]
        path = self.write([header(requests=1), bad])
        with self.assertRaisesRegex(records.RecordError, "latency_ns"):
            records.load(path)

    def test_latency_must_equal_end_minus_start(self):
        path = self.write([header(requests=1), request(0, latency_ns=7)])
        with self.assertRaisesRegex(records.RecordError, "end minus start"):
            records.load(path)

    def test_match_must_cover_the_whole_object(self):
        path = self.write([header(requests=1), request(0, bytes_read=10)])
        with self.assertRaisesRegex(records.RecordError, "whole object"):
            records.load(path)

    def test_missing_requests_are_rejected(self):
        path = self.write([header(), request(0)])
        with self.assertRaisesRegex(records.RecordError, "promises"):
            records.load(path)

    def test_mode_must_match_header(self):
        path = self.write([header(requests=1), request(0, io_mode="direct")])
        with self.assertRaisesRegex(records.RecordError, "I/O mode"):
            records.load(path)

    def test_skipped_run_needs_reason_and_no_requests(self):
        skipped = header(status="skipped", io_mode_actual="none")
        with self.assertRaisesRegex(records.RecordError, "reason"):
            records.load(self.write([skipped]))
        skipped["reason"] = "synthetic"
        head, requests = records.load(self.write([skipped]))
        self.assertEqual(records.summarize(head, requests)["status"], "skipped")
        with self.assertRaisesRegex(records.RecordError, "skipped run"):
            records.load(self.write([skipped, request(0)]))

    def test_failed_requests_are_excluded_from_latency(self):
        slow = request(1, validation="mismatch", end_ns=1000 + 9_000_000)
        slow["latency_ns"] = slow["end_ns"] - slow["start_ns"]
        path = self.write([header(), request(0), slow])
        summary = records.summarize(*records.load(path))
        self.assertEqual(summary["verified_requests"], 1)
        self.assertEqual(summary["validation"]["mismatch"], 1)
        self.assertAlmostEqual(summary["latency_ms"]["max"], 0.0005)

    def test_percentile_uses_nearest_rank(self):
        values = list(range(1, 101))
        self.assertEqual(records.percentile(values, 0.5), 50)
        self.assertEqual(records.percentile(values, 0.99), 99)
        self.assertEqual(records.percentile([5], 0.99), 5)


if __name__ == "__main__":
    unittest.main()
