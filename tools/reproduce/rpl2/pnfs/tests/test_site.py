"""Unit tests for mountstats.py and site_summary.py on synthetic records.

The mountstats text is a hand-written sample in the kernel's format, and
every reader record is synthetic; no measured data is used.
"""

import json
import os
import subprocess
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
import mountstats  # noqa: E402

SAMPLE = """device rootfs mounted on / with fstype rootfs
device srv:/export mounted on /mnt/pnfs with fstype nfs4 statvers=1.1
\topts:\trw,vers=4.2,rsize=1048576,wsize=1048576,hard,proto=tcp
\tpnfs:\tLAYOUT_FLEX_FILES
\tper-op statistics
\t        NULL: 0 0 0 0 0 0 0 0 0
\t        READ: {read} {read} 0 1000 2000 0 {rtt} 30 0
\t        OPEN: {open} {open} 0 100 200 0 {rtt} 3 0
\t       CLOSE: {open} {open} 0 100 200 0 {rtt} 3 0
\t     GETATTR: 5 5 0 100 200 0 1 1 0
"""


def record(mode, pipeline, objs_per_s, cpu_us, hits, unsupported=0):
    batch, wall = 16, 2.0
    complete = int(objs_per_s * wall / batch)
    return {
        "read_mode": mode,
        "pipeline": pipeline,
        "batch": batch,
        "complete": complete,
        "wall_s": wall,
        "cpu_s": cpu_us * complete * batch / 1e6,
        "p50_ms": 0.1,
        "p99_ms": 0.3,
        "failures": 0,
        "digest_mismatches": 0,
        "read_stats": {
            "inline_hits": hits,
            "inline_misses": 0,
            "fanned_out": complete * batch - hits,
            "inline_unsupported": unsupported,
        },
    }


class MountstatsTest(unittest.TestCase):
    def test_parse_and_delta(self):
        before = mountstats.parse(SAMPLE.format(read=10, open=4, rtt=2))["/mnt/pnfs"]
        after = mountstats.parse(SAMPLE.format(read=26, open=20, rtt=5))["/mnt/pnfs"]
        self.assertEqual(before["fstype"], "nfs4")
        self.assertIn("pnfs:\tLAYOUT_FLEX_FILES", "\n".join(before["header"]))
        d = mountstats.delta(before, after)
        self.assertEqual(d["READ"], {"calls": 16, "rtt_ms": 3})
        self.assertEqual(d["OPEN"]["calls"], 16)
        self.assertNotIn("GETATTR", d)  # unchanged ops are dropped
        self.assertIsNone(mountstats.delta(None, after))

    def test_snapshot_is_none_off_nfs(self):
        self.assertIsNone(mountstats.snapshot(tempfile.gettempdir()))


class SiteSummaryTest(unittest.TestCase):
    def test_readmodes_and_fingerprint(self):
        with tempfile.TemporaryDirectory() as out:
            rm = os.path.join(out, "readmodes")
            os.makedirs(rm)
            json.dump(
                record("pool", 1, 500000, 14.0, 0), open(f"{rm}/pool-p1.json", "w")
            )
            r = record("inline", 4, 600000, 6.5, 0, unsupported=1)
            json.dump(r, open(f"{rm}/inline-p4.json", "w"))
            fp = os.path.join(out, "fingerprint")
            os.makedirs(fp)
            for held, calls in ((0, 100), (64, 0)):
                rpcs = {"OPEN": {"calls": calls, "rtt_ms": 1}} if calls else {}
                json.dump(
                    {
                        "held": held,
                        "hot": 8,
                        "batch": 1,
                        "complete": 100,
                        "p50_ms": 1.0,
                        "failures": 0,
                        "digest_mismatches": 0,
                        "nfs_rpcs": {"first": {}, "repeat": rpcs},
                    },
                    open(f"{fp}/held{held}.json", "w"),
                )
            p = subprocess.run(
                [sys.executable, os.path.join(HERE, "site_summary.py"), out],
                capture_output=True,
                text=True,
            )
            self.assertEqual(p.returncode, 0, p.stderr)
            s = json.load(open(os.path.join(out, "site-summary.json")))
            rows = {(x["mode"], x["in_flight"]): x for x in s["readmodes"]}
            self.assertEqual(rows[("pool", 1)]["objects_per_s"], 500000)
            self.assertAlmostEqual(rows[("pool", 1)]["cpu_us_per_object"], 14.0, 1)
            self.assertEqual(rows[("inline", 4)]["inline_unsupported"], 1)
            self.assertEqual(s["fingerprint"]["0"]["repeat"], {"OPEN": 1.0})
            self.assertEqual(s["fingerprint"]["64"]["repeat"], {})
            self.assertIn(
                "read_mode", open(os.path.join(out, "site-summary.md")).read()
            )


if __name__ == "__main__":
    unittest.main()
