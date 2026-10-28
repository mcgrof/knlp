"""Unit tests for summarize.py and check_receipt.py on synthetic records.

Every record here is synthetic, generated with known medians or known
counters; no lab data is used.
"""

import json
import os
import subprocess
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def run(*args):
    return subprocess.run([sys.executable, *args], capture_output=True, text=True)


def reader_record(
    held, block, lat, failures=0, bad=0, batch=1, seconds=0, size=8 << 20
):
    s = sorted(lat)
    return {
        "mode": "read",
        "client": "cli",
        "block": block,
        "held": held,
        "batch": batch,
        "hot": 8,
        "size": size,
        "io": "direct",
        "seconds": seconds,
        "requests": len(lat) + failures,
        "complete": len(lat),
        "failures": failures,
        "verified_objects": len(lat) // 8,
        "digest_mismatches": bad,
        "p50_ms": s[len(s) // 2],
        "p90_ms": s[int(0.9 * len(s))],
        "p99_ms": s[min(len(s) - 1, int(0.99 * len(s)))],
        "max_ms": s[-1],
        "mean_ms": sum(s) / len(s),
        "latencies_ms": lat,
        "first_touch_ms": [],
        "windows": [],
        "held_fd_stats": None,
        "extension": "synthetic",
    }


class Screen(unittest.TestCase):
    def test_paired_block_statistic(self):
        d = tempfile.mkdtemp()
        for b, (stock, held) in enumerate([(1.0, 0.8), (1.2, 0.9), (1.1, 0.9)], 1):
            json.dump(
                reader_record(0, b, [stock] * 99 + [50.0]),
                open(f"{d}/b1-held0-k{b}.json", "w"),
            )
            json.dump(
                reader_record(64, b, [held] * 100),
                open(f"{d}/b1-held64-k{b}.json", "w"),
            )
        p = run(f"{HERE}/summarize.py", "screen", d)
        self.assertEqual(p.returncode, 0, p.stdout + p.stderr)
        s = json.load(open(f"{d}/screen-summary.json"))["shapes"]["b1"]
        self.assertAlmostEqual(s["mean_diff_ms"], 0.7 / 3, places=6)
        self.assertEqual(s["held_capacity"], 64)
        lo, hi = s["boot95_diff_ms"]
        self.assertGreaterEqual(lo, 0.2 - 1e-9)
        self.assertLessEqual(hi, 0.3 + 1e-9)

    def test_failed_or_mismatched_cell_fails(self):
        d = tempfile.mkdtemp()
        json.dump(
            reader_record(0, 1, [1.0] * 10, failures=1),
            open(f"{d}/b1-held0-k1.json", "w"),
        )
        json.dump(
            reader_record(64, 1, [0.9] * 10, bad=1), open(f"{d}/b1-held64-k1.json", "w")
        )
        self.assertEqual(run(f"{HERE}/summarize.py", "screen", d).returncode, 1)


class Sweep(unittest.TestCase):
    def cell(self, d, n, held, ops_after, cpu_after):
        c = f"{d}/n{n}-held{held}"
        os.makedirs(c)
        for k in range(1, n + 1):
            json.dump(
                reader_record(held, 0, [1.0] * 100, batch=16, seconds=10),
                open(f"{c}/c{k}.json", "w"),
            )
        ops = lambda v: "".join(
            f'pnfs_mds_nfs_op{{op="{o}"}} {x}\n' for o, x in v.items()
        )  # noqa: E731
        open(f"{c}/mds-ops.before", "w").write(ops({"open": 0, "close": 0}))
        open(f"{c}/mds-ops.after", "w").write(ops(ops_after))
        open(f"{c}/cg.before", "w").write("mds 0 rondb 0\n")
        open(f"{c}/cg.after", "w").write(f"mds {cpu_after} rondb {cpu_after}\n")

    def test_ops_and_cpu_per_thousand_reads(self):
        d = tempfile.mkdtemp()
        self.cell(d, 2, 0, {"open": 3200, "close": 3200}, 1_600_000)
        self.cell(d, 2, 64, {"open": 0, "close": 0}, 0)
        p = run(f"{HERE}/summarize.py", "sweep", d)
        self.assertEqual(p.returncode, 0, p.stdout + p.stderr)
        rows = {r["held"]: r for r in json.load(open(f"{d}/sweep-summary.json"))}
        # 2 clients x 100 requests x 16 objects = 3200 object reads
        self.assertAlmostEqual(rows[0]["mds_ops_per_1k_reads"]["open"], 1000.0)
        self.assertAlmostEqual(rows[0]["metadata_cpu_ms_per_1k_reads"], 1000.0)
        self.assertEqual(rows[64]["mds_ops_per_1k_reads"], {})
        self.assertAlmostEqual(rows[0]["object_reads_per_s"], 320.0)


class Receipt(unittest.TestCase):
    def write(
        self, d, ds_read=8 << 20, mds_reads=0, ok_digest=True, clean=True, events=None
    ):
        open(f"{d}/client.txt", "w").write(
            f"aa {'aa' if ok_digest else 'bb'} {8 << 20}\n"
        )
        open(f"{d}/ds-io.before", "w").write("io 0 0\n")
        open(f"{d}/ds-io.after", "w").write(f"io {ds_read} 0\n")
        reqs = {"10.0.0.20:2049/nfs": {"nfsv3 READ": 8}}
        if mds_reads:
            reqs["10.0.0.10:2049/nfs"] = {"nfsv4 READ": mds_reads}
        json.dump(
            {"requests_by_peer": reqs, "events": events or {}, "failed_by_peer": {}},
            open(f"{d}/trace-summary.json", "w"),
        )
        open(f"{d}/mds-ops.txt", "w").write('pnfs_mds_nfs_op{op="read"} 0\n')
        open(f"{d}/down.log", "w").write("TEARDOWN CLEAN\n" if clean else "LEAK\n")

    def check(self, **kw):
        d = tempfile.mkdtemp()
        self.write(d, **kw)
        p = run(f"{HERE}/check_receipt.py", d, "10.0.0.20", "10.0.0.10")
        return p.returncode, json.load(open(f"{d}/receipt.json"))

    def test_pass(self):
        rc, r = self.check()
        self.assertEqual((rc, r["result"]), (0, "PASS"))

    def test_each_failure_is_caught(self):
        for kw, check in [
            ({"ds_read": 0}, "ds_bytes_read"),
            ({"mds_reads": 3}, "no_mds_read_rpcs"),
            ({"ok_digest": False}, "digest_match"),
            ({"clean": False}, "teardown_clean"),
            ({"events": {"pnfs_mds_fallback_read_done": 1}}, "no_fallback_or_ff_error"),
        ]:
            rc, r = self.check(**kw)
            self.assertEqual(rc, 1, kw)
            self.assertFalse(r["checks"][check], kw)

    def test_missing_outputs_fail(self):
        d = tempfile.mkdtemp()
        p = run(f"{HERE}/check_receipt.py", d, "10.0.0.20", "10.0.0.10")
        self.assertEqual(p.returncode, 1)
        self.assertFalse(
            json.load(open(f"{d}/receipt.json"))["checks"]["all_steps_ran"]
        )


if __name__ == "__main__":
    unittest.main()
