# SPDX-License-Identifier: MIT
"""Object-set and end-to-end checks on a temporary local directory.

These tests need no NFS mount, root, GPU or network service. They write
small synthetic objects to a temporary directory outside the source tree.
"""

import contextlib
import hashlib
import io
import json
import os
import shutil
import stat
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from tools.reproduce.rpl2 import objects, records, run


class Workspace(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp(prefix="rpl2-test-"))
        self.addCleanup(self.cleanup)
        self.target = self.root / "target"
        self.output = self.root / "output"
        self.target.mkdir()

    def cleanup(self):
        for path in self.root.rglob("*"):
            if path.is_file():
                path.chmod(stat.S_IWUSR | stat.S_IRUSR)
        shutil.rmtree(self.root)

    def cfg(self, **changes):
        value = dict(run.DEFAULTS)
        value |= {
            "profile": "mounted-fs",
            "target_dir": str(self.target),
            "output_dir": str(self.output),
            "object_bytes": 64 * 1024,
            "objects": 3,
            "requests": 7,
            "io_mode": "buffered",
        }
        return value | changes

    def quiet(self, function, *args, **kwargs):
        with contextlib.redirect_stdout(io.StringIO()) as out:
            status = function(*args, **kwargs)
        return status, out.getvalue()


class ObjectSetTests(Workspace):
    def test_objects_are_immutable_and_digested(self):
        directory, manifest = objects.prepare(self.target, 5, 8192, 2)
        self.assertEqual(len(manifest["objects"]), 2)
        for index, item in enumerate(manifest["objects"]):
            path = directory / item["file"]
            data = path.read_bytes()
            self.assertEqual(data, objects.payload(5, index, 8192))
            self.assertEqual(hashlib.sha256(data).hexdigest(), item["sha256"])
            self.assertEqual(stat.S_IMODE(path.stat().st_mode), 0o444)

    def test_matching_set_is_reused(self):
        first, _ = objects.prepare(self.target, 5, 8192, 2)
        mtime = (first / objects.MANIFEST).stat().st_mtime_ns
        second, _ = objects.prepare(self.target, 5, 8192, 2)
        self.assertEqual(first, second)
        self.assertEqual((second / objects.MANIFEST).stat().st_mtime_ns, mtime)

    def test_changed_manifest_is_refused(self):
        directory, manifest = objects.prepare(self.target, 5, 8192, 2)
        manifest["seed"] = 6
        (directory / objects.MANIFEST).write_text(json.dumps(manifest))
        with self.assertRaisesRegex(objects.ObjectError, "refusing"):
            objects.prepare(self.target, 5, 8192, 2)

    def test_unknown_files_are_not_overwritten(self):
        directory = self.target / objects.set_name(5, 8192, 2)
        directory.mkdir()
        (directory / "other").write_text("keep")
        with self.assertRaisesRegex(objects.ObjectError, "without a manifest"):
            objects.prepare(self.target, 5, 8192, 2)
        self.assertEqual((directory / "other").read_text(), "keep")

    def test_mount_type_has_no_source_or_address(self):
        info = objects.mount_of(self.target)
        self.assertIn("type", info)
        self.assertLessEqual(set(info), {"type", "nfs_version"})

    def test_direct_size_must_be_aligned(self):
        with self.assertRaisesRegex(objects.ObjectError, "multiple"):
            objects.check_size(5000, "direct")
        objects.check_size(5000, "buffered")


class RunTests(Workspace):
    def test_buffered_run_records_verified_requests(self):
        status, _ = self.quiet(run.run, self.cfg(), stamp="synthetic")
        self.assertEqual(status, 0)
        path = self.output / "rpl2-synthetic" / "records.jsonl"
        head, requests = records.load(path)
        self.assertEqual(head["io_mode_actual"], "buffered")
        self.assertEqual(len(requests), 7)
        self.assertEqual([r["key"] for r in requests[:4]][-1], "obj-000000")
        self.assertTrue(all(r["validation"] == "match" for r in requests))
        status, _ = self.quiet(run.report, path)
        self.assertEqual(status, 0)
        summary = json.loads((path.parent / "summary.json").read_text())
        self.assertEqual(summary["verified_requests"], 7)

    def test_corrupt_object_is_never_reported_as_a_match(self):
        directory, manifest = objects.prepare(self.target, 1, 64 * 1024, 3)
        victim = directory / manifest["objects"][1]["file"]
        victim.chmod(0o644)
        data = bytearray(victim.read_bytes())
        data[100] ^= 0xFF
        victim.write_bytes(bytes(data))
        status, out = self.quiet(run.run, self.cfg(), stamp="corrupt")
        self.assertEqual(status, 1)
        self.assertIn("did not return the expected bytes", out)
        path = self.output / "rpl2-corrupt" / "records.jsonl"
        _, requests = records.load(path)
        bad = {r["key"] for r in requests if r["validation"] != "match"}
        self.assertEqual(bad, {"obj-000001"})
        status, _ = self.quiet(run.report, path)
        self.assertEqual(status, 1)

    def test_object_truncated_after_preparation_is_a_short_read(self):
        prepared = objects.prepare(self.target, 1, 64 * 1024, 3)
        directory, manifest = prepared
        victim = directory / manifest["objects"][0]["file"]
        victim.chmod(0o644)
        victim.write_bytes(victim.read_bytes()[:1000])
        with self.assertRaisesRegex(objects.ObjectError, "wrong size"):
            objects.prepare(self.target, 1, 64 * 1024, 3)
        with mock.patch.object(objects, "prepare", return_value=prepared):
            status, _ = self.quiet(run.run, self.cfg(), stamp="short")
        self.assertEqual(status, 1)
        _, requests = records.load(self.output / "rpl2-short" / "records.jsonl")
        self.assertEqual(requests[0]["validation"], "short_read")
        self.assertEqual(requests[0]["bytes_read"], 1000)

    def test_unsupported_direct_io_is_a_clear_skip(self):
        with mock.patch.object(objects, "probe_direct", return_value=(False, "test")):
            status, out = self.quiet(run.run, self.cfg(io_mode="direct"), stamp="skip")
        self.assertEqual(status, 0)
        self.assertIn("SKIP", out)
        path = self.output / "rpl2-skip" / "records.jsonl"
        head, requests = records.load(path)
        self.assertEqual((head["status"], requests), ("skipped", []))
        status, _ = self.quiet(run.report, path)
        self.assertEqual(status, 0)

    def test_direct_run_when_the_filesystem_supports_it(self):
        supported, reason = objects.probe_direct(self.target)
        if not supported:
            self.skipTest(f"direct I/O unavailable here: {reason}")
        status, _ = self.quiet(run.run, self.cfg(io_mode="direct"), stamp="direct")
        self.assertEqual(status, 0)
        head, requests = records.load(self.output / "rpl2-direct" / "records.jsonl")
        self.assertEqual(head["io_mode_actual"], "direct")
        self.assertTrue(all(r["validation"] == "match" for r in requests))

    def test_paths_inside_the_source_tree_are_refused(self):
        with self.assertRaises(SystemExit):
            self.quiet(run.run, self.cfg(output_dir=str(run.ROOT / "build")))
        with self.assertRaises(SystemExit):
            self.quiet(run.run, self.cfg(target_dir=str(run.ROOT)))

    def test_missing_paths_name_the_make_variable(self):
        with self.assertRaisesRegex(SystemExit, "RPL2_TARGET"):
            run.run(self.cfg(target_dir=""))

    def test_doctor_reports_a_usable_target(self):
        status, out = self.quiet(run.doctor, self.cfg())
        self.assertEqual(status, 0, out)
        self.assertIn("is writable", out)
        status, out = self.quiet(run.doctor, self.cfg(target_dir=str(self.root / "x")))
        self.assertEqual(status, 1)


class ConfigurationTests(Workspace):
    def test_defconfig_values_are_read(self):
        path = self.root / "config"
        path.write_text(
            "\n".join(
                [
                    "CONFIG_KNLP_REPRODUCE_RPL2=y",
                    "CONFIG_KNLP_RPL2_MOUNTED_FS=y",
                    "CONFIG_KNLP_RPL2_IO_BUFFERED=y",
                    'CONFIG_KNLP_RPL2_TARGET_DIR="/synthetic/target"',
                    'CONFIG_KNLP_RPL2_OUTPUT_DIR=""',
                    "CONFIG_KNLP_RPL2_OBJECT_BYTES=4096",
                    "CONFIG_KNLP_RPL2_OBJECTS=2",
                    "CONFIG_KNLP_RPL2_REQUESTS=3",
                    "CONFIG_KNLP_RPL2_SEED=9",
                ]
            )
            + "\n"
        )
        cfg = run.configuration(path)
        self.assertEqual(cfg["profile"], "mounted-fs")
        self.assertEqual(cfg["io_mode"], "buffered")
        self.assertEqual(cfg["target_dir"], "/synthetic/target")
        self.assertEqual(
            (cfg["object_bytes"], cfg["objects"], cfg["requests"], cfg["seed"]),
            (4096, 2, 3, 9),
        )

    def test_other_profiles_are_refused(self):
        path = self.root / "config"
        path.write_text("CONFIG_KNLP_REPRODUCE_PD=y\n")
        with self.assertRaises(SystemExit):
            run.configuration(path)


if __name__ == "__main__":
    unittest.main()
