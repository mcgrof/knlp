# SPDX-License-Identifier: MIT
"""Reject green exits that conceal incomplete ownership qualification."""

import copy
import json
import tempfile
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path
from unittest.mock import patch

from tools.reproduce.kmesh_contracts import cell, contract, run


class ContractReceiptTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        cfg = contract.profile()
        xml = ET.Element("testsuites")
        suite = ET.SubElement(xml, "testsuite")
        for case in cfg["cases"]:
            path, name = case.split("::")
            ET.SubElement(
                suite,
                "testcase",
                name=name,
                classname=path.removesuffix(".py").replace("/", "."),
            )
        ET.ElementTree(xml).write(self.root / "junit.xml")
        self.receipt = {
            "profile_sha256": contract.digest(contract.HERE / "profile.json"),
            "harness_files_sha256": contract.helper_hashes(),
            "source": {"commit": cfg["lmcache_commit"], "clean": True},
            "source_files_sha256": cfg["source_files_sha256"],
            "native_sha256": {"lmcache_native": "2" * 64},
            "exit_code": 0,
            "execution": {
                "backend": "cpu",
                "visible_cuda_devices": 0,
                "cuda_hidden": True,
            },
            "selectors": cfg["selectors"],
            "collected": cfg["cases"],
            "reports": [
                {"nodeid": case, "when": stage, "outcome": "passed"}
                for case in cfg["cases"]
                for stage in ("setup", "call", "teardown")
            ],
            "junit_sha256": contract.digest(self.root / "junit.xml"),
        }

    def check(self, receipt):
        (self.root / "receipt.json").write_text(json.dumps(receipt))
        return contract.verify(self.root)

    def test_complete_synthetic_receipt(self):
        self.assertEqual(self.check(self.receipt)["executed_cases"], 91)

    def test_skip_and_expected_failure_cannot_qualify(self):
        for changes in ({"outcome": "skipped"}, {"wasxfail": "unavailable"}):
            candidate = copy.deepcopy(self.receipt)
            candidate["reports"][0].update(changes)
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                self.check(candidate)

    def test_missing_duplicate_and_unknown_cases_or_phases(self):
        for key in ("collected", "reports"):
            for values in (
                self.receipt[key][:-1],
                self.receipt[key] + self.receipt[key][:1],
            ):
                with self.subTest(key=key), self.assertRaises(ValueError):
                    self.check(self.receipt | {key: values})
        candidate = copy.deepcopy(self.receipt)
        candidate["reports"][0]["nodeid"] = "another-case"
        with self.assertRaises(ValueError):
            self.check(candidate)

    def test_source_helpers_and_build_provenance_are_required(self):
        for key in ("source_files_sha256", "harness_files_sha256", "native_sha256"):
            candidate = copy.deepcopy(self.receipt)
            del candidate[key][next(iter(candidate[key]))]
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.check(candidate)
        with self.assertRaises(ValueError):
            self.check(
                self.receipt | {"source": self.receipt["source"] | {"clean": False}}
            )

    def test_gpu_or_failed_execution_refuses(self):
        for changes in ({"exit_code": 1}, {"execution": {"backend": "cuda"}}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                self.check(self.receipt | changes)

    def test_xml_identity_and_digest_cannot_disagree(self):
        xml = ET.parse(self.root / "junit.xml")
        xml.find(".//testcase").set("name", "another-test")
        xml.write(self.root / "junit.xml")
        with self.assertRaisesRegex(ValueError, "digest"):
            self.check(self.receipt)
        self.receipt["junit_sha256"] = contract.digest(self.root / "junit.xml")
        with self.assertRaisesRegex(ValueError, "names"):
            self.check(self.receipt)

    def test_parent_and_direct_worker_reject_source_capture_paths(self):
        checkout = Path(run.__file__).resolve().parents[3]
        for source in (checkout, self.root / "lmcache"):
            with self.subTest(source=source), patch.dict(
                "os.environ",
                {
                    "KMESH_CONTRACTS_SOURCE": str(source),
                    "KMESH_CONTRACTS_OUTPUT": str(source / "results"),
                },
            ), patch("sys.argv", ["contracts", "run"]), patch.object(
                run.subprocess, "run"
            ) as execute:
                with self.assertRaises(SystemExit):
                    run.main()
                execute.assert_not_called()
            with patch(
                "sys.argv",
                ["cell", "--source", str(source), "--output", str(source / "results")],
            ), self.assertRaisesRegex(RuntimeError, "outside"):
                cell.main()


if __name__ == "__main__":
    unittest.main()
