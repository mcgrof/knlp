# SPDX-License-Identifier: MIT
"""Reject native-test captures whose green exit hides missing execution."""

import copy
import json
import tempfile
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path
from unittest.mock import patch

from tools.reproduce.kmesh_native import cell, contract, run


class NativeReceiptTests(unittest.TestCase):
    profile_name = "paged-cache"

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        cfg = contract.profile(self.profile_name)
        cases = sorted(contract.expected_cases(self.profile_name))
        xml = ET.Element("testsuites")
        suite = ET.SubElement(xml, "testsuite")
        for case in cases:
            path, name = case.split("::")
            ET.SubElement(
                suite,
                "testcase",
                classname=path.removesuffix(".py").replace("/", "."),
                name=name,
            )
        ET.ElementTree(xml).write(self.root / "junit.xml")
        self.receipt = {
            "profile_name": self.profile_name,
            "profile_sha256": contract.digest(contract.profile_path(self.profile_name)),
            "harness_files_sha256": {
                name: contract.digest(contract.HERE / name)
                for name in contract.HARNESS_FILES
            },
            "source": {"commit": cfg["lmcache_commit"], "clean": True},
            "source_files_sha256": cfg["source_files_sha256"],
            "native_sha256": cfg["native_sha256"],
            "exit_code": 0,
            "device": {"backend": "cuda", "count": 1},
            "selectors": contract.selectors(self.profile_name),
            "collected": cases,
            "reports": [
                {"nodeid": name, "when": when, "outcome": "passed", "wasxfail": None}
                for name in cases
                for when in ("setup", "call", "teardown")
            ],
            "junit_sha256": contract.digest(self.root / "junit.xml"),
        }

    def check_receipt(self, receipt):
        (self.root / "receipt.json").write_text(json.dumps(receipt))
        return contract.verify(self.root, self.profile_name)

    def test_complete_synthetic_receipt(self):
        self.assertEqual(
            self.check_receipt(self.receipt)["executed_cases"],
            len(contract.expected_cases(self.profile_name)),
        )

    def test_named_profile_is_required_and_cannot_select_an_external_file(self):
        with self.assertRaisesRegex(ValueError, "profile"):
            self.check_receipt(self.receipt | {"profile_name": "unknown-profile"})
        with self.assertRaises(ValueError):
            contract.profile("../other-profile")

    def test_skip_cannot_qualify(self):
        self.receipt["reports"][0]["outcome"] = "skipped"
        with self.assertRaisesRegex(ValueError, "skipped"):
            self.check_receipt(self.receipt)

    def test_expected_failure_cannot_qualify(self):
        self.receipt["reports"][1]["wasxfail"] = "native binding unavailable"
        with self.assertRaisesRegex(ValueError, "expected-failure"):
            self.check_receipt(self.receipt)

    def test_missing_and_duplicate_phases(self):
        for reports in (
            self.receipt["reports"][:-1],
            self.receipt["reports"] + self.receipt["reports"][:1],
        ):
            candidate = self.receipt | {"reports": reports}
            with self.subTest(count=len(reports)), self.assertRaisesRegex(
                ValueError, "test phases"
            ):
                self.check_receipt(candidate)

    def test_missing_and_duplicate_cases(self):
        for cases in (
            self.receipt["collected"][:-1],
            self.receipt["collected"] + self.receipt["collected"][:1],
        ):
            with self.subTest(count=len(cases)), self.assertRaisesRegex(
                ValueError, "collected cases"
            ):
                self.check_receipt(self.receipt | {"collected": cases})

    def test_other_build_or_source(self):
        for key in ("native_sha256", "source_files_sha256", "harness_files_sha256"):
            candidate = copy.deepcopy(self.receipt)
            candidate[key][next(iter(candidate[key]))] = "0" * 64
            with self.subTest(field=key), self.assertRaises(ValueError):
                self.check_receipt(candidate)
        candidate = self.receipt | {"source": self.receipt["source"] | {"clean": False}}
        with self.assertRaisesRegex(ValueError, "edited source"):
            self.check_receipt(candidate)

    def test_xml_names_must_match_cases(self):
        xml = ET.parse(self.root / "junit.xml")
        xml.find(".//testcase").set("name", "another_test")
        xml.write(self.root / "junit.xml")
        self.receipt["junit_sha256"] = contract.digest(self.root / "junit.xml")
        with self.assertRaisesRegex(ValueError, "XML names"):
            self.check_receipt(self.receipt)

    def test_captured_outputs_must_be_external(self):
        checkout = Path(run.__file__).resolve().parents[3]
        for source in (checkout, self.root / "lmcache"):
            with self.subTest(source=source), patch.dict(
                "os.environ",
                {
                    "KMESH_NATIVE_OUTPUT": str(source / "results"),
                    "KMESH_NATIVE_SOURCE": str(source),
                },
            ), patch("sys.argv", ["native", "run"]), patch.object(
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


class RegisteredCopyReceiptTests(NativeReceiptTests):
    profile_name = "registered-copy"


if __name__ == "__main__":
    unittest.main()
