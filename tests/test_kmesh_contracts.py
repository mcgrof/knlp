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
    profile_name = "contexts"

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        cfg = contract.profile(self.profile_name)
        self.cfg = cfg
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
            "profile_name": self.profile_name,
            "profile_sha256": contract.digest(contract.profile_path(self.profile_name)),
            "harness_files_sha256": contract.helper_hashes(),
            "source": {"commit": cfg["lmcache_commit"], "clean": True},
            "source_files_sha256": cfg["source_files_sha256"],
            "additional_sources": {
                package: pin | {"clean": True}
                for package, pin in cfg.get("additional_sources", {}).items()
            },
            "protobuf_descriptors_sha256": cfg.get("protobuf_descriptors_sha256", {}),
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
        return contract.verify(self.root, self.profile_name)

    def test_complete_synthetic_receipt(self):
        self.assertEqual(
            self.check(self.receipt)["executed_cases"], len(self.cfg["cases"])
        )

    def test_profile_selection_is_part_of_the_receipt(self):
        with self.assertRaisesRegex(ValueError, "profile"):
            self.check(self.receipt | {"profile_name": "another-profile"})
        with self.assertRaises(ValueError):
            contract.profile("../profile")

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

    def test_additional_source_and_wire_identity_cannot_be_omitted(self):
        for key in ("additional_sources", "protobuf_descriptors_sha256"):
            candidate = copy.deepcopy(self.receipt)
            if candidate[key]:
                del candidate[key][next(iter(candidate[key]))]
            else:
                candidate[key]["unexpected"] = "different"
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.check(candidate)

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


class RegisteredStateReceiptTests(ContractReceiptTests):
    profile_name = "registered-state"

    def test_capture_cannot_be_written_into_the_additional_source(self):
        vllm = self.root / "vllm"
        with patch.dict(
            "os.environ",
            {
                "KMESH_CONTRACTS_PROFILE": self.profile_name,
                "KMESH_CONTRACTS_SOURCE": str(self.root / "lmcache"),
                "KMESH_CONTRACTS_VLLM_SOURCE": str(vllm),
                "KMESH_CONTRACTS_OUTPUT": str(vllm / "results"),
            },
        ), patch("sys.argv", ["contracts", "run"]), patch.object(
            run.subprocess, "run"
        ) as execute:
            with self.assertRaises(SystemExit):
                run.main()
            execute.assert_not_called()
        with patch(
            "sys.argv",
            [
                "cell",
                "--source",
                str(self.root / "lmcache"),
                "--output",
                str(vllm / "results"),
                "--profile",
                self.profile_name,
                "--vllm-source",
                str(vllm),
            ],
        ), self.assertRaisesRegex(RuntimeError, "outside"):
            cell.main()


class RotaryTransfersReceiptTests(RegisteredStateReceiptTests):
    profile_name = "rotary-transfers"


class OrderedRegistrationReceiptTests(RegisteredStateReceiptTests):
    profile_name = "ordered-registration"


class OwnedRawCopyReceiptTests(OrderedRegistrationReceiptTests):
    profile_name = "owned-raw-copy"


class ManagedRawContextReceiptTests(OwnedRawCopyReceiptTests):
    profile_name = "managed-raw-context"


class ManagedRawServiceReceiptTests(ManagedRawContextReceiptTests):
    profile_name = "managed-raw-service"


class NativeSpansReceiptTests(ManagedRawServiceReceiptTests):
    profile_name = "native-spans"


if __name__ == "__main__":
    unittest.main()
