# SPDX-License-Identifier: MIT
"""Verify complete CPU ownership and request-transport test execution."""

import json
import re
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

from tools.reproduce.kmesh_native.contract import digest

HERE = Path(__file__).resolve().parent
CHECKOUT = HERE.parents[2]
HELPERS = (
    "tools/reproduce/kmesh_contracts/contract.py",
    "tools/reproduce/kmesh_contracts/cell.py",
    "tools/reproduce/kmesh_contracts/run.py",
    "tools/reproduce/kmesh_native/cell.py",
    "tools/reproduce/kmesh_native/contract.py",
)
PROFILES = {
    "contexts": HERE / "profile.json",
    "registered-state": HERE / "profiles" / "registered-state.json",
    "rotary-transfers": HERE / "profiles" / "rotary-transfers.json",
    "ordered-registration": HERE / "profiles" / "ordered-registration.json",
    "owned-raw-copy": HERE / "profiles" / "owned-raw-copy.json",
    "managed-raw-context": HERE / "profiles" / "managed-raw-context.json",
    "managed-raw-service": HERE / "profiles" / "managed-raw-service.json",
    "native-spans": HERE / "profiles" / "native-spans.json",
    "interior-spans": HERE / "profiles" / "interior-spans.json",
}


def profile_path(name: str = "contexts") -> Path:
    """Select a reviewed, packaged source profile."""
    if type(name) is not str or name not in PROFILES:
        raise ValueError("unknown CPU ownership profile")
    return PROFILES[name]


def profile(name: str = "contexts") -> dict:
    """Load source configuration and its explicit required case selection."""
    return json.loads(profile_path(name).read_text())


def helper_hashes() -> dict:
    """Include the shared pytest capture implementation in receipt identity."""
    return {name: digest(CHECKOUT / name) for name in HELPERS}


def verify(root: Path, name: str = "contexts") -> dict:
    """Reject missing, skipped, failed, or mismatched CPU test phases.

    These saved-file checks establish consistency, not independent hardware or
    build attestation. Native extensions are recorded, while this profile tests
    CPU contracts and does not qualify their GPU kernels.
    """
    cfg = profile(name)
    if cfg["schema_version"] != 1 or not cfg["cases"]:
        raise ValueError("unsupported or empty source profile")
    receipt = json.loads((root / "receipt.json").read_text())
    cases = set(cfg["cases"])
    if len(cases) != len(cfg["cases"]):
        raise ValueError("duplicate case in source profile")
    if receipt["profile_name"] != name or receipt["profile_sha256"] != digest(
        profile_path(name)
    ):
        raise ValueError("wrong CPU ownership profile")
    if receipt["harness_files_sha256"] != helper_hashes():
        raise ValueError("qualification helper hashes differ")
    if receipt["source"] != {"commit": cfg["lmcache_commit"], "clean": True}:
        raise ValueError("wrong or edited source")
    if receipt["source_files_sha256"] != cfg["source_files_sha256"]:
        raise ValueError("source file hashes differ")
    expected_additional = {
        package: {
            "commit": pin["commit"],
            "clean": True,
            "source_files_sha256": pin["source_files_sha256"],
        }
        for package, pin in cfg.get("additional_sources", {}).items()
    }
    if receipt["additional_sources"] != expected_additional:
        raise ValueError("additional source identity differs")
    if receipt["protobuf_descriptors_sha256"] != cfg.get(
        "protobuf_descriptors_sha256", {}
    ):
        raise ValueError("generated protobuf descriptors differ")
    hashes = receipt["native_sha256"]
    if set(hashes) != {"lmcache_native"} or any(
        not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value)
        for value in hashes.values()
    ):
        raise ValueError("native build identity is incomplete")
    if receipt["exit_code"] != 0 or receipt["execution"] != {
        "backend": "cpu",
        "visible_cuda_devices": 0,
        "cuda_hidden": True,
    }:
        raise ValueError("execution failed or used another backend")
    if receipt["selectors"] != cfg["selectors"]:
        raise ValueError("different test selectors")
    collected = receipt["collected"]
    if len(collected) != len(cases) or set(collected) != cases:
        raise ValueError("missing, extra or duplicate collected cases")
    counts = Counter()
    for row in receipt["reports"]:
        if row["outcome"] != "passed" or row.get("wasxfail"):
            raise ValueError("failed, skipped or expected-failure test")
        counts[(row["nodeid"], row["when"])] += 1
    if counts != Counter(
        (case, stage) for case in cases for stage in ("setup", "call", "teardown")
    ):
        raise ValueError("incomplete or duplicate test phases")
    xml_path = root / "junit.xml"
    if receipt["junit_sha256"] != digest(xml_path):
        raise ValueError("test XML digest differs")
    xml_cases = ET.parse(xml_path).getroot().findall(".//testcase")
    if len(xml_cases) != len(cases) or any(
        case.find(tag) is not None
        for case in xml_cases
        for tag in ("error", "failure", "skipped")
    ):
        raise ValueError("test XML is incomplete or failed")
    actual = Counter((case.get("classname"), case.get("name")) for case in xml_cases)
    expected = Counter(
        (case.split("::")[0].removesuffix(".py").replace("/", "."), case.split("::")[1])
        for case in cases
    )
    if actual != expected:
        raise ValueError("test XML names differ")
    return {
        "profile": name,
        "source_commit": cfg["lmcache_commit"],
        "executed_cases": len(cases),
        "limits": cfg["limits"],
    }
