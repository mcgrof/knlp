# SPDX-License-Identifier: MIT
"""Check complete native-cache test receipts without importing a GPU runtime."""

import hashlib
import json
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
HARNESS_FILES = ("contract.py", "cell.py", "run.py")
EXECUTOR = "tests/v1/multiprocess/test_cb_plan_executor_gpu.py"
RAW = "tests/v1/test_fused_rerope.py::test_option2_matches_option1_and_preserves_v"
SELECTORS = (
    RAW,
    EXECUTOR + "::test_overlap_slot_reuse_is_bit_exact",
    EXECUTOR + "::test_grouped_layout_rotation_preserves_raw_values",
)


def expected_cases() -> set[str]:
    """Enumerate the selected cases; missing or skipped cases cannot qualify."""
    basic = {
        SELECTORS[1] + f"[{chunks}-{batch}-{fmt}]"
        for chunks, batch in ((12, 4), (96, 16), (44, 16))
        for fmt in ("packed", "split", "split_nhd", "cs_hnd", "cs_nhd")
    }
    grouped = {
        SELECTORS[2] + f"[{entry}-{chunks}-{batch}-{fmt}]"
        for entry in ("direct", "registered")
        for chunks, batch in ((11, 4), (69, 64))
        for fmt in ("split", "split_nhd", "cs_hnd", "cs_nhd")
    }
    return {RAW} | basic | grouped


def digest(path: Path) -> str:
    """Hash one local artifact, following a native-extension symlink."""
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def profile() -> dict:
    """Load the pinned source, native binaries and test configuration."""
    return json.loads((HERE / "profile.json").read_text())


def verify(root: Path) -> dict:
    """Reject partial, skipped, mismatched or failed native captures.

    Saved-file checks establish consistency with the pinned test program.
    They do not independently authenticate the machine that produced it.
    """
    cfg = profile()
    receipt = json.loads((root / "receipt.json").read_text())
    cases = expected_cases()
    if receipt["profile_sha256"] != digest(HERE / "profile.json"):
        raise ValueError("wrong native qualification profile")
    if receipt["harness_files_sha256"] != {
        name: digest(HERE / name) for name in HARNESS_FILES
    }:
        raise ValueError("qualification helper hashes differ")
    if (
        receipt["source"]["commit"] != cfg["lmcache_commit"]
        or not receipt["source"]["clean"]
    ):
        raise ValueError("wrong or edited source")
    if receipt["source_files_sha256"] != cfg["source_files_sha256"]:
        raise ValueError("source file hashes differ")
    if receipt["native_sha256"] != cfg["native_sha256"]:
        raise ValueError("native extension hashes differ")
    if receipt["exit_code"] != 0 or receipt["device"]["backend"] != "cuda":
        raise ValueError("native execution failed or used another backend")
    if receipt["device"]["count"] != 1:
        raise ValueError("select exactly one CUDA device")
    if receipt["selectors"] != list(SELECTORS):
        raise ValueError("different test selectors")
    collected = receipt["collected"]
    if len(collected) != len(cases) or set(collected) != cases:
        raise ValueError("missing, extra or duplicate collected cases")
    reports = receipt["reports"]
    counts = Counter()
    for row in reports:
        if row["outcome"] != "passed" or row.get("wasxfail"):
            raise ValueError("failed, skipped or expected-failure test")
        counts[(row["nodeid"], row["when"])] += 1
    required = Counter(
        (case, stage) for case in cases for stage in ("setup", "call", "teardown")
    )
    if counts != required:
        raise ValueError("incomplete or duplicate test phases")
    xml_path = root / "junit.xml"
    if receipt["junit_sha256"] != digest(xml_path):
        raise ValueError("test XML digest differs")
    xml = ET.parse(xml_path).getroot()
    xml_cases = xml.findall(".//testcase")
    if len(xml_cases) != len(cases) or any(
        case.find(tag) is not None
        for case in xml_cases
        for tag in ("error", "failure", "skipped")
    ):
        raise ValueError("test XML does not report complete passing execution")
    xml_names = Counter((case.get("classname"), case.get("name")) for case in xml_cases)
    names = Counter(
        (case.split("::")[0].removesuffix(".py").replace("/", "."), case.split("::")[1])
        for case in cases
    )
    if xml_names != names:
        raise ValueError("test XML names differ from captured cases")
    return {
        "source_commit": cfg["lmcache_commit"],
        "executed_cases": len(cases),
        "geometry": cfg["geometry"],
        "limits": cfg["limits"],
    }
