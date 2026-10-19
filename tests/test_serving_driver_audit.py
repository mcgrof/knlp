# SPDX-License-Identifier: MIT
"""Known sampling gaps and startup peaks must survive the driver audit."""
import datetime
import json

import pytest

from tools.kv.native.serving_driver_audit import audit


def capture(root, *, gap=False, peak=False, timezone=0):
    def write(path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value))

    write(
        root / "environment/GPU-SAMPLER-TIMEZONE.json",
        {"timezone_offset_seconds": timezone, "timezone_names": ["UTC"]},
    )
    lines = []
    for index, (block, arm) in enumerate(
        (b, a)
        for b in [*range(10), "descriptive"]
        for a in ["candidate", "native_fi", "native_fa3"]
    ):
        start = 1700000000 + index * 10
        stage = f"fp16-1k-{block}-{arm}"
        write(
            root / f"commands/{index}.json",
            {
                "stage": stage,
                "started_at": start,
                "ended_at": start + 4,
                "returncode": 0,
            },
        )
        write(
            root / stage / "server/inputs/protocol.json",
            {"total_driver_envelope_bytes": 4096 * 2**20},
        )
        offsets = [0, 4] if gap and index == 0 else range(5)
        for offset in offsets:
            stamp = datetime.datetime.fromtimestamp(
                start + offset, datetime.timezone.utc
            ).strftime("%Y/%m/%d %H:%M:%S.%f")[:-3]
            memory = 5000 if peak and index == 0 and offset == 0 else 1000
            lines.append(f"{stamp}, NVIDIA H100 80GB HBM3, 20 %, {memory} MiB, 100 W")
    (root / "environment/gpu-samples.csv").write_text("\n".join(lines) + "\n")


def test_complete_windows_include_startup_peak(tmp_path):
    capture(tmp_path, peak=True)
    result = audit(tmp_path, "fp16-1k")
    assert result["all_33_windows_present"] and result["all_sample_windows_covered"]
    assert not result["all_observed_driver_samples_within_envelope"]
    assert (
        max(w["maximum_observed_driver_used_bytes"] for w in result["windows"])
        == 5000 * 2**20
    )


def test_missing_samples_are_disclosed(tmp_path):
    capture(tmp_path, gap=True)
    result = audit(tmp_path, "fp16-1k")
    assert not result["all_sample_windows_covered"]
    assert sum(not w["coverage_within_2_5_seconds"] for w in result["windows"]) == 1


def test_unknown_clock_offset_is_rejected(tmp_path):
    capture(tmp_path, timezone=3600)
    with pytest.raises(AssertionError):
        audit(tmp_path, "fp16-1k")
