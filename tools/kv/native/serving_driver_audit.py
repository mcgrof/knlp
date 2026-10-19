# SPDX-License-Identifier: MIT
"""Supplement HTTP boundary snapshots with observed one-second driver maxima."""
from pathlib import Path
import argparse, csv, datetime, hashlib, json, re, statistics

if not __debug__:
    raise RuntimeError("Evidence audits require Python assertions enabled")


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def read(p):
    return json.loads(p.read_text())


def audit(root, prefix):
    path = root / "environment/gpu-samples.csv"
    tzpath = root / "environment/GPU-SAMPLER-TIMEZONE.json"
    tz = read(tzpath)
    assert (
        tz["timezone_names"]
        and tz["timezone_offset_seconds"] == 0
        and set(tz["timezone_names"])
        <= {
            "UTC",
            "GMT",
        }
    )
    samples = []
    for row in csv.reader(path.read_text().splitlines()):
        assert len(row) == 5, row
        t = (
            datetime.datetime.strptime(row[0].strip(), "%Y/%m/%d %H:%M:%S.%f")
            .replace(tzinfo=datetime.timezone.utc)
            .timestamp()
        )
        assert row[1].strip() == "NVIDIA H100 80GB HBM3"
        memory = re.fullmatch(r"(\d+) MiB", row[3].strip())
        util = re.fullmatch(r"(\d+) %", row[2].strip())
        power = re.fullmatch(r"([\d.]+) W", row[4].strip())
        assert memory and util and power
        samples.append((t, int(memory[1]) * 2**20, int(util[1]), float(power[1])))
    assert samples and all(b[0] > a[0] for a, b in zip(samples, samples[1:]))
    pattern = re.compile(
        re.escape(prefix) + r"-(\d+|descriptive)-(candidate|native_fi|native_fa3)$"
    )
    rows = []
    for f in sorted((root / "commands").glob("*.json")):
        command = read(f)
        m = pattern.fullmatch(command["stage"])
        if not m:
            continue
        start, end = command["started_at"], command["ended_at"]
        assert end >= start
        window = [s for s in samples if start <= s[0] <= end]
        assert len(window) > 1
        gaps = [b[0] - a[0] for a, b in zip(window, window[1:])]
        edges = [window[0][0] - start, end - window[-1][0]]
        base = root / command["stage"]
        protocol = read(base / "server/inputs/protocol.json")
        maximum = max(s[1] for s in window)
        coverage = max([*gaps, *edges]) <= 2.5
        rows.append(
            {
                "block": int(m[1]) if m[1] != "descriptive" else m[1],
                "arm": m[2],
                "command_receipt_sha256": sha(f),
                "protocol_sha256": sha(base / "server/inputs/protocol.json"),
                "returncode": command["returncode"],
                "window_seconds": end - start,
                "samples": len(window),
                "first_and_last_sample_epoch": [window[0][0], window[-1][0]],
                "largest_sample_gap_seconds": max(gaps),
                "edge_gaps_seconds": edges,
                "coverage_within_2_5_seconds": coverage,
                "maximum_observed_driver_used_bytes": maximum,
                "driver_envelope_bytes": protocol["total_driver_envelope_bytes"],
                "driver_envelope_passed_at_samples": maximum
                <= protocol["total_driver_envelope_bytes"],
                "gpu_utilization_percent": {
                    "median": statistics.median(s[2] for s in window),
                    "min": min(s[2] for s in window),
                    "max": max(s[2] for s in window),
                },
            }
        )
    identities = {(x["block"], x["arm"]) for x in rows}
    expected = {
        (b, a)
        for b in [*range(10), "descriptive"]
        for a in ["candidate", "native_fi", "native_fa3"]
    }
    assert len(identities) == len(rows) and identities == expected
    return {
        "sampling_interval_nominal_seconds": 1,
        "capture_sha256": sha(path),
        "timezone_receipt_sha256": sha(tzpath),
        "same_host_wallclock_window_mapping": True,
        "all_33_windows_present": True,
        "all_sample_windows_covered": all(
            x["coverage_within_2_5_seconds"] for x in rows
        ),
        "all_observed_driver_samples_within_envelope": all(
            x["driver_envelope_passed_at_samples"] for x in rows
        ),
        "windows": rows,
        "scope": "Observed NVIDIA driver memory during each complete arm command, including startup and shutdown. One-second samples can miss shorter peaks. The 2.5-second flag describes sampling coverage only, not latency or a newly selected performance gate. Same-host wall clocks select windows; no client/server clock subtraction is used for request latency.",
        "reader_sha256": sha(Path(__file__)),
    }


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--prefix", required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    r = audit(a.root, a.prefix)
    a.output.write_text(json.dumps(r, indent=2) + "\n")
    print({k: v for k, v in r.items() if k != "windows"})
