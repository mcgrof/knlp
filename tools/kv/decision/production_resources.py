# SPDX-License-Identifier: MIT
"""Observe the common device envelope without inserting CUDA work.

NVML samples describe whole-device driver memory. They can miss sub-sample
peaks; Torch's allocator maximum is a separate measurement. Numerical-oracle
intervals are explicitly exempt from the production resource ceiling.
"""

import json
import math
from pathlib import Path
import threading
import time


POLICY = {
    "device_name": "NVIDIA H100 80GB HBM3",
    "driver_envelope_bytes": 76 * 1024**3,
    "driver_sample_interval_seconds": 0.1,
    "driver_maximum_sample_gap_seconds": 0.5,
    "server_cpus": 8,
    "client_cpus": 2,
    "control_cpus": 2,
}


def placement(allowed, cpu_quota=None):
    """Use disjoint fixed CPU sets throughout one dedicated device allocation."""
    allowed = sorted(set(allowed))
    needed = sum(POLICY[k] for k in ("server_cpus", "client_cpus", "control_cpus"))
    if len(allowed) < needed or (cpu_quota is not None and cpu_quota < needed):
        raise ValueError("The dedicated allocation lacks the common CPU resources")
    return {
        "server": allowed[:8],
        "client": allowed[8:10],
        "control": allowed[10:12],
    }


def audit_window(rows, start_ns, end_ns, *, policy=POLICY, enforce=True):
    """Require bracketing samples and retain sampling/ceiling failures explicitly."""
    if type(start_ns) is not int or type(end_ns) is not int or end_ns <= start_ns:
        raise ValueError("Invalid device-observation window")
    if not rows:
        raise ValueError("Missing device samples")
    times = [r["monotonic_ns"] for r in rows]
    if any(type(t) is not int for t in times) or any(
        b <= a for a, b in zip(times, times[1:])
    ):
        raise ValueError("Device sample clocks are not strictly increasing")
    before = [i for i, t in enumerate(times) if t <= start_ns]
    after = [i for i, t in enumerate(times) if t >= end_ns]
    if not before or not after:
        raise ValueError("Device samples do not bracket the entire interval")
    selected = rows[before[-1] : after[0] + 1]
    if len({r["uuid"] for r in selected}) != 1:
        raise ValueError("Observed device identity changed")
    for row in selected:
        if row["name"] != policy["device_name"]:
            raise ValueError("Observed GPU differs from the declared hardware")
        if (
            any(
                type(row[key]) is not int or row[key] < 0
                for key in ("driver_used_bytes", "driver_total_bytes")
            )
            or not 0 <= row["driver_used_bytes"] <= row["driver_total_bytes"]
        ):
            raise ValueError("Invalid driver-memory sample")
        if row["driver_total_bytes"] < policy["driver_envelope_bytes"]:
            raise ValueError("Device capacity is smaller than the common envelope")
    gaps = [
        (b["monotonic_ns"] - a["monotonic_ns"]) / 1e9
        for a, b in zip(selected, selected[1:])
    ]
    maximum = max(r["driver_used_bytes"] for r in selected)
    covered = max(gaps) <= policy["driver_maximum_sample_gap_seconds"]
    within = maximum <= policy["driver_envelope_bytes"]
    return {
        "start_ns": start_ns,
        "end_ns": end_ns,
        "samples": len(selected),
        "uuid": selected[0]["uuid"],
        "first_and_last_sample_ns": [
            selected[0]["monotonic_ns"],
            selected[-1]["monotonic_ns"],
        ],
        "largest_sample_gap_seconds": max(gaps),
        "maximum_observed_driver_used_bytes": maximum,
        "driver_envelope_bytes": policy["driver_envelope_bytes"],
        "sampling_coverage_passed": covered,
        "envelope_passed_at_samples": within,
        "enforced": enforce,
        "eligible": covered and (within or not enforce),
        "scope": "Whole-device NVML samples, including the nearest bracketing observations; shorter peaks can be missed. Numerical oracle intervals are exempt only when enforced=false.",
    }


class DriverSampler:
    """A CPU-only observer, with synchronous boundaries and surfaced errors."""

    def __init__(self, output, *, policy=POLICY):
        self.output = Path(output)
        self.policy = dict(policy)
        self.rows = []
        self.error = None
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._thread = None

    def check(self):
        if self.error is not None:
            raise RuntimeError("Driver-memory sampler failed") from self.error

    def sample(self):
        import pynvml as nvml

        self.check()
        with self._lock:
            memory = nvml.nvmlDeviceGetMemoryInfo(self.handle)
            utilization = nvml.nvmlDeviceGetUtilizationRates(self.handle)
            row = {
                "monotonic_ns": time.monotonic_ns(),
                "epoch_seconds": time.time(),
                "uuid": self.identity["uuid"],
                "name": self.identity["name"],
                "driver_used_bytes": int(memory.used),
                "driver_total_bytes": int(memory.total),
                "utilization_percent": int(utilization.gpu),
                "power_watts": nvml.nvmlDeviceGetPowerUsage(self.handle) / 1000,
            }
            with self.output.open("a") as stream:
                stream.write(json.dumps(row, allow_nan=False) + "\n")
            self.rows.append(row)
            return row

    def __enter__(self):
        import pynvml as nvml

        if self.output.exists():
            raise FileExistsError(self.output)
        nvml.nvmlInit()
        try:
            if nvml.nvmlDeviceGetCount() != 1:
                raise ValueError("Require exactly one visible dedicated NVML device")
            self.handle = nvml.nvmlDeviceGetHandleByIndex(0)

            def string(value):
                return value.decode() if isinstance(value, bytes) else value

            self.identity = {
                "uuid": string(nvml.nvmlDeviceGetUUID(self.handle)),
                "name": string(nvml.nvmlDeviceGetName(self.handle)),
                "driver": string(nvml.nvmlSystemGetDriverVersion()),
                "initial_compute_processes": [
                    int(p.pid)
                    for p in nvml.nvmlDeviceGetComputeRunningProcesses(self.handle)
                ],
            }
            if self.identity["name"] != self.policy["device_name"]:
                raise ValueError("Dedicated device hardware differs")
            if self.identity["initial_compute_processes"]:
                raise ValueError("Dedicated GPU already has compute processes")
            interval = self.policy["driver_sample_interval_seconds"]
            if not math.isfinite(interval) or interval <= 0:
                raise ValueError("Invalid NVML sampling interval")
            self.output.parent.mkdir(parents=True, exist_ok=True)
            self.sample()

            def observe():
                try:
                    while not self._stop.wait(interval):
                        self.sample()
                except Exception as error:
                    self.error = error

            self._thread = threading.Thread(target=observe, daemon=True)
            self._thread.start()
            return self
        except BaseException:
            nvml.nvmlShutdown()
            raise

    def window(self, start_ns, end_ns, *, enforce=True):
        self.sample()
        with self._lock:
            return audit_window(
                self.rows, start_ns, end_ns, policy=self.policy, enforce=enforce
            )

    def __exit__(self, kind, value, traceback):
        import pynvml as nvml

        self._stop.set()
        self._thread.join(timeout=5)
        try:
            if self._thread.is_alive():
                raise RuntimeError("Driver sampler did not stop")
            self.sample()
        finally:
            nvml.nvmlShutdown()
