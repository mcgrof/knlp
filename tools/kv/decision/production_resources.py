# SPDX-License-Identifier: MIT
"""Observe the common device envelope without inserting CUDA work.

NVML samples describe whole-device driver memory. They can miss sub-sample
peaks; Torch's allocator maximum is a separate measurement. Numerical-oracle
intervals are explicitly exempt from the production resource ceiling.
"""

import json
import math
from pathlib import Path
import queue
import threading
import time


POLICY = {
    "device_name": "NVIDIA H100 80GB HBM3",
    "driver_envelope_bytes": 76 * 1024**3,
    "driver_sample_interval_seconds": 0.02,
    "driver_maximum_sample_gap_seconds": 0.5,
    "server_cpus": 8,
    "client_cpus": 2,
    "control_cpus": 2,
}
MAX_PENDING_SAMPLES = 512


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
    versions = {r.get("sample_schema", 1) for r in rows}
    if (
        any(type(r.get("sample_schema", 1)) is not int for r in rows)
        or len(versions) != 1
        or not versions <= {1, 2}
    ):
        raise ValueError("Mixed or unknown device-sample schemas")
    # A driver read can occur anywhere between entry and return. Preserve old
    # point-timestamp captures, but use conservative bounds for new captures.
    starts = times
    if versions == {2}:
        starts = [r.get("memory_query_started_ns") for r in rows]
        if (
            any(type(t) is not int for t in starts)
            or any(
                start > end
                or type(row.get("memory_query_finished_ns")) is not int
                or row["memory_query_finished_ns"] != end
                for row, start, end in zip(rows, starts, times)
            )
            or any(a > b for a, b in zip(times, starts[1:]))
        ):
            raise ValueError("Invalid memory-query clock bounds")
    before = [i for i, t in enumerate(times) if t <= start_ns]
    after = [i for i, t in enumerate(starts) if t >= end_ns]
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
    first, last = before[-1], after[0]
    gaps = [(times[i + 1] - starts[i]) / 1e9 for i in range(first, last)]
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
    """Sample memory independently of logging, with lossless bounded delivery.

    Utilization and power queries do not belong on this safety-critical path.
    Slow log I/O is isolated from observation; errors, queue exhaustion and
    failure to flush a boundary all reject the invocation rather than lose rows.
    """

    def __init__(self, output, *, policy=POLICY):
        self.output = Path(output)
        self.policy = dict(policy)
        self.rows = []
        self.error = None
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._thread = None
        self._writer_thread = None
        self._writer_stop = threading.Event()
        self._pending = queue.Queue(maxsize=MAX_PENDING_SAMPLES)
        self._written = 0
        self._write_condition = threading.Condition()

    def check(self):
        if self.error is not None:
            raise RuntimeError("Driver-memory sampler failed") from self.error

    def sample(self):
        import pynvml as nvml

        self.check()
        with self._lock:
            started = time.monotonic_ns()
            memory = nvml.nvmlDeviceGetMemoryInfo(self.handle)
            finished = time.monotonic_ns()
            row = {
                "sample_schema": 2,
                "monotonic_ns": finished,
                "memory_query_started_ns": started,
                "memory_query_finished_ns": finished,
                "epoch_seconds": time.time(),
                "uuid": self.identity["uuid"],
                "name": self.identity["name"],
                "driver_used_bytes": int(memory.used),
                "driver_total_bytes": int(memory.total),
            }
            try:
                self._pending.put_nowait(row)
            except queue.Full as error:
                self.error = error
                raise RuntimeError("Driver sample log queue is full") from error
            self.rows.append(row)
            return row

    def _write(self, stream):
        try:
            while not self._writer_stop.is_set() or not self._pending.empty():
                try:
                    row = self._pending.get(timeout=0.1)
                except queue.Empty:
                    continue
                stream.write(json.dumps(row, allow_nan=False) + "\n")
                stream.flush()
                with self._write_condition:
                    self._written += 1
                    self._write_condition.notify_all()
        except Exception as error:
            self.error = error
            self._stop.set()
        finally:
            try:
                stream.close()
            except Exception as error:
                self.error = self.error or error
            with self._write_condition:
                self._write_condition.notify_all()

    def _flush_through(self, count):
        deadline = time.monotonic() + 5
        with self._write_condition:
            while self._written < count:
                self.check()
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise RuntimeError("Driver sample log missed its flush boundary")
                self._write_condition.wait(min(remaining, 0.1))
        self.check()

    def _observe(self):
        """Target query-start cadence and recover immediately after a slow read.

        A synchronous driver query can occupy most of the allowed coverage
        gap. Its latency must not be followed by an unconditional extra sleep.
        Start the next query when its target is due, without accumulating a
        burst of missed historical deadlines. The audit still uses the full
        entry/return bounds and rejects inadequate coverage.
        """
        interval = self.policy["driver_sample_interval_seconds"]
        deadline = time.monotonic() + interval
        try:
            while not self._stop.wait(max(0.0, deadline - time.monotonic())):
                started = time.monotonic()
                self.sample()
                deadline = started + interval
        except Exception as error:
            self.error = error

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
            stream = self.output.open("x")
            self._writer_thread = threading.Thread(
                target=self._write, args=(stream,), daemon=True
            )
            self._writer_thread.start()
            self.sample()
            self._flush_through(1)

            self._thread = threading.Thread(target=self._observe, daemon=True)
            self._thread.start()
            return self
        except BaseException:
            self._stop.set()
            self._writer_stop.set()
            if self._writer_thread is not None:
                self._writer_thread.join(timeout=5)
            nvml.nvmlShutdown()
            raise

    def window(self, start_ns, end_ns, *, enforce=True):
        self.sample()
        with self._lock:
            rows = list(self.rows)
        self._flush_through(len(rows))
        return audit_window(rows, start_ns, end_ns, policy=self.policy, enforce=enforce)

    def __exit__(self, kind, value, traceback):
        import pynvml as nvml

        self._stop.set()
        self._thread.join(timeout=5)
        try:
            if self._thread.is_alive():
                raise RuntimeError("Driver sampler did not stop")
            self.sample()
        finally:
            self._writer_stop.set()
            self._writer_thread.join(timeout=5)
            nvml.nvmlShutdown()
        if self._writer_thread.is_alive():
            raise RuntimeError("Driver sample log writer did not stop")
        self.check()
