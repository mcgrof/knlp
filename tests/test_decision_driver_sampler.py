# SPDX-License-Identifier: MIT
"""Memory observation must not wait for unrelated telemetry or log writes."""

import json
import io
from pathlib import Path
import sys
import threading
import time
from types import SimpleNamespace

import pytest

from tools.kv.decision.production_resources import DriverSampler, POLICY, audit_window
from tools.kv.decision import production_resources as resources


@pytest.fixture
def nvml(monkeypatch):
    calls = {"utilization": 0, "power": 0, "shutdown": 0}

    def count(name, result):
        calls[name] += 1
        return result

    module = SimpleNamespace(
        nvmlInit=lambda: None,
        nvmlShutdown=lambda: count("shutdown", None),
        nvmlDeviceGetCount=lambda: 1,
        nvmlDeviceGetHandleByIndex=lambda index: index,
        nvmlDeviceGetUUID=lambda handle: "GPU-fixed",
        nvmlDeviceGetName=lambda handle: POLICY["device_name"],
        nvmlSystemGetDriverVersion=lambda: "test-driver",
        nvmlDeviceGetComputeRunningProcesses=lambda handle: [],
        nvmlDeviceGetMemoryInfo=lambda handle: SimpleNamespace(
            used=60 * 1024**3, total=80 * 1024**3
        ),
        nvmlDeviceGetUtilizationRates=lambda handle: count(
            "utilization", SimpleNamespace(gpu=50)
        ),
        nvmlDeviceGetPowerUsage=lambda handle: count("power", 300000),
    )
    monkeypatch.setitem(sys.modules, "pynvml", module)
    return calls


def test_memory_sampler_does_not_request_unrelated_telemetry(tmp_path, nvml):
    with DriverSampler(tmp_path / "samples.jsonl") as sampler:
        sampler.sample()
    assert nvml == {"utilization": 0, "power": 0, "shutdown": 1}


def delayed_sink(path, monkeypatch):
    entered, release = threading.Event(), threading.Event()
    original = Path.open
    writes = 0

    class SlowStream:
        def __init__(self, stream):
            self.stream = stream

        def write(self, text):
            nonlocal writes
            writes += 1
            if writes == 2:
                entered.set()
                if not release.wait(3):
                    raise TimeoutError("Test did not release the delayed log sink")
            return self.stream.write(text)

        def flush(self):
            return self.stream.flush()

        def close(self):
            return self.stream.close()

        def __enter__(self):
            return self

        def __exit__(self, *args):
            self.close()

    def opened(target, *args, **kwargs):
        stream = original(target, *args, **kwargs)
        if target == path and args and args[0] in ("a", "x"):
            return SlowStream(stream)
        return stream

    monkeypatch.setattr(Path, "open", opened)
    return entered, release


def test_slow_log_sink_does_not_block_memory_sampling(tmp_path, monkeypatch, nvml):
    path = tmp_path / "samples.jsonl"
    entered, release = delayed_sink(path, monkeypatch)
    policy = {**POLICY, "driver_sample_interval_seconds": 0.01}
    with DriverSampler(path, policy=policy) as sampler:
        try:
            assert entered.wait(1)
            deadline = time.monotonic() + 0.8
            while len(sampler.rows) < 20 and time.monotonic() < deadline:
                time.sleep(0.01)
            assert len(sampler.rows) >= 20
        finally:
            release.set()
        window = sampler.window(
            sampler.rows[0]["monotonic_ns"] + 1, time.monotonic_ns()
        )
        assert window["eligible"]
    retained = [json.loads(line) for line in path.read_text().splitlines()]
    assert retained == sampler.rows
    assert all(
        a["monotonic_ns"] < b["monotonic_ns"] for a, b in zip(retained, retained[1:])
    )


def test_log_queue_exhaustion_is_fatal_and_retains_accepted_rows(
    tmp_path, monkeypatch, nvml
):
    path = tmp_path / "samples.jsonl"
    entered, release = delayed_sink(path, monkeypatch)
    monkeypatch.setattr(resources, "MAX_PENDING_SAMPLES", 2)
    policy = {**POLICY, "driver_sample_interval_seconds": 0.01}
    with pytest.raises(RuntimeError, match="sampler failed"):
        with DriverSampler(path, policy=policy) as sampler:
            try:
                assert entered.wait(1)
                deadline = time.monotonic() + 1
                while sampler.error is None and time.monotonic() < deadline:
                    time.sleep(0.01)
                assert sampler.error is not None
                sampler.check()
            finally:
                release.set()
    assert [json.loads(line) for line in path.read_text().splitlines()] == sampler.rows
    assert nvml["shutdown"] == 1


def test_log_write_failure_is_surfaced_before_model_work(tmp_path, monkeypatch, nvml):
    path = tmp_path / "samples.jsonl"
    original = Path.open

    class BrokenStream(io.StringIO):
        def write(self, text):
            raise OSError("Simulated full filesystem")

    monkeypatch.setattr(
        Path,
        "open",
        lambda target, *a, **kw: (
            BrokenStream() if target == path else original(target, *a, **kw)
        ),
    )
    with pytest.raises(RuntimeError, match="sampler failed"):
        with DriverSampler(path):
            pytest.fail("A broken observation log must prevent model startup")
    assert nvml["shutdown"] == 1


def virtual_observations(tmp_path, monkeypatch, durations, *, interval=None):
    """Run the real scheduling loop with controlled query wall times."""
    clock = SimpleNamespace(seconds=0.0, stopped=False)

    class Stop:
        def wait(self, timeout):
            if clock.stopped:
                return True
            assert timeout >= 0
            clock.seconds += timeout
            return False

    policy = dict(POLICY)
    if interval is not None:
        policy["driver_sample_interval_seconds"] = interval
    sampler = DriverSampler(tmp_path / "unused.jsonl", policy=policy)
    sampler._stop = Stop()

    def sample():
        start = round(clock.seconds * 1e9)
        clock.seconds += durations[len(sampler.rows)]
        end = round(clock.seconds * 1e9)
        sampler.rows.append(
            {
                "sample_schema": 2,
                "monotonic_ns": end,
                "memory_query_started_ns": start,
                "memory_query_finished_ns": end,
                "uuid": "GPU-fixed",
                "name": POLICY["device_name"],
                "driver_used_bytes": 60 * 1024**3,
                "driver_total_bytes": 80 * 1024**3,
            }
        )
        clock.stopped = len(sampler.rows) == len(durations)

    sampler.sample = sample
    monkeypatch.setattr(
        resources, "time", SimpleNamespace(monotonic=lambda: clock.seconds)
    )
    sampler._observe()
    assert sampler.error is None
    assert len(sampler.rows) == len(durations)
    audit = audit_window(
        sampler.rows,
        sampler.rows[0]["monotonic_ns"] + 1,
        sampler.rows[-1]["memory_query_started_ns"] - 1,
        policy=policy,
    )
    return sampler.rows, audit


def test_query_cost_does_not_accumulate_into_target_cadence(tmp_path, monkeypatch):
    rows, audit = virtual_observations(tmp_path, monkeypatch, [0.005] * 4)
    starts = [r["memory_query_started_ns"] for r in rows]
    assert starts == [20_000_000, 40_000_000, 60_000_000, 80_000_000]
    assert audit["eligible"]


def test_long_query_is_followed_by_immediate_recovery(tmp_path, monkeypatch):
    rows, audit = virtual_observations(
        tmp_path, monkeypatch, [0.0001, 0.433589902, 0.0001, 0.0001]
    )
    assert rows[2]["memory_query_started_ns"] == rows[1]["memory_query_finished_ns"]
    assert (
        rows[3]["memory_query_started_ns"] - rows[2]["memory_query_started_ns"]
        == 20_000_000
    )
    assert audit["largest_sample_gap_seconds"] == pytest.approx(0.453589902)
    assert audit["eligible"]


def test_slow_query_exposes_insufficient_headroom_at_old_cadence(tmp_path, monkeypatch):
    _, audit = virtual_observations(
        tmp_path,
        monkeypatch,
        [0.0001, 0.433589902, 0.0001, 0.0001],
        interval=0.1,
    )
    assert audit["largest_sample_gap_seconds"] == pytest.approx(0.533589902)
    assert not audit["eligible"]


def test_more_frequent_queries_do_not_relax_the_coverage_gate(tmp_path, monkeypatch):
    _, audit = virtual_observations(
        tmp_path, monkeypatch, [0.0001, 0.49, 0.0001, 0.0001]
    )
    assert POLICY["driver_maximum_sample_gap_seconds"] == 0.5
    assert audit["largest_sample_gap_seconds"] == pytest.approx(0.51)
    assert not audit["sampling_coverage_passed"]
    assert not audit["eligible"]
