# SPDX-License-Identifier: MIT
"""Check that unmeasured peaks survive the measurement counter reset."""

from types import SimpleNamespace

from tools.kv.native import instrument


def test_memory_peak_reset_boundaries(monkeypatch):
    counters = {"allocated": 500, "reserved": 600, "peak_a": 1000, "peak_r": 2000}
    cuda = instrument.torch.cuda
    monkeypatch.setattr(cuda, "memory_allocated", lambda: counters["allocated"])
    monkeypatch.setattr(cuda, "memory_reserved", lambda: counters["reserved"])
    monkeypatch.setattr(cuda, "max_memory_allocated", lambda: counters["peak_a"])
    monkeypatch.setattr(cuda, "max_memory_reserved", lambda: counters["peak_r"])
    monkeypatch.setattr(cuda, "mem_get_info", lambda: (10000, 20000))

    def reset():
        counters["peak_a"] = counters["allocated"]
        counters["peak_r"] = counters["reserved"]

    monkeypatch.setattr(cuda, "reset_peak_memory_stats", reset)
    worker = SimpleNamespace(model_runner=SimpleNamespace())
    result = instrument.reset_warm_peak(worker)
    assert result["prior_peaks_before_measurement_reset"] == {
        "allocated": 1000,
        "reserved": 2000,
    }
    assert result["peak_allocated_since_reset"] == 500
    assert result["peak_reserved_since_reset"] == 600
