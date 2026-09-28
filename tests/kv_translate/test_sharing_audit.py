# SPDX-License-Identifier: GPL-2.0
"""The forecast and the position audit, on inputs whose answer is known."""

from __future__ import annotations

import json
import os

import pytest

torch = pytest.importorskip("torch")

from research.kv_translate.sharing import audit_positions, forecast  # noqa: E402


def test_bands_stop_at_the_prefix_and_cover_it():
    assert audit_positions.bands(1024) == [(0, 64), (64, 256), (256, 1024)]
    got = audit_positions.bands(4096)
    assert got[0] == (0, 64) and got[-1] == (2048, 4096)
    assert all(a[1] == b[0] for a, b in zip(got, got[1:]))
    assert audit_positions.bands(100)[-1] == (64, 100)


def test_an_identical_cache_has_no_error_anywhere():
    x = torch.randn(2, 300, 16)
    rows = audit_positions.compare(x, x.clone(), 300)
    assert all(r["relative_error"] == pytest.approx(0.0, abs=1e-12) for r in rows)
    assert all(r["cosine"] == pytest.approx(1.0, abs=1e-5) for r in rows)


def test_damage_in_one_band_shows_in_that_band_alone():
    torch.manual_seed(0)
    native = torch.randn(2, 300, 16)
    mapped = native.clone()
    mapped[:, 256:] = 0.0
    rows = {
        (r["from"], r["to"]): r for r in audit_positions.compare(mapped, native, 300)
    }
    assert rows[(0, 64)]["relative_error"] == pytest.approx(0.0, abs=1e-12)
    assert rows[(64, 256)]["relative_error"] == pytest.approx(0.0, abs=1e-12)
    assert rows[(256, 300)]["relative_error"] == pytest.approx(1.0, abs=1e-6)


def write(path, rows):
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


@pytest.fixture
def run(tmp_path):
    d = tmp_path / "run"
    (d / "contexts").mkdir(parents=True)
    (d / "contexts" / "a.json").write_text(
        json.dumps({"finished_utc": "2026-01-01T00:00:00Z"})
    )
    write(
        d / "CACHE_BYTES.jsonl",
        [
            {
                "context_id": "a",
                "role": "source",
                "seconds_prefill": 1.0,
                "seconds_write": 2.0,
                "logical_bytes": 10,
            },
            {
                "context_id": "a",
                "role": "target",
                "seconds_prefill": 3.0,
                "seconds_write": 4.0,
                "logical_bytes": 20,
            },
        ],
    )
    gen = {"context_id": "a", "domain": "code", "length_bucket": 4096, "new_tokens": 5}
    write(
        d / "generations.jsonl",
        [
            {
                **gen,
                "mapper": None,
                "request": {"seconds_total": 1.0},
                "seconds_generating": 4.0,
            },
            {
                **gen,
                "mapper": "cache_bridge",
                "request": {"seconds_total": 2.0},
                "seconds_generating": 3.0,
            },
            {
                **gen,
                "mapper": "full_head",
                "request": {"seconds_total": 50.0},
                "seconds_generating": 50.0,
            },
        ],
    )
    write(
        d / "timings.jsonl",
        [
            {"domain": "code", "length_bucket": 4096, "seconds_total": 0.5}
            for _ in range(6)
        ],
    )
    return str(d)


def records(*items):
    return [{"context_id": c, "domain": d, "length_bucket": n} for c, d, n in items]


def test_the_forecast_multiplies_measured_rates_by_what_is_left(run):
    recs = records(("a", "code", 4096), ("b", "code", 4096), ("c", "code", 4096))
    got = forecast.forecast(run, recs, rate=3.6, storage_rate=0.0, fixed={"load": 100})
    row = got["strata"][0]
    assert row["contexts_left"] == 2 and row["contexts_measured"] == 1
    assert row["seconds_per_context"] == {
        "preparation": 10.0,
        "quality": 10.0,
        "timing": 3.0,
    }
    assert got["seconds"]["total"] == pytest.approx(2 * 23.0 + 100)
    assert got["seconds"]["total_with_contingency"] == pytest.approx(146 * 1.3)
    assert got["dollars_with_contingency"] == pytest.approx(
        round(146 * 1.3 / 3600 * 3.6, 2)
    )
    assert got["complete"] is True


def test_a_length_that_was_not_measured_is_not_guessed(run):
    recs = records(("a", "code", 4096), ("z", "code", 32768), ("y", "document", 4096))
    got = forecast.forecast(run, recs, rate=1.0, storage_rate=0.0, fixed={})
    assert got["complete"] is False
    assert {(u["domain"], u["length"]) for u in got["lengths_with_no_measurement"]} == {
        ("code", 32768),
        ("document", 4096),
    }
    assert got["strata"] == []
