# SPDX-License-Identifier: GPL-2.0
"""The paired comparison, on outcomes whose answer is known by construction."""

from __future__ import annotations

import json
import os

import pytest

np = pytest.importorskip("numpy")

from research.kv_translate.published import paired  # noqa: E402


def outcomes(bits):
    return {f"leaf/{i}": float(b) for i, b in enumerate(bits)}


def write(work, task, mode, by_leaf):
    d = os.path.join(work, "outcomes")
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, f"{task}.{mode}.json"), "w") as f:
        json.dump(by_leaf, f)


def test_identical_arms_differ_by_nothing_with_no_spread():
    base = outcomes([1, 0, 1, 1, 0, 1, 0, 1])
    r = paired.compare(base, dict(base), resamples=500)
    assert r["difference"] == 0.0
    assert r["difference_interval_95"] == [0.0, 0.0]
    assert r["retention_percent"] == pytest.approx(100.0)
    assert r["only_baseline_right"] == 0 and r["only_arm_right"] == 0
    assert r["both_right"] == 5 and r["neither_right"] == 3


def test_changed_answers_are_counted_by_direction():
    base = outcomes([1, 1, 1, 1, 0, 0, 0, 0, 1, 1])
    arm = outcomes([1, 1, 0, 0, 1, 0, 0, 0, 1, 1])
    r = paired.compare(base, arm, resamples=500)
    assert r["examples"] == 10
    assert r["only_baseline_right"] == 2
    assert r["only_arm_right"] == 1
    assert r["both_right"] == 4 and r["neither_right"] == 3
    assert r["baseline_accuracy"] == pytest.approx(0.6)
    assert r["arm_accuracy"] == pytest.approx(0.5)
    assert r["difference"] == pytest.approx(-0.1)
    assert r["retention_percent"] == pytest.approx(100.0 * 0.5 / 0.6)


def test_pairing_is_tighter_than_treating_the_arms_as_independent():
    """Two arms that agree on all but a few examples have a small difference
    and a narrow interval, however uncertain each accuracy is on its own."""
    rng = np.random.default_rng(3)
    base_bits = (rng.random(400) < 0.7).astype(float)
    arm_bits = base_bits.copy()
    arm_bits[:4] = 1 - arm_bits[:4]
    r = paired.compare(outcomes(base_bits), outcomes(arm_bits), resamples=2000)
    lo, hi = r["difference_interval_95"]
    independent_se = (2 * 0.7 * 0.3 / 400) ** 0.5
    assert hi - lo < 2 * 1.96 * independent_se / 3
    assert lo <= r["difference"] <= hi


def test_the_interval_is_the_same_every_time():
    base = outcomes([1, 0] * 50)
    arm = outcomes([1, 1, 0, 0] * 25)
    a = paired.compare(base, arm, resamples=300)
    b = paired.compare(base, arm, resamples=300)
    assert a["difference_interval_95"] == b["difference_interval_95"]
    assert a["retention_interval_95"] == b["retention_interval_95"]


def test_the_deficit_bound_is_the_pessimistic_side():
    base = outcomes([1] * 80 + [0] * 20)
    arm = outcomes([1] * 70 + [0] * 30)
    r = paired.compare(base, arm, resamples=2000)
    assert r["difference"] == pytest.approx(-0.10)
    assert r["deficit_upper_bound_95_one_sided"] > 0.10


def test_arms_scored_on_different_examples_are_refused():
    base = outcomes([1, 0, 1])
    arm = outcomes([1, 0])
    with pytest.raises(ValueError, match="same examples"):
        paired.compare(base, arm)
    with pytest.raises(ValueError, match="no examples"):
        paired.compare({}, {})


def test_a_grouped_task_pools_the_examples_of_every_subject(tmp_path):
    w = str(tmp_path)
    write(w, "t", "native", {"a": {"0": 1.0, "1": 1.0}, "b": {"0": 0.0, "1": 1.0}})
    write(w, "t", "arm", {"a": {"0": 1.0, "1": 0.0}, "b": {"0": 0.0, "1": 1.0}})
    got = paired.load_outcomes(w, "t", "native")
    assert sorted(got) == ["a/0", "a/1", "b/0", "b/1"]
    r = paired.report(w, ["t"], ["native", "arm"], resamples=200)
    t = r["arms"]["arm"]["tasks"]["t"]
    assert t["examples"] == 4
    assert t["baseline_accuracy"] == pytest.approx(0.75)
    assert t["arm_accuracy"] == pytest.approx(0.5)
    assert "native" not in r["arms"]


def test_the_mean_over_tasks_weights_each_task_equally(tmp_path):
    w = str(tmp_path)
    write(w, "big", "native", {"x": {str(i): 1.0 for i in range(100)}})
    write(w, "big", "arm", {"x": {str(i): 1.0 for i in range(100)}})
    write(w, "small", "native", {"x": {"0": 1.0, "1": 1.0}})
    write(w, "small", "arm", {"x": {"0": 1.0, "1": 0.0}})
    r = paired.report(w, ["big", "small"], ["native", "arm"], resamples=200)
    arm = r["arms"]["arm"]
    assert arm["mean_retention_percent"] == pytest.approx((100.0 + 50.0) / 2)
    assert arm["lowest_task_retention_percent"] == pytest.approx(50.0)
    lo, hi = arm["mean_retention_interval_95"]
    assert lo <= arm["mean_retention_percent"] <= hi
