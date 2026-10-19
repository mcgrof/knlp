import pytest

from tools.kv.native.serving_statistics import holm, paired_effect


def test_exact_sign_median_interval_and_separate_effect_screen():
    row = paired_effect(
        list(range(101, 111)), [100] * 10, higher_is_better=True, screen_percent=6
    )
    assert row["wins"] == 10
    assert row["one_sided_sign_p"] == 1 / 1024
    assert row["median_effect_percent"] == pytest.approx(5.5)
    assert row["median_effect_interval_percent"] == pytest.approx([2, 9])
    assert row["median_interval_coverage_at_least"] == 1 - 22 / 1024
    assert row["median_screen_passed"] is False
    assert row["screen_is_not_an_interval_bound"]


def test_latency_direction_ties_and_incomplete_blocks():
    row = paired_effect(
        [90] * 9 + [100], [100] * 10, higher_is_better=False, screen_percent=5
    )
    assert row["one_sided_sign_p"] == 11 / 1024
    assert row["ties"] == 1
    assert row["median_effect_percent"] == pytest.approx(10)
    family = [
        dict(row, id="fi", all_blocks_valid_and_complete=True),
        dict(row, id="fa3", all_blocks_valid_and_complete=False),
    ]
    result = holm(family)
    assert all(r["holm_adjusted_p"] == 22 / 1024 for r in result)
    assert result[0]["superiority_supported"]
    assert not result[1]["superiority_supported"]


def test_small_sample_interval_and_invalid_measurement():
    row = paired_effect([2, 3], [1, 1], higher_is_better=True, screen_percent=1)
    assert row["median_interval_unbounded"]
    assert row["median_effect_interval_percent"] == [None, None]
    with pytest.raises(ValueError):
        paired_effect([float("nan")], [1], higher_is_better=True, screen_percent=1)
    with pytest.raises(ValueError):
        paired_effect([1], [1, 2], higher_is_better=True, screen_percent=1)
