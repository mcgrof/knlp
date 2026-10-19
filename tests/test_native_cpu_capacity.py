# SPDX-License-Identifier: MIT
"""CPU admission must not mistake visible host CPUs for an allocation."""

import pytest

from tools.kv.native.cpu_capacity import capacity


@pytest.mark.parametrize("version", [1, 2])
@pytest.mark.parametrize("quota,expected", [(1020000, 10.2), (2400000, 24), (-1, 32)])
def test_effective_cgroup_quota(tmp_path, version, quota, expected):
    if version == 2:
        (tmp_path / "cpu.max").write_text(f"{'max' if quota < 0 else quota} 100000\n")
    else:
        root = tmp_path / "cpu,cpuacct"
        root.mkdir()
        (root / "cpu.cfs_quota_us").write_text(str(quota))
        (root / "cpu.cfs_period_us").write_text("100000")
    actual = capacity(tmp_path, affinity=32)
    assert actual["effective_cores"] == expected
    assert actual["quota"]["version"] == version
    assert capacity(tmp_path, affinity=8)["effective_cores"] == 8


def test_missing_quota_is_not_assumed_unlimited(tmp_path):
    with pytest.raises(RuntimeError, match="unavailable"):
        capacity(tmp_path, affinity=192)
