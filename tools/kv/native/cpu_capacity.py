# SPDX-License-Identifier: MIT
"""Read effective affinity and cgroup v1/v2 CPU quotas without host-wide scans."""

import argparse
import json
import os
from pathlib import Path


def quota(root):
    root = Path(root)
    v2 = root / "cpu.max"
    if v2.is_file():
        amount, period = v2.read_text().split()
        return {
            "version": 2,
            "path": str(v2),
            "raw": [amount, period],
            "cores": None if amount == "max" else int(amount) / int(period),
        }
    for directory in (root, root / "cpu", root / "cpu,cpuacct", root / "cpuacct,cpu"):
        amount, period = directory / "cpu.cfs_quota_us", directory / "cpu.cfs_period_us"
        if amount.is_file() and period.is_file():
            q, p = int(amount.read_text()), int(period.read_text())
            return {
                "version": 1,
                "path": str(amount),
                "raw": [q, p],
                "cores": None if q < 0 else q / p,
            }
    raise RuntimeError(
        "CPU quota files are unavailable; inspect the container mount layout"
    )


def capacity(root="/sys/fs/cgroup", affinity=None):
    limits = quota(root)
    affinity = len(os.sched_getaffinity(0)) if affinity is None else affinity
    available = affinity if limits["cores"] is None else min(affinity, limits["cores"])
    return {"quota": limits, "affinity_cpus": affinity, "effective_cores": available}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--minimum", type=int, default=22)
    args = p.parse_args()
    result = capacity()
    print(json.dumps(result, indent=2), flush=True)
    if result["effective_cores"] < args.minimum:
        raise SystemExit(
            "allocated CPU capacity is below the matched study requirement"
        )


if __name__ == "__main__":
    main()
