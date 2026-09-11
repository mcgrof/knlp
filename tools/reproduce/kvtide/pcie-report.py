#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Summarize recorded KVTide physical-PCIe repetitions."""

import argparse
import csv
import statistics
from collections import defaultdict
from pathlib import Path


def load_rows(path: Path):
    rows = []
    with path.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            if row["rep"].startswith("warmup-") or row["iops"] == "ERR":
                continue
            row["iops"] = float(row["iops"])
            row["MiBps"] = float(row["MiBps"])
            row["system_cpu_cores"] = float(row["system_cpu_cores"])
            rows.append(row)
    return rows


def summarize(rows):
    grouped = defaultdict(list)
    for row in rows:
        key = (
            row["profile"],
            int(row["iosize"]),
            int(row["qd"]),
            int(row["threads"]),
            row["arm"],
        )
        grouped[key].append(row)

    summaries = {}
    for key, group in grouped.items():
        summaries[key] = {
            "iops": statistics.median(row["iops"] for row in group),
            "mibps": statistics.median(row["MiBps"] for row in group),
            "cores": statistics.median(row["system_cpu_cores"] for row in group),
            "reps": len(group),
        }
    return summaries


def render(summaries):
    lines = [
        "KVTide physical-PCIe summary",
        "",
        "All rows are synthetic read stress tests. A ratio above one means",
        "premap beat SPDK in that measured cell; it is not a general claim.",
        "The latest SPDK arm uses spdk_nvme_perf while premap uses",
        "uring_nvm_perf, so their ratio compares complete software paths.",
        "",
        "profile  bytes     qd  thr  arm          median IOPS  median MiB/s  CPU cores  reps",
    ]
    for key in sorted(summaries):
        profile, size, qd, threads, arm = key
        value = summaries[key]
        lines.append(
            f"{profile:<8} {size:>8} {qd:>4} {threads:>4}  {arm:<12} "
            f"{value['iops']:>11.2f} {value['mibps']:>13.2f} "
            f"{value['cores']:>10.3f} {value['reps']:>5}"
        )

    ratios = []
    cells = {(key[0], key[1], key[2], key[3]) for key in summaries}
    for cell in sorted(cells):
        premap = summaries.get((*cell, "premap"))
        spdk = summaries.get((*cell, "spdk"))
        if not premap or not spdk or not spdk["iops"]:
            continue
        ratios.append((*cell, premap["iops"] / spdk["iops"]))
    if ratios:
        lines.extend(
            [
                "",
                "Premap / SPDK median IOPS",
                "",
                "profile  bytes     qd  thr  ratio  measured outcome",
            ]
        )
        for profile, size, qd, threads, ratio in ratios:
            outcome = "premap higher" if ratio > 1.0 else "SPDK higher"
            lines.append(
                f"{profile:<8} {size:>8} {qd:>4} {threads:>4} "
                f"{ratio:>6.2f}x  {outcome}"
            )
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("csv", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    report = render(summarize(load_rows(args.csv)))
    if args.output:
        args.output.write_text(report, encoding="utf-8")
    print(report, end="")


if __name__ == "__main__":
    main()
