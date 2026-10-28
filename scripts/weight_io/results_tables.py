#!/usr/bin/env python3
"""Render the Stage B result tables from the aggregated medians.

Reads metrics_medians.csv (from aggregate_runs.py) and the per-selection
ranges.json files of the AWQ slice runs, and prints Markdown tables:
layer arms, embedding rows, the chunk and queue-depth sweep, and the
AWQ neuron slices with their 4 KiB page-union footprint. Every number
is a median over the run's trials with the min-max spread where it
matters; nothing is re-measured here.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

PAGE = 4096


def f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def mib(x):
    return f"{x / 2**20:,.1f}"


def page_union(ranges, page=PAGE):
    pages = set()
    for s, n in ranges:
        for p in range(s // page, (s + n - 1) // page + 1):
            pages.add(p)
    return len(pages) * page


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", required=True)
    ap.add_argument("--medians", default=None)
    args = ap.parse_args()
    root = Path(args.root)
    med = list(csv.DictReader(open(args.medians or root / "metrics_medians.csv")))
    by = {r["run"]: r for r in med}

    def lat(r):
        p50, p95 = f(r["lat_p50_us_med"]), f(r["lat_p95_us_med"])
        if p50 is None:
            return "-"
        return f"{p50:,.0f} / {p95:,.0f}"

    def spread(r):
        return f"{f(r['elapsed_ms_med']):,.1f} [{f(r['elapsed_ms_min']):,.1f}-{f(r['elapsed_ms_max']):,.1f}]"

    print("### Layer arms (one decoder layer, all tensors, QD 1)\n")
    print(
        "| Model | Unit | Arm | Cache | W (MiB) | F (MiB) | D (MiB) | Commands | D/W | Elapsed ms, median [min-max] | MiB/s | Device p50 / p95 us | n |"
    )
    print("|---|---|---|---|---:|---:|---:|---:|---:|---|---:|---|---:|")
    for r in sorted(
        (r for r in med if r["group"] == "layers"),
        key=lambda r: (r["repo"], int(r["unit"][5:]), r["arm"], r["cache"]),
    ):
        print(
            f"| {r['repo'].split('/')[1]} | {r['unit']} | {r['arm']} | {r['cache']} | {mib(f(r['W_required']))} | {mib(f(r['F_bytes_med']))} | {mib(f(r['D_target_bytes_med']))} | {f(r['D_target_cmds_med']):,.0f} | {f(r['D_over_W_med']):.3f} | {spread(r)} | {f(r['target_bw_MiBps_med']) or 0:,.0f} | {lat(r)} | {r['n_trials']} |"
        )

    print("\n### Embedding rows (2,048 B useful each, QD 1)\n")
    print(
        "| Row | Arm | Cache | F (B) | D (B) | Commands | D/W | Elapsed ms, median [min-max] | Device p50 us | n |"
    )
    print("|---|---|---|---:|---:|---:|---:|---|---:|---:|")
    for r in sorted(
        (r for r in med if r["group"] == "rows"),
        key=lambda r: (int(r["unit"].split("row")[1]), r["arm"], r["cache"]),
    ):
        p50 = f(r["lat_p50_us_med"])
        print(
            f"| {r['unit']} | {r['arm']} | {r['cache']} | {f(r['F_bytes_med']):,.0f} | {f(r['D_target_bytes_med']):,.0f} | {f(r['D_target_cmds_med']):,.0f} | {f(r['D_over_W_med']):.0f} | {spread(r)} | {p50 if p50 is None else format(p50, ',.0f')} | {r['n_trials']} |"
        )

    for grp, title in (
        ("chunks", "Chunk and queue-depth sweep (file-aligned chunks, O_DIRECT, cold)"),
        (
            "chunks-unaligned",
            "Chunk sweep, retired range-relative cut (one padded page per chunk)",
        ),
    ):
        rows = [r for r in med if r["group"] == grp]
        if not rows:
            continue
        print(f"\n### {title}\n")
        qds = sorted({int(r["qd"]) for r in rows})
        print(
            "| Chunk | D/W | Commands | "
            + " | ".join(f"QD {q} MiB/s (ms)" for q in qds)
            + " | p50 us at QD 1 / QD max |"
        )
        print("|---:|---:|---:|" + "---:|" * len(qds) + "---|")
        for c in sorted({int(r["chunk"]) for r in rows}):
            cells = []
            for q in qds:
                r = next(
                    (r for r in rows if int(r["chunk"]) == c and int(r["qd"]) == q),
                    None,
                )
                cells.append(
                    f"{f(r['target_bw_MiBps_med']):,.0f} ({f(r['elapsed_ms_med']):,.0f})"
                    if r
                    else "-"
                )
            r1 = next(
                (r for r in rows if int(r["chunk"]) == c and int(r["qd"]) == qds[0]),
                None,
            )
            rq = next(
                (r for r in rows if int(r["chunk"]) == c and int(r["qd"]) == qds[-1]),
                None,
            )
            label = f"{c // 1024} KiB" if c < 2**20 else f"{c // 2**20} MiB"
            print(
                f"| {label} | {f(r1['D_over_W_med']):.2f} | {f(r1['D_target_cmds_med']):,.0f} | "
                + " | ".join(cells)
                + f" | {f(r1['lat_p50_us_med']):,.0f} / {f(rq['lat_p50_us_med']):,.0f} |"
            )

    print("\n### AWQ neuron slices (layer 14 MLP, cold, QD 1)\n")
    print(
        "| Projection | Neurons | Pattern | W (B) | Page union 4 KiB (B) | Arm | F (B) | D (B) | Commands | D/W | Elapsed ms | Device p50 us | n |"
    )
    print("|---|---:|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|")
    awq = [r for r in med if r["group"] == "awq"]
    order = {"pread": 0, "direct": 1, "direct-merged": 2}
    for r in sorted(
        awq,
        key=lambda r: (
            r["run"].split("/")[1].split("-")[0],
            r["run"].split("/")[1].split("-")[2],
            int(r["run"].split("/")[1].split("-")[1][1:]),
            order.get(r["arm"], 9),
        ),
    ):
        sel = r["run"].split("/")[1]
        proj, n, pattern = sel.split("-")
        rj = root / "runs/awq" / sel / "ranges.json"
        pu = ""
        if rj.exists():
            doc = json.loads(rj.read_text())
            rr = doc["ranges"] if isinstance(doc, dict) else doc
            pu = f"{page_union([tuple(x) for x in rr]):,}"
        print(
            f"| {proj} | {n[1:]} | {pattern} | {f(r['W_required']):,.0f} | {pu} | {r['arm']} | {f(r['F_bytes_med']):,.0f} | {f(r['D_target_bytes_med']):,.0f} | {f(r['D_target_cmds_med']):,.0f} | {f(r['D_over_W_med']):,.1f} | {f(r['elapsed_ms_med']):,.1f} | {f(r['lat_p50_us_med']):,.0f} | {r['n_trials']} |"
        )

    intent = [
        r
        for r in med
        if r["group"] == "intent" and r["F_syscall_read_bytes_med"] not in ("", None)
    ]
    if intent:
        print(
            "\n### Observed intent (syscall and fault tracers on the reader, per trial window; faults per run)\n"
        )
        print(
            "| Run | Cache | W (MiB) | F declared (MiB) | Read calls | Read bytes (MiB) | Mapped (MiB) | Demand faults (MiB, run) | Page-cache fill (MiB, run) | D (MiB) | Commands | n |"
        )
        print("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
        for r in sorted(intent, key=lambda r: r["run"]):
            print(
                f"| {r['run'][7:]} | {r['cache']} | {mib(f(r['W_required']) or 0)} | {mib(f(r['F_bytes_med']) or 0)} | {f(r['F_syscall_read_calls_med']) or 0:,.0f} | {mib(f(r['F_syscall_read_bytes_med']) or 0)} | {mib(f(r['F_mmap_bytes_med']) or 0)} | {mib(f(r['fault_bytes_run']) or 0)} | {mib(f(r['pgcache_fill_bytes_run']) or 0)} | {mib(f(r['D_target_bytes_med']) or 0)} | {f(r['D_target_cmds_med']) or 0:,.0f} | {r['n_trials']} |"
            )

    gpu = [r for r in med if r["group"] == "gpu"]
    if gpu:
        print("\n### GPU consumption (W7900, synthetic activations unless native)\n")
        print(
            "| Run | Mode | W (MiB) | F (MiB) | D (MiB) | Commands | Elapsed ms, median [min-max] | read ms | copy ms | compute ms | n |"
        )
        print("|---|---|---:|---:|---:|---:|---|---:|---:|---:|---:|")
        for r in sorted(gpu, key=lambda r: r["run"]):
            print(
                f"| {r['run'][4:]} | {r['mode']} | {mib(f(r['W_required']) or 0)} | {mib(f(r['F_bytes_med']) or 0)} | {mib(f(r['D_target_bytes_med']) or 0)} | {f(r['D_target_cmds_med']) or 0:,.0f} | {spread(r)} | {f(r['read_ms_med']) or 0:,.1f} | {f(r['copy_ms_med']) or 0:,.1f} | {f(r['compute_ms_med']) or 0:,.2f} | {r['n_trials']} |"
            )


if __name__ == "__main__":
    main()
