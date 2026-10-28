#!/usr/bin/env python3
"""Project page footprints and command counts for selected byte ranges.

For each selected unit (a layer, an expert slab, an embedding row) this
computes, from real absolute file offsets:

  W_required     the sum of the selected byte lengths
  footprint[P]   the number of distinct fetch granules of size P that the
                 ranges touch, times P, for P in 512 B, 4 KiB, 16 KiB,
                 64 KiB and 2 MiB. This is a union of granule indices per
                 file, not ceil(sum/P).
  commands       the number of NVMe read commands a direct read of each
                 range would need under a given maximum command size,
                 assuming the file is contiguous (an optimistic lower
                 bound; real extents can split further)

The granule sizes are labeled hypothetical fetch units. They are not a
claim about the SSD's indirection unit, NAND page or an HBF page size.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

GRANULES = [512, 4096, 16384, 65536, 2 << 20]


def load(path):
    rows = []
    with open(path) as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def footprint(ranges_by_file: dict, P: int) -> int:
    n = 0
    for _, rs in ranges_by_file.items():
        pages = set()
        for a, b in rs:
            pages.update(range(a // P, (b - 1) // P + 1))
        n += len(pages)
    return n * P


def commands(ranges, max_cmd: int) -> int:
    n = 0
    for a, b in ranges:
        # align to 4 KiB as a direct read would, then split by max_cmd
        s = a - a % 4096
        e = b + (-b % 4096)
        n += math.ceil((e - s) / max_cmd)
    return n


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--selection", required=True)
    ap.add_argument(
        "--max-cmd",
        type=int,
        default=128 * 1024,
        help="maximum NVMe command bytes on the measured device",
    )
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    sel = load(args.selection)
    units = defaultdict(list)
    for s in sel:
        units[(s["repo"], s["unit"])].append(s)
    out = {"max_cmd_bytes": args.max_cmd, "granules": GRANULES, "units": []}
    for (repo, unit), rows in sorted(units.items()):
        by_file = defaultdict(list)
        for r in rows:
            by_file[r["file"]].append((r["abs_start"], r["abs_end"]))
        allr = [x for rs in by_file.values() for x in rs]
        W = sum(b - a for a, b in allr)
        fp = {str(P): footprint(by_file, P) for P in GRANULES}
        rec = {
            "repo": repo,
            "unit": unit,
            "n_ranges": len(allr),
            "n_files": len(by_file),
            "W_required": W,
            "footprint": fp,
            "footprint_ratio": {k: (v / W if W else None) for k, v in fp.items()},
            "direct_commands_contiguous": commands(allr, args.max_cmd),
            "kinds": sorted({r["kind"] for r in rows}),
            "dtypes": sorted({r["dtype"] for r in rows}),
        }
        # per-expert and per-kind breakdown
        bykind = defaultdict(int)
        for r in rows:
            bykind[r["kind"]] += r["nbytes"]
        rec["bytes_by_kind"] = dict(bykind)
        experts = defaultdict(int)
        for r in rows:
            if r.get("expert") is not None:
                experts[r["expert"]] += r["nbytes"]
        if experts:
            rec["bytes_per_expert"] = dict(sorted(experts.items()))
        out["units"].append(rec)
        print(
            f"{repo} {unit}: W={W} B, 4K footprint x{rec['footprint_ratio']['4096']:.4f}, 2M footprint x{rec['footprint_ratio'][str(2 << 20)]:.3f}, cmds>={rec['direct_commands_contiguous']}"
        )
    Path(args.out).write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
