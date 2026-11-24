#!/usr/bin/env python3
"""Tensor and expert read-unit sizes from safetensors headers.

Input: the per-tensor inventory (one JSON object per tensor with repo,
file, abs_start, abs_end, nbytes, dtype, layer_index, expert, kind).
Output (stdout, JSON): per checkpoint, power-of-two size buckets of
tensors by count and by bytes, the share of bytes in tensors whose data
starts off a 4 KiB boundary, and for MoE checkpoints the size of one
routed expert and the number of contiguous file extents it occupies.
"""

import collections, json, math, sys


def bucket(n):
    return 0 if n <= 0 else int(math.log2(n))


def main(path):
    R = collections.defaultdict(
        lambda: {
            "n": 0,
            "bytes": 0,
            "cnt": collections.Counter(),
            "byt": collections.Counter(),
            "unaligned_bytes": 0,
            "dtype_bytes": collections.Counter(),
            "experts": collections.defaultdict(list),
        }
    )
    for line in open(path):
        e = json.loads(line)
        r = R[e["repo"]]
        s = e["nbytes"]
        b = bucket(s)
        r["n"] += 1
        r["bytes"] += s
        r["cnt"][b] += 1
        r["byt"][b] += s
        r["dtype_bytes"][e["dtype"]] += s
        if e["start_mod_4k"]:
            r["unaligned_bytes"] += s
        if e["expert"] is not None:
            r["experts"][(e["layer_index"], e["expert"])].append(
                (e["file"], e["abs_start"], e["abs_end"])
            )
    out = {}
    for repo, r in R.items():
        o = {
            "tensors": r["n"],
            "bytes": r["bytes"],
            "count_by_log2": dict(sorted(r["cnt"].items())),
            "bytes_by_log2": dict(sorted(r["byt"].items())),
            "unaligned_byte_share": r["unaligned_bytes"] / r["bytes"],
            "dtype_bytes": dict(r["dtype_bytes"]),
        }
        if r["experts"]:
            sizes, extents, ebytes = collections.Counter(), collections.Counter(), 0
            for parts in r["experts"].values():
                parts.sort()
                n, last = 0, None
                for f, a, b in parts:
                    if last is None or last[0] != f or a != last[1]:
                        n += 1
                    last = (f, b)
                sz = sum(b - a for _, a, b in parts)
                ebytes += sz
                sizes[sz] += 1
                extents[n] += 1
            o["experts"] = {
                "units": len(r["experts"]),
                "byte_share": ebytes / r["bytes"],
                "unit_bytes": dict(sizes.most_common(3)),
                "extents_per_unit": dict(extents.most_common(3)),
            }
        out[repo] = o
    json.dump(out, sys.stdout, indent=1)


if __name__ == "__main__":
    main(sys.argv[1])
