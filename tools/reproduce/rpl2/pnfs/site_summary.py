#!/usr/bin/env python3
"""Summarize a site run (site.sh) and set it beside the reference results.

  site_summary.py <out-dir>

Reads whichever steps exist under <out-dir> (doctor, fingerprint, screen,
sweep, readmodes, kernel) and writes <out-dir>/site-summary.json and
<out-dir>/site-summary.md. Every reference figure carries its scope: the
filesystem, object size, client count and host class it was measured on.
Compare like with like, and read a difference against that scope.
"""

import glob
import json
import os
import re
import statistics
import sys

# Measured in the one-host pNFS lab (Lattice MDS with RonDB, one NFSv3 data
# server, Linux NFSv4.2 client, 32-thread desktop host) and on local XFS.
REFERENCE = {
    "fingerprint": "pNFS lab, 8 MiB objects: after first touch, open per read "
    "sends OPEN, CLOSE and two GETATTRs to the MDS per object read (4 ops); "
    "held descriptors send none.",
    "screen_b1": "pNFS lab, local client, 1 x 8 MiB per request: held "
    "descriptors 0.205 ms faster per request, 22.4% [18.2, 26.9]. "
    "Remote client at 1 Gb/s: 0.602 ms, 0.8% (link-bound).",
    "screen_b16": "pNFS lab, local client, 16 x 8 MiB per request: 0.549 ms, "
    "4.5% [3.2, 6.2]; open per read stalled 13 of 600 requests (0.4-1.14 s).",
    "sweep_256k": "pNFS lab, 256 KiB buffered, 16-key requests, N clients each "
    "its own NFS client identity: open per read peaks near 44,527 reads/s at "
    "4 clients and falls to 37,926 at 16; held reaches 139,479-140,427 at "
    "8-16 clients with zero MDS operations.",
    "sweep_8m": "pNFS lab, 8 MiB O_DIRECT: open per read p99 0.86-3.3 s "
    "(CLOSE stalls) in every cell; one client 442 against 865 reads/s held.",
    "readmodes": "Local XFS, 16 hot 256 KiB objects in the page cache (4 MiB, "
    "fits L3), one connector, 16-key requests, 16 readers, held descriptors: pool 511k/446k/477k obj/s at 1/4/16 in flight, "
    "inline 176k/613k/287k, adaptive (target 16) 457k/284k/215k; CPU per "
    "object pool 14.1/36.9/36.1 us, inline 5.8/6.6/39.8 us. pNFS lab (Linux "
    "7.3-rc2 client, RWF_NOWAIT refused): all three modes 417k-505k obj/s "
    "at 1/4/16 in flight, 16 OPENs for 16 objects; a build without one open "
    "per file at a time stalled at 16 (NFS4ERR_DELAY).",
    "kernel": "8-vCPU virtual machine, loopback NFSv4.2 with read delegations, 6 runs "
    "per cell, base against patched: one shared credential 393,134 -> "
    "382,077 opens/s at 1 thread and 1,311,758 -> 1,307,902 at 8 (no gain); "
    "a credential per thread 1,908,312 -> 2,081,534 at 8 threads (+9.1%, "
    "wide range).",
}


def load(path):
    try:
        return json.load(open(path))
    except (OSError, ValueError):
        return None


def rpcs_per_object(rec, phase, objects):
    d = (rec.get("nfs_rpcs") or {}).get(phase)
    if d is None:
        return None
    if not objects:
        return {}
    return {op: round(v["calls"] / objects, 3) for op, v in sorted(d.items())}


def fingerprint(d):
    rows = {}
    for f in sorted(glob.glob(os.path.join(d, "held*.json"))):
        r = load(f)
        held = r["held"]
        rows[held] = {
            "first": rpcs_per_object(r, "first", r["hot"]),
            "repeat": rpcs_per_object(r, "repeat", r["complete"] * r["batch"]),
            "p50_ms": r["p50_ms"],
            "failures": r["failures"] + r["digest_mismatches"],
        }
    return rows


def sweep(d):
    rows = []
    for cell in sorted(
        glob.glob(os.path.join(d, "n*-held*")),
        key=lambda c: [int(x) for x in re.findall(r"\d+", os.path.basename(c))],
    ):
        n, held = map(int, re.search(r"n(\d+)-held(\d+)$", cell).groups())
        recs = [load(f) for f in sorted(glob.glob(os.path.join(cell, "c*.json")))]
        recs = [r for r in recs if r]
        if len(recs) != n:
            rows.append({"n": n, "held": held, "error": f"{len(recs)} of {n}"})
            continue
        objs = sum(r["complete"] * r["batch"] for r in recs)
        mount = next((r for r in recs if r.get("nfs_rpcs")), None)
        row = {
            "n": n,
            "held": held,
            "object_reads_per_s": round(objs / recs[0]["seconds"]),
            "p50_ms_median_client": statistics.median(r["p50_ms"] for r in recs),
            "p99_ms_worst_client": max(r["p99_ms"] for r in recs),
            "client_cpu_us_per_read": (
                round(1e6 * sum(r["cpu_s"] for r in recs) / objs, 2) if objs else None
            ),
            # Per-mount counters cover every reader on this host.
            "nfs_rpcs_per_read": (
                rpcs_per_object(mount, "read", objs) if mount else None
            ),
            "failures": sum(r["failures"] + r["digest_mismatches"] for r in recs),
        }
        before, after = (
            os.path.join(cell, f"mds-ops.{x}") for x in ("before", "after")
        )
        if os.path.exists(before) and os.path.getsize(before):

            def ops(path):
                out = {}
                for line in open(path):
                    k, v = line.rsplit(" ", 1)
                    out[re.search(r'op="([^"]+)"', k).group(1)] = float(v)
                return out

            b, a = ops(before), ops(after)
            row["mds_ops_per_1k_reads"] = {
                k: round(1000 * (a[k] - b.get(k, 0)) / objs, 1)
                for k in a
                if a[k] - b.get(k, 0)
            }
        rows.append(row)
    return rows


def readmodes(d):
    rows = []
    for f in sorted(glob.glob(os.path.join(d, "*-p*.json"))):
        r = load(f)
        objs = r["complete"] * r["batch"]
        rs = r.get("read_stats") or {}
        tried = rs.get("inline_hits", 0) + rs.get("inline_misses", 0)
        rows.append(
            {
                "mode": r["read_mode"],
                "in_flight": r["pipeline"],
                "objects_per_s": round(objs / r["wall_s"]) if r["wall_s"] else None,
                "cpu_us_per_object": (
                    round(1e6 * r["cpu_s"] / objs, 2) if objs else None
                ),
                "p50_ms": r["p50_ms"],
                "p99_ms": r["p99_ms"],
                "inline_share": (
                    round(rs.get("inline_hits", 0) / objs, 3) if objs else None
                ),
                "inline_misses": rs.get("inline_misses"),
                "inline_tried": tried,
                "inline_unsupported": rs.get("inline_unsupported"),
                "failures": r["failures"] + r["digest_mismatches"],
            }
        )
    rows.sort(
        key=lambda x: (x["in_flight"], ["pool", "inline", "adaptive"].index(x["mode"]))
    )
    return rows


def md_table(rows, cols):
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for r in rows:
        lines.append("| " + " | ".join(str(r.get(c, "")) for c in cols) + " |")
    return "\n".join(lines)


def main():
    out = sys.argv[1]
    res = {"reference": REFERENCE}
    md = ["# Site run summary", ""]
    doc = load(os.path.join(out, "doctor", "doctor.json"))
    if doc:
        res["doctor"] = doc
        md += [
            "## Filesystem",
            "",
            f"- type `{doc['fstype']}`, label `{doc.get('label') or '-'}`, data "
            f"servers `{doc.get('data_servers') or '-'}`, drives `{doc.get('drives') or '-'}`",
            f"- O_DIRECT {doc.get('o_direct')}; RWF_NOWAIT {doc.get('rwf_nowait')}",
            f"- NFS: {'; '.join(doc['nfs_header']) if doc.get('nfs_header') else 'not an NFS mount'}",
            "",
        ]
    fp = os.path.join(out, "fingerprint")
    if os.path.isdir(fp):
        res["fingerprint"] = fingerprint(fp)
        md += ["## RPCs per object (fingerprint)", "", "```"]
        md += [f"held {h}: {v}" for h, v in sorted(res["fingerprint"].items())]
        md += ["```", "", f"Reference: {REFERENCE['fingerprint']}", ""]
    sc = load(os.path.join(out, "screen", "screen-summary.json"))
    if sc:
        res["screen"] = sc
        md += ["## Paired screen, stock against held", ""]
        for shape, s in sc["shapes"].items():
            if "mean_diff_ms" in s:
                ci = s.get("boot95_rel")
                md.append(
                    f"- {shape}: held faster by {s['mean_diff_ms']:.3f} ms, "
                    f"{100 * s['mean_rel']:.1f}%"
                    + (f" [{100 * ci[0]:.1f}, {100 * ci[1]:.1f}]" if ci else "")
                )
        md += [
            "",
            f"Reference b1: {REFERENCE['screen_b1']}",
            f"Reference b16: {REFERENCE['screen_b16']}",
            "",
        ]
    sw = os.path.join(out, "sweep")
    if os.path.isdir(sw):
        res["sweep"] = sweep(sw)
        cols = ["n", "held", "object_reads_per_s", "p50_ms_median_client"]
        cols += ["p99_ms_worst_client", "client_cpu_us_per_read", "nfs_rpcs_per_read"]
        cols += ["mds_ops_per_1k_reads", "failures"]
        md += ["## Many readers on this host", "", md_table(res["sweep"], cols), ""]
        md += [f"Reference: {REFERENCE['sweep_256k']}", ""]
    rm = os.path.join(out, "readmodes")
    if os.path.isdir(rm):
        res["readmodes"] = readmodes(rm)
        cols = ["in_flight", "mode", "objects_per_s", "cpu_us_per_object", "p50_ms"]
        cols += ["p99_ms", "inline_share", "inline_unsupported", "failures"]
        md += ["## read_mode", "", md_table(res["readmodes"], cols), ""]
        md += [f"Reference: {REFERENCE['readmodes']}", ""]
    kr = load(os.path.join(out, "kernel", "kernel.json"))
    if kr or os.path.exists(os.path.join(out, "kernel", "kernel.log")):
        res["kernel"] = kr or {"log": "kernel/kernel.log"}
        md += [
            "## Kernel selftest",
            "",
            f"{res['kernel']}",
            "",
            REFERENCE["kernel"],
            "",
        ]
    json.dump(res, open(os.path.join(out, "site-summary.json"), "w"), indent=1)
    open(os.path.join(out, "site-summary.md"), "w").write("\n".join(md) + "\n")
    print("\n".join(md))
    bad = any(
        r.get("failures") or "error" in r
        for r in res.get("sweep", []) + res.get("readmodes", [])
    ) or any(v["failures"] for v in res.get("fingerprint", {}).values())
    sys.exit(1 if bad or (sc and sc.get("failed_cells")) else 0)


if __name__ == "__main__":
    main()
