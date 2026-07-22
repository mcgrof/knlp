#!/usr/bin/env python3
"""Summarize the HBM ledger: post-prefill vs chunked peak, OOM cliffs,
and the analytic-vs-measured fragmentation gap. Emits a markdown table."""
import json, sys
from collections import defaultdict

path = sys.argv[1] if len(sys.argv) > 1 else "hbm_ledger.json"
rows = json.load(open(path))
by = {(r["schedule"], r["T"], r["c"], r["C"]): r for r in rows}

Ts = sorted({r["T"] for r in rows})
cs = sorted({r["c"] for r in rows})
Cs = sorted({r["C"] for r in rows})


def cell(sch, T, c, C):
    r = by.get((sch, T, c, C), {})
    if r.get("oom"):
        return "OOM"
    v = r.get("measured_peak_gib")
    return f"{v:.2f}" if isinstance(v, (int, float)) else "?"


print("# HBM ledger — measured peak HBM (GiB), W7900 48 GB\n")
print("Peak HBM (base weights + source KV + compact KV) per schedule. "
      "Both schedules end with the same compact KV; the gap is pure source "
      "residency. OOM = exceeded the 48 GB card (a valid result).\n")
for c in cs:
    print(f"\n## compression {c}x\n")
    print("| T | C | post-prefill-replace | chunked-streaming | "
          "reduction | analytic(ppr) |")
    print("|---:|---:|---:|---:|---:|---:|")
    for T in Ts:
        for C in Cs:
            ppr = by.get(("post_prefill_replace", T, c, C), {})
            chk = by.get(("chunked_streaming", T, c, C), {})
            pv = cell("post_prefill_replace", T, c, C)
            cv = cell("chunked_streaming", T, c, C)
            red = ""
            if (isinstance(ppr.get("measured_peak_gib"), (int, float))
                    and isinstance(chk.get("measured_peak_gib"), (int, float))
                    and chk["measured_peak_gib"] > 0):
                red = f"{ppr['measured_peak_gib']/chk['measured_peak_gib']:.2f}x"
            an = ppr.get("analytic_peak_gib", "")
            Tk = f"{T//1024}K"
            print(f"| {Tk} | {C} | {pv} | {cv} | {red} | {an} |")

# fragmentation / analytic accuracy on non-OOM post-prefill cells
gaps = []
for r in rows:
    if (not r.get("oom") and isinstance(r.get("measured_peak_gib"), (int, float))
            and isinstance(r.get("analytic_peak_gib"), (int, float))):
        gaps.append(r["measured_peak_gib"] - r["analytic_peak_gib"])
if gaps:
    print(f"\n## Analytic vs measured\n\n"
          f"Across {len(gaps)} non-OOM cells the measured peak exceeds the "
          f"analytic model by mean {sum(gaps)/len(gaps):+.3f} GiB "
          f"(min {min(gaps):+.3f}, max {max(gaps):+.3f}) — allocator "
          f"fragmentation / reserved-but-unused overhead.")

# concurrency cliff
print("\n## Concurrency / OOM cliff\n")
for c in cs:
    for T in Ts:
        for sch in ["post_prefill_replace", "chunked_streaming"]:
            firstoom = None
            for C in Cs:
                if by.get((sch, T, c, C), {}).get("oom"):
                    firstoom = C
                    break
            if firstoom is not None:
                print(f"- {sch} T={T//1024}K c={c}x: OOM at C={firstoom}")
