#!/usr/bin/env python3
"""Reconcile what the Cartridge reserve has actually spent, from evidence.

The stage before this one may not claim a device until the reserve is
reconciled, and it may not treat an unknown debit as zero. Both halves matter:
the lane has twice planned against a residual that later evidence shrank, and
each time the error pointed the same way, because an unmeasured allocation
looks free.

Two things changed since the last attempt at this, and together they close the
gap. The provider now returns a creation timestamp for a terminated instance —
it did not before, which is why two allocations were recorded with a null
occupancy — and the session transcript carries the provider's own
"Successfully terminated" line with a timestamp beside it. Creation from the
provider, termination from the transcript, and the occupancy is a subtraction
rather than an estimate.

What that produced is unwelcome. The second acquisition for the source-utility
screen lived for two hours and twenty-five minutes while the screen itself ran
somewhere else, and was recorded as an unknown. It is the single largest debit
in the reserve, larger than the measurement it failed to perform.

Every figure here is a lower bound on consumption and therefore an upper bound
on what is left, because the local qualification and smoke time is still not
measured. It is listed, with zero explicitly refused as a value for it.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))

RESERVE_SECONDS = 14400.0
STAGE_CAP_SECONDS = 10800.0
BLOCKED = "CARTRIDGE_RESOURCE_BLOCKED_ACCOUNTING_UNRESOLVED"
INSUFFICIENT = "CARTRIDGE_RESOURCE_BLOCKED"

# Occupancy that is measured, with where each boundary came from. A row here is
# a subtraction between two recorded clocks, never a guess.
ALLOCATIONS = [
    {
        "id": "1fddf8254f334648b26618d8ca8859c2",
        "name": "cartridge-c1a",
        "purpose": "source-utility screen, first acquisition",
        "gpu": "A10-24GB",
        "usd_per_hour": 1.29,
        "created_utc": "2026-09-23T21:29:00Z",
        "created_evidence": "provider status created_at, read 2026-09-26",
        "terminated_utc": "2026-09-23T21:46:26Z",
        "terminated_evidence": (
            "session transcript, provider's 'Successfully terminated pod' line"
        ),
        "reached_ssh": False,
        "produced": "nothing; wedged in provisioning past the launcher's timeout",
    },
    {
        "id": "8a9bd6c4502d44e9bb969430e5692ba0",
        "name": "cartridge-c1b",
        "purpose": "source-utility screen, second acquisition",
        "gpu": "A10-24GB",
        "usd_per_hour": 1.29,
        "created_utc": "2026-09-23T22:10:01Z",
        "created_evidence": "provider status created_at, read 2026-09-26",
        "terminated_utc": "2026-09-24T00:34:52Z",
        "terminated_evidence": (
            "session transcript, provider's 'Successfully terminated pod' line"
        ),
        "reached_ssh": True,
        "produced": (
            "nothing measurable: inputs were staged, then the instance lost its "
            "address and the screen ran on local hardware instead, while this "
            "allocation stayed alive and billable"
        ),
    },
]

# Local hardware bills nothing and consumes the reserve anyway, by the lane's
# own rule.
LOCAL = [
    {
        "what": "source-utility screen on a local W7900",
        "seconds": 2914.6,
        "evidence": "cartridge-c1-20260923/C1_RESULT.json elapsed_s",
        "usd": 0.0,
    }
]

# Named, still unmeasured, and explicitly not zero.
UNRESOLVED = [
    "local smoke and chunked-prefill qualification before the screen",
    "setup, transfer and export time outside any stage timer",
    "any worker occupancy not visible from this host",
]


def _parse(ts):
    return dt.datetime.fromisoformat(ts.replace("Z", "+00:00"))


def occupancy(alloc):
    return (
        _parse(alloc["terminated_utc"]) - _parse(alloc["created_utc"])
    ).total_seconds()


def reconcile(reserve=RESERVE_SECONDS, stage_cap=STAGE_CAP_SECONDS):
    rented = []
    for a in ALLOCATIONS:
        s = occupancy(a)
        rented.append(
            {**a, "occupied_seconds": s, "usd": round(s / 3600 * a["usd_per_hour"], 2)}
        )
    rented_s = sum(r["occupied_seconds"] for r in rented)
    local_s = sum(l["seconds"] for l in LOCAL)
    debit = rented_s + local_s
    residual = reserve - debit
    cap = min(stage_cap, residual)
    return {
        "contract": "c2_accounting_v1",
        "reserve_seconds": reserve,
        "stage_cap_seconds": stage_cap,
        "rented_allocations": rented,
        "rented_seconds": rented_s,
        "rented_usd": round(sum(r["usd"] for r in rented), 2),
        "local_occupancy": LOCAL,
        "local_seconds": local_s,
        "recorded_debit_seconds": debit,
        "residual_seconds": residual,
        "residual_is_an_upper_bound": True,
        "unresolved_debits": UNRESOLVED,
        "unresolved_rule": (
            "these are named and unmeasured; none may be entered as zero, so "
            "the residual above can only shrink"
        ),
        "c2_cap_seconds": cap,
        "c2_cap_rule": "min(stage cap, reconciled residual)",
        "method": (
            "creation from the provider's status, which now returns a "
            "created_at for a terminated instance; termination from the "
            "provider's own confirmation line recorded in the session "
            "transcript. Each occupancy is a subtraction between two recorded "
            "clocks, not an estimate"
        ),
    }


def verdict_for(required_seconds, rec=None):
    """Whether a stage of this size can be run at all under the reserve."""
    rec = rec or reconcile()
    fits = required_seconds <= rec["c2_cap_seconds"]
    return {
        "required_seconds": required_seconds,
        "available_seconds": rec["c2_cap_seconds"],
        "fits": fits,
        "shortfall_seconds": None if fits else required_seconds - rec["c2_cap_seconds"],
        "verdict": None if fits else INSUFFICIENT,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--required-seconds",
        type=float,
        default=None,
        help="a stage's complete forecast, to test against the residual",
    )
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    rec = reconcile()
    if args.required_seconds is not None:
        rec["requested"] = verdict_for(args.required_seconds, rec)

    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(rec, f, indent=2, sort_keys=True)
            f.write("\n")

    print(f"reserve                     {rec['reserve_seconds']:9.1f} s")
    for r in rec["rented_allocations"]:
        print(
            f"  rented {r['name']:14s} {r['occupied_seconds']:9.1f} s  "
            f"${r['usd']:5.2f}  {r['id'][:12]}"
        )
    for l in rec["local_occupancy"]:
        print(f"  local  {'W7900 screen':14s} {l['seconds']:9.1f} s  $ 0.00")
    print(f"recorded debit              {rec['recorded_debit_seconds']:9.1f} s")
    print(f"RESIDUAL (upper bound)      {rec['residual_seconds']:9.1f} s")
    print(f"C2 cap = {rec['c2_cap_rule']}  {rec['c2_cap_seconds']:9.1f} s")
    print("still unmeasured, not zero:")
    for u in rec["unresolved_debits"]:
        print(f"  - {u}")
    if "requested" in rec:
        q = rec["requested"]
        if q["fits"]:
            print(f"\na {q['required_seconds']:.0f}s stage fits")
            return 0
        print(
            f"\n{q['verdict']}: a {q['required_seconds']:.0f}s stage is short by "
            f"{q['shortfall_seconds']:.0f}s"
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
