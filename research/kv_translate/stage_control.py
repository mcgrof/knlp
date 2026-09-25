# SPDX-License-Identifier: GPL-2.0
"""Decide whether a stage may advance, and stop one that runs past its budget.

Two defects in the last run's driver, both of which let work proceed that
should not have.

It advanced from one stage to the next on the process exit status. The gate
exits zero for a valid negative verdict -- that is correct, since failing to
find a signal is not a failure to run -- so a stage whose economics had already
failed handed its successor an advancing edge it had not earned. Advancement
has to read the verdict, not the exit code.

And a stage cap was a message printed after the fact. A stage that ran past its
allocation was noted and kept running, which makes the cap a description of
what happened rather than a limit on it. Here the deadline is passed to the
stage and checked as it works, so the stage ends inside its budget with what it
has, and says so.
"""

from __future__ import annotations

import json
import os
import time

# Verdicts that permit the next stage to start. Everything else -- including
# every valid negative -- stops the queue.
ADVANCING = {
    "SCREEN_SIGNAL",
    "SCREEN_SIGNAL_UNIFORM",
    "SCREEN_SIGNAL_MAPPED_RANKING",
    "RECOVERY_DEVELOPMENT_SIGNAL",
    "SOURCE_UTILITY_ESTABLISHED",
    "HANDOFF_DEVELOPMENT_SIGNAL",
    "COLD_DEVELOPMENT_SIGNAL",
}

# Fields a stage must actually have written for its verdict to be readable.
# A receipt that records a verdict and not the evidence behind it is a claim,
# and a driver that advances on one is advancing on a claim.
REQUIRED_RECEIPT_FIELDS = {
    "qualification": (
        "passed",
        "envelope_checks",
        "exact_checks",
        "native_prefill_repeats_identically",
    ),
    "gate": ("verdict", "advances", "valid_evidence"),
}


class Deadline:
    """A budget a stage can consult, rather than one it is judged against after.

    ``remaining`` is what a caller checks before starting another unit of work;
    ``expired`` is the same question asked as a predicate. Neither raises: a
    stage that runs out should finish what it holds and report an incomplete
    result, which is a different and more useful outcome than a traceback.
    """

    def __init__(self, seconds, *, name="stage", reserve=0.0):
        self.name = name
        self.budget = float(seconds)
        # Work that must still happen after the last unit: writing rows,
        # exporting, releasing. A deadline that ends exactly at the cap leaves
        # no time to record what it did.
        self.reserve = float(reserve)
        self.start = time.time()

    def elapsed(self):
        return time.time() - self.start

    def remaining(self):
        return self.budget - self.reserve - self.elapsed()

    def expired(self):
        return self.remaining() <= 0.0

    def enough_for(self, seconds):
        """Is there room for a unit of work expected to take this long?"""
        return self.remaining() >= float(seconds)

    def record(self):
        return {
            "stage": self.name,
            "budget_s": self.budget,
            "reserve_s": self.reserve,
            "elapsed_s": round(self.elapsed(), 1),
            "remaining_s": round(self.remaining(), 1),
            "expired": self.expired(),
        }


def read_verdict(path):
    """The semantic outcome a gate wrote, or an explicit absence."""
    if not os.path.exists(path):
        return {"verdict": "ABSENT", "advances": False, "reason": f"no {path}"}
    try:
        d = json.load(open(path))
    except (json.JSONDecodeError, OSError) as e:
        return {"verdict": "UNREADABLE", "advances": False, "reason": str(e)}
    v = d.get("verdict")
    if v is None:
        return {"verdict": "NO_VERDICT_FIELD", "advances": False, "reason": path}
    # The gate's own advancement flag is authoritative when it is present; the
    # verdict set is the fallback, so a gate that stops writing the flag fails
    # closed rather than open.
    adv = d.get("advances")
    if adv is None:
        adv = v in ADVANCING
    return {"verdict": v, "advances": bool(adv), "source": path}


def read_qualification(path, required=REQUIRED_RECEIPT_FIELDS["qualification"]):
    """A qualification receipt, checked for the fields that make it one.

    A driver that reads only ``passed`` will advance on a receipt whose
    exactness checks were never run, because an absent check is not a failing
    one. Every field the contract names must be present, and every exact check
    inside must be true -- those admit no tolerance, so a false one is a
    failure whatever the envelope says.
    """
    if not os.path.exists(path):
        return {"passed": False, "reason": f"absent: {path}"}
    try:
        d = json.load(open(path))
    except (json.JSONDecodeError, OSError) as e:
        return {"passed": False, "reason": f"unreadable: {e}"}
    missing = [f for f in required if f not in d]
    if missing:
        return {
            "passed": False,
            "reason": f"receipt lacks {missing}",
            "fields_present": sorted(d),
        }
    exact = d.get("exact_checks") or {}
    failed_exact = sorted(k for k, v in exact.items() if not v)
    envelope = d.get("envelope_checks") or {}
    failed_env = sorted(
        k for k, v in envelope.items() if isinstance(v, dict) and not v.get("passed")
    )
    ok = bool(d.get("passed")) and not failed_exact and not failed_env
    return {
        "passed": ok,
        "declared_passed": bool(d.get("passed")),
        "failed_exact_checks": failed_exact,
        "failed_envelope_checks": failed_env,
        "native_prefill_repeats_identically": d.get(
            "native_prefill_repeats_identically"
        ),
        "source": path,
        "rule": (
            "a qualification advances only if it declares a pass AND every "
            "exact check is true AND every envelope check passed; an "
            "absent check is not a passing one"
        ),
    }


def may_advance(gate_paths):
    """Every named gate must have written an advancing verdict."""
    checks = [read_verdict(p) for p in gate_paths]
    blocking = [c for c in checks if not c["advances"]]
    return {
        "advances": not blocking,
        "checks": checks,
        "blocked_by": [c["verdict"] for c in blocking],
        "rule": (
            "advancement reads the verdict a gate wrote, never a process exit "
            "status: a gate exits zero for a valid negative, so an exit-status "
            "rule advances on exactly the results that should stop the queue"
        ),
    }


def inclusive_forecast(
    stage_seconds,
    *,
    setup,
    transfers,
    diagnosis,
    export,
    already_used,
    cap,
    occupancy_multiplier=None,
):
    """What an allocation costs, against what is actually left of the cap.

    The previous forecast counted setup and export and still understated,
    because it forecast stage time and compared it against a cap measured in
    occupied time. Reconciled against the filesystem record, the enablement
    stages so far occupied roughly nineteen hundred seconds per thousand
    seconds of stage work: provisioning, weight downloads, transfers,
    diagnosis between stages and the allocations that never produced an
    address are all real and none of them appear in a stage timer.

    So this takes the stage plan, adds the overheads by name, optionally
    applies a measured occupancy multiplier to the stage part, and subtracts
    what has already been used. A forecast that does not fit says so.
    """
    stages = float(sum(stage_seconds.values()))
    charged_stages = stages * float(occupancy_multiplier or 1.0)
    overhead = float(setup) + float(transfers) + float(diagnosis) + float(export)
    total = charged_stages + overhead
    remaining = float(cap) - float(already_used)
    out = {
        "stage_seconds": dict(stage_seconds),
        "stage_total_s": stages,
        "occupancy_multiplier": occupancy_multiplier,
        "stage_seconds_as_occupied": charged_stages,
        "overheads_s": {
            "setup": float(setup),
            "transfers": float(transfers),
            "diagnosis": float(diagnosis),
            "export": float(export),
        },
        "forecast_occupied_s": total,
        "cap_s": float(cap),
        "already_used_s": float(already_used),
        "remaining_s": remaining,
        "fits": bool(total <= remaining),
        "headroom_s": remaining - total,
        "rule": (
            "a cap measured in occupied time must be compared against a "
            "forecast in occupied time; summing stage timers understates "
            "it by everything that happens outside a stage"
        ),
    }
    if not out["fits"]:
        out["shortfall_s"] = total - remaining
    return out


def forecast(stage_seconds, *, setup=0.0, export=0.0, cap=None):
    """What the whole allocation costs, not just the stages inside it.

    The last run's accounting summed three stage durations and compared them
    with an allocation-time cap, which omits setup, failed starts, profiling,
    export and release -- everything the meter charges for. A forecast that
    only counts the stages understates what is being asked for.
    """
    stages = float(sum(stage_seconds.values()))
    total = stages + float(setup) + float(export)
    out = {
        "stage_seconds": dict(stage_seconds),
        "stage_total_s": stages,
        "setup_s": float(setup),
        "export_and_release_s": float(export),
        "total_occupied_s": total,
        "note": (
            "occupied allocation time, which is what is billed: stage work "
            "plus setup, export and release. Stage durations alone are not a "
            "forecast of an allocation."
        ),
    }
    if cap is not None:
        out["cap_s"] = float(cap)
        out["fits"] = total <= float(cap)
        out["headroom_s"] = float(cap) - total
    return out
