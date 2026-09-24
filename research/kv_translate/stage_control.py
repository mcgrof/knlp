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
