#!/usr/bin/env python3
"""The run plan for a receiver-initializer comparison, and its failure rules.

Every rule here exists because the pilot hit the failure it prevents. The
pilot's own driver was a shell script that skipped a cell when a file existed,
trusted a trainer to write the checkpoints it was asked for, treated a lost
worker as a released allocation, and reported a verdict from whichever
checkpoint happened to be last. None of those are wrong in an interesting way;
they are wrong in the ordinary way that only shows up once a real run breaks.

The matrix is declared, not discovered. Sixteen cells: one no-cartridge floor
plus three initializers at five checkpoints, twenty questions each, three
hundred and twenty answers. The pilot planned that and delivered three hundred
and fourteen, and the gap was invisible until a reviewer counted, because
nothing in the driver knew how many answers it was supposed to produce.

A cell counts as done only when its receipt says so AND its rows carry exactly
the expected question identities under the expected content hash. The pilot
skipped on file existence, which is how a fourteen-row cell and a twenty-row
cell became indistinguishable to the thing deciding what to re-run.

Resuming a training arm needs the optimizer, the schedule, the generator state
and the data position. A key-value checkpoint alone is a state you can evaluate
and not a state you can continue, and treating one as the other silently
restarts the schedule.

An allocation is released when the provider says it is gone, not when it stops
answering. Those are different observations and the pilot conflated them.

Nothing here runs a model. It decides what should run, what may be skipped, and
what must be refused.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile

FLOOR_ARM = "receiver_no_cartridge"
INITIALIZER_ARMS = (
    "native_first_p",
    "mapped_learned_source",
    "mapped_fresh_first_p_source",
)
CHECKPOINTS = (0, 25, 50, 100, 200)
QUESTIONS_PER_CELL = 20

UNKNOWN = "UNKNOWN"
RESUME_STATE_KEYS = ("optimizer", "scheduler", "rng", "data_position")


# --------------------------------------------------------------------------
# the matrix
# --------------------------------------------------------------------------


def cells():
    """Every cell the comparison owes, declared up front.

    The floor has no checkpoint axis: it is the receiver with no trainable
    prefix, which does not train and so cannot have one. Everything else is an
    arm crossed with the checkpoint grid, including zero updates -- the pilot
    lost that row on all three arms because it asked a trainer to save it and
    the trainer silently declined, and nobody noticed until the run was over.
    """
    out = [{"arm": FLOOR_ARM, "checkpoint": None, "id": FLOOR_ARM}]
    for arm in INITIALIZER_ARMS:
        for c in CHECKPOINTS:
            out.append({"arm": arm, "checkpoint": c, "id": f"{arm}.step{c}"})
    return out


def expected_totals():
    n = len(cells())
    return {"cells": n, "questions": n * QUESTIONS_PER_CELL}


def initializer_cell_is_not_a_trainer_output(cell):
    """Zero updates is the initializer itself, so never wait for a trainer.

    The pilot passed SAVE_AT=0 to a trainer that ignores it, then read the
    empty result as 'that checkpoint does not exist'. The cache at zero updates
    is the file the initializer factory already wrote; it needs scoring, not
    producing.
    """
    return cell["checkpoint"] == 0


# --------------------------------------------------------------------------
# what counts as a finished cell
# --------------------------------------------------------------------------


def _canonical(obj):
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def content_hash(rows):
    """Identity of a cell's answers, independent of file layout."""
    payload = [
        {
            "question_id": r["question_id"],
            "generated": r.get("generated", ""),
            "correct": int(r.get("correct", 0)),
        }
        for r in sorted(rows, key=lambda r: r["question_id"])
    ]
    return hashlib.sha256(_canonical(payload).encode("utf-8")).hexdigest()


def cell_status(rows, receipt, expected_ids):
    """ "complete", "partial", "absent" or "invalid", with the reason.

    Deliberately strict about two things the pilot got wrong. A cell is not
    complete because a file exists; it is complete because the receipt says the
    cell finished and the rows carry exactly the expected identities. And a
    duplicate identity is invalid rather than merely surplus, because twenty
    rows containing nineteen distinct questions looks complete to anything that
    counts lines.
    """
    if rows is None:
        return {"status": "absent", "reason": "no rows"}
    ids = [r["question_id"] for r in rows]
    uniq = set(ids)
    if len(uniq) != len(ids):
        dup = sorted({i for i in ids if ids.count(i) > 1})
        return {"status": "invalid", "reason": f"duplicate question ids: {dup}"}
    unexpected = sorted(uniq - set(expected_ids))
    if unexpected:
        return {"status": "invalid", "reason": f"unexpected question ids: {unexpected}"}
    missing = sorted(set(expected_ids) - uniq)
    if missing:
        return {
            "status": "partial",
            "reason": f"{len(uniq)} of {len(expected_ids)} answers",
            "missing": missing,
        }
    if receipt is None or not receipt.get("completed"):
        return {
            "status": "partial",
            "reason": "rows are all present but no completion receipt; the "
            "writer may have died between the last row and the receipt",
        }
    return {"status": "complete", "reason": "rows and receipt agree"}


def may_skip(status, receipt, execution_identity):
    """A finished cell may be skipped only under the identity that produced it.

    Resuming someone else's half-finished cell silently mixes two software
    stacks inside one measurement. The pilot never checked, because it never
    recorded an identity to check against.
    """
    if status.get("status") != "complete":
        return False, f"not complete: {status.get('reason')}"
    got = (receipt or {}).get("execution_identity")
    if got != execution_identity:
        return False, (
            f"completed under a different execution identity ({got!r}, now "
            f"{execution_identity!r}); a cell finished by another stack is "
            "evidence, but it is not this run's evidence"
        )
    return True, "complete under this identity"


# --------------------------------------------------------------------------
# durability
# --------------------------------------------------------------------------


def atomic_write_json(path, obj):
    """Write through a temp file and rename, so a kill cannot truncate.

    The pilot flushed rows per question, which saved most of a cell when the
    instance vanished; receipts were written whole and could in principle have
    been caught mid-write. Rename is atomic on the same filesystem, so a reader
    sees the old file or the new one and never half of either.
    """
    d = os.path.dirname(os.path.abspath(path)) or "."
    os.makedirs(d, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=d, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(obj, f, indent=2, sort_keys=True)
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


def read_rows_tolerating_a_killed_writer(path):
    """Whole rows only, and say how many were lost to a partial line.

    A writer killed mid-row leaves a final line that is not JSON. Discarding it
    silently is right; not saying so is not.
    """
    if not os.path.exists(path):
        return None, {"truncated_lines": 0}
    rows, bad = [], 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                bad += 1
    return rows, {"truncated_lines": bad}


# --------------------------------------------------------------------------
# training resume
# --------------------------------------------------------------------------


def training_resume_is_possible(state):
    """A key-value checkpoint is a state you can score, not one you can continue.

    Restarting from a bare cache resets the optimizer moments, the schedule
    position and the data order, so the resumed run is a different experiment
    wearing the same name. Refuse instead.
    """
    if not state:
        return False, ["no resume state at all; only a cache was saved"]
    missing = [k for k in RESUME_STATE_KEYS if k not in state or state[k] is None]
    if missing:
        return False, missing
    return True, []


# --------------------------------------------------------------------------
# the things that must be refused
# --------------------------------------------------------------------------


class Refused(Exception):
    """Raised where continuing would produce a number nobody should trust."""


def require_unchanged_target(frozen, current):
    """The bar and the questions it was frozen against may not move mid-run."""
    for k in ("question_set_sha256", "threshold", "floor_correct", "native200_correct"):
        if frozen.get(k) != current.get(k):
            raise Refused(
                f"the frozen target changed during the run: {k} was "
                f"{frozen.get(k)!r}, now {current.get(k)!r}"
            )
    return True


def require_unchanged_model(frozen, current):
    for k in ("model", "revision", "tokenizer_sha256"):
        if frozen.get(k) != current.get(k):
            raise Refused(
                f"the model identity changed during the run: {k} was "
                f"{frozen.get(k)!r}, now {current.get(k)!r}"
            )
    return True


def check_subprocess(result, what):
    """A stage that failed must stop the run, not leave a hole in the matrix.

    The pilot's map step failed and the shell carried on, so both candidate
    arms then failed on a missing file and the run reported ALL DONE having
    produced nothing. Every later stage inherited a fault it could not see.
    """
    code = result if isinstance(result, int) else getattr(result, "returncode", 1)
    if code != 0:
        raise Refused(
            f"{what} exited {code}; later stages depend on it and would "
            "otherwise fail one by one against a missing input while the run "
            "reported success"
        )
    return True


# --------------------------------------------------------------------------
# release
# --------------------------------------------------------------------------


def release_state(provider_says_present, worker_reachable, terminate_acknowledged):
    """Unreachable is not released, and only the provider can say otherwise.

    The pilot lost ssh, saw a status field it had already recorded as
    unreliable, and treated the allocation as gone. It happened to be gone. The
    inference was still invalid, and the same inference on a live instance
    bills until somebody notices.
    """
    if not provider_says_present:
        return {"released": True, "evidence": "provider no longer lists the allocation"}
    if terminate_acknowledged:
        return {
            "released": False,
            "evidence": "terminate acknowledged but the provider still lists it",
            "action": "poll until the listing drops it",
        }
    return {
        "released": False,
        "evidence": (
            "worker unreachable" if not worker_reachable else "worker reachable"
        ),
        "action": "issue an explicit terminate and confirm against the listing; "
        "losing contact is not releasing an allocation",
    }


def deadline_allows(now_s, deadline_s, export_reserve_s, estimated_stage_s):
    """Refuse a stage that would eat the time reserved for getting results out.

    Finishing a cell and losing it is worse than not starting the cell.
    """
    remaining = deadline_s - now_s
    if remaining <= export_reserve_s:
        return False, f"only {remaining:.0f}s left, all of it reserved for export"
    if estimated_stage_s > remaining - export_reserve_s:
        return False, (
            f"stage needs {estimated_stage_s:.0f}s but only "
            f"{remaining - export_reserve_s:.0f}s is available before the "
            "export reserve"
        )
    return True, "fits with the export reserve intact"


# --------------------------------------------------------------------------
# reporting
# --------------------------------------------------------------------------


def earliest_passing_checkpoint(arm_scores, threshold, complete_checkpoints):
    """Earliest pass, or an explicit unknown when an earlier cell is missing.

    Reporting the earliest COMPLETE pass as if it were the earliest pass is
    what let the pilot state a cost comparison it had not measured: the fresh
    arm's zero-update cell was fourteen answers, so whether it already passed
    there is unresolved and may stay that way.
    """
    passing = [
        c
        for c in sorted(complete_checkpoints)
        if arm_scores.get(c, {}).get("correct", -1) >= threshold
    ]
    earliest_complete = passing[0] if passing else None
    incomplete_before = [
        c
        for c in sorted(arm_scores)
        if c not in complete_checkpoints
        and (earliest_complete is None or c < earliest_complete)
    ]
    return {
        "earliest_complete_passing": earliest_complete,
        "incomplete_checkpoints_before_it": incomplete_before,
        "earliest_passing": (earliest_complete if not incomplete_before else UNKNOWN),
        "why": (
            "resolved"
            if not incomplete_before
            else f"unresolved: checkpoints {incomplete_before} are incomplete and "
            "could have passed earlier"
        ),
    }


def cost_to_quality(arm_earliest, native_earliest, per_update_seconds=None):
    """Update counts are not costs, and saying so is the whole point.

    The pilot reported four times the updates as four times the cost. Updates
    are one term; building the initializer, fitting and applying a map, and
    evaluating are others, and none of them were timed per recipe.
    """
    if arm_earliest == UNKNOWN or native_earliest == UNKNOWN:
        return {
            "update_ratio": UNKNOWN,
            "cost_ratio": UNKNOWN,
            "why": "an earliest passing checkpoint is unresolved",
        }
    ratio = None
    if arm_earliest and native_earliest:
        ratio = arm_earliest / native_earliest
    return {
        "update_ratio": ratio,
        "cost_ratio": UNKNOWN if per_update_seconds is None else None,
        "why": (
            "update count only; initializer construction, map fit and apply, "
            "and evaluation are not included and were never timed per recipe"
            if per_update_seconds is None
            else "measured"
        ),
    }
