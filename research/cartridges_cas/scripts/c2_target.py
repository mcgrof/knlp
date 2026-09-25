#!/usr/bin/env python3
"""Compute and freeze the pilot's success threshold, and make the order real.

The target is the receiver's no-cartridge floor plus ninety percent of what the
native initializer earns over that floor by two hundred updates. It has to be
fixed before any mapped arm is scored, because a threshold chosen after seeing
the candidate is not a threshold.

Saying that in a plan does not make it true. Discipline is not a mechanism, and
this program has already published a number that turned out to compare two
different artifacts, so the ordering here is enforced by the tool rather than
promised by the operator:

``freeze`` refuses to run if any mapped arm's rows or caches already exist in
the run directory, and refuses to overwrite an existing target. It reads the
two reference row files and nothing else, and records their hashes along with a
hash of the directory listing as it stood at that moment.

``check`` recomputes those hashes and fails if a reference moved after the
freeze.

And the runner refuses to evaluate any arm whose name begins with ``mapped_``
unless a target file exists, passes ``check``, and was frozen against the same
question set the runner is using. So the mapped arms cannot start before a
target exists, and a target cannot be quietly recomputed after they do.

The arithmetic is integer throughout. ``ceil(F + 0.90*(N-F))`` is exactly
``ceil((F + 9N)/10)``, and doing it in floating point puts an exact case such
as F=0, N=10 -- which should be nine -- one ulp above nine and rounds it to
ten. An off-by-one in the bar is an off-by-one in the verdict.

The spec also forbids a parser or termination regression without saying against
what. Leaving that to be decided later means deciding it once the numbers are
visible, so it is named here and written into the frozen file: an arm's bad
count is the number of its rows that either failed to resolve to an option or
ran into the generation cap, and each mapped checkpoint is compared against the
native arm at the same checkpoint.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))

from research.cartridges_cas.scripts.score_longhealth import UNRESOLVED  # noqa: E402

TARGET_FILENAME = "TARGET.json"
FLOOR_ROWS = "rows.floor.jsonl"
NATIVE200_ROWS = "rows.native_first_p.step200.jsonl"
MAPPED_PREFIX = "mapped_"
FRACTION_NUMERATOR = 9
FRACTION_DENOMINATOR = 10
NO_HEADROOM = "RECEIVER_INITIALIZER_NO_INFORMATIVE_REFERENCE"
BAD_RULE = (
    "a row is bad if its outcome did not resolve to an option "
    f"({', '.join(UNRESOLVED)}) or it hit the generation cap; each mapped "
    "checkpoint is compared against the native arm at the same checkpoint, and "
    "the floor's count is recorded alongside for context"
)


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def threshold_from(floor_correct, native200_correct):
    """The bar, in whole answers, or None when the references leave no room.

    ``-(-x // d)`` is ceiling division on integers. The identity being used is
    ``ceil(F + 0.90*(N-F)) == ceil((F + 9N)/10)``; it holds for every pair in
    range, and unlike the float form it cannot round an exact value up.
    """
    if native200_correct <= floor_correct:
        return None
    total = floor_correct + FRACTION_NUMERATOR * native200_correct
    return -(-total // FRACTION_DENOMINATOR)


def read_rows(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    if not rows:
        raise SystemExit(f"INVALID: {path} has no rows")
    return rows


def count_correct(rows):
    return sum(int(r["correct"]) for r in rows)


def count_bad(rows):
    return sum(
        1 for r in rows if r.get("outcome") in UNRESOLVED or bool(r.get("cap_hit"))
    )


def listing_sha256(run_dir):
    names = sorted(os.listdir(run_dir))
    blob = json.dumps(names, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest(), names


def mapped_artifacts(run_dir):
    """Anything that would mean a mapped arm has already produced something."""
    found = []
    for name in sorted(os.listdir(run_dir)):
        if name.startswith(f"rows.{MAPPED_PREFIX}") and name.endswith(".jsonl"):
            found.append(name)
    carts = os.path.join(run_dir, "carts")
    if os.path.isdir(carts):
        found.extend(
            f"carts/{n}"
            for n in sorted(os.listdir(carts))
            if n.startswith(MAPPED_PREFIX)
        )
    return found


def knlp_commit():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=os.path.join(os.path.dirname(__file__), "..", "..", ".."),
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except Exception:
        return None


def do_freeze(run_dir, questions_path, force_question_sha=""):
    target_path = os.path.join(run_dir, TARGET_FILENAME)
    if os.path.exists(target_path):
        raise SystemExit(
            f"REFUSING: {target_path} already exists. A target is frozen once; "
            "recomputing it after the fact is the thing this file prevents."
        )

    already = mapped_artifacts(run_dir)
    if already:
        raise SystemExit(
            "REFUSING: mapped arms have already produced artifacts "
            f"({', '.join(already)}). A target frozen now would be chosen with "
            "the candidate's outcome already on disk, which is not a freeze."
        )

    with open(questions_path, encoding="utf-8") as f:
        questions = json.load(f)
    qsha = questions["question_set_sha256"]
    if force_question_sha and qsha != force_question_sha:
        raise SystemExit(
            f"INVALID: question set hashes to {qsha}, expected {force_question_sha}"
        )
    n_questions = len(questions["questions"])

    floor_path = os.path.join(run_dir, FLOOR_ROWS)
    native_path = os.path.join(run_dir, NATIVE200_ROWS)
    for p in (floor_path, native_path):
        if not os.path.exists(p):
            raise SystemExit(
                f"BLOCKED: {p} does not exist. Both references must be measured "
                "before the target can be frozen."
            )

    floor_rows = read_rows(floor_path)
    native_rows = read_rows(native_path)
    for name, rows in (("floor", floor_rows), ("native200", native_rows)):
        if len(rows) != n_questions:
            raise SystemExit(
                f"INVALID: {name} has {len(rows)} rows for {n_questions} questions"
            )
        ids = {r["question_id"] for r in rows}
        expected = {q["question_id"] for q in questions["questions"]}
        if ids != expected:
            raise SystemExit(
                f"INVALID: {name} rows do not cover the frozen question set"
            )

    floor_correct = count_correct(floor_rows)
    native_correct = count_correct(native_rows)
    threshold = threshold_from(floor_correct, native_correct)
    lsha, names = listing_sha256(run_dir)

    target = {
        "contract": "c2_frozen_target_v1",
        "frozen_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "knlp_commit": knlp_commit(),
        "question_set_sha256": qsha,
        "questions": n_questions,
        "floor_correct": floor_correct,
        "native200_correct": native_correct,
        "fraction_of_headroom": f"{FRACTION_NUMERATOR}/{FRACTION_DENOMINATOR}",
        "threshold": threshold,
        "threshold_rule": (
            "ceil(floor + 0.90*(native200 - floor)), computed as "
            "ceil((floor + 9*native200)/10) in integers so an exact value is "
            "never rounded up"
        ),
        "verdict_if_no_headroom": NO_HEADROOM if threshold is None else None,
        "floor_bad": count_bad(floor_rows),
        "native200_bad": count_bad(native_rows),
        "bad_rule": BAD_RULE,
        "inputs": {
            FLOOR_ROWS: sha256_file(floor_path),
            NATIVE200_ROWS: sha256_file(native_path),
        },
        "directory_listing_sha256": lsha,
        "directory_listing": names,
        "scope": (
            "the threshold is a function of these two reference numbers and "
            "nothing else, and both were measured on this device in this run; "
            "a reference from another device or another stage is not "
            "substitutable"
        ),
    }

    with open(target_path, "w", encoding="utf-8") as f:
        json.dump(target, f, indent=2, sort_keys=True)
        f.write("\n")
    return target


def do_check(run_dir, questions_path=""):
    target_path = os.path.join(run_dir, TARGET_FILENAME)
    if not os.path.exists(target_path):
        raise SystemExit(f"BLOCKED: no {TARGET_FILENAME} in {run_dir}")
    with open(target_path, encoding="utf-8") as f:
        target = json.load(f)

    problems = []
    for name, want in target["inputs"].items():
        path = os.path.join(run_dir, name)
        if not os.path.exists(path):
            problems.append(f"{name} is gone")
            continue
        got = sha256_file(path)
        if got != want:
            problems.append(f"{name} changed after the freeze ({got} != {want})")

    if questions_path:
        with open(questions_path, encoding="utf-8") as f:
            qsha = json.load(f)["question_set_sha256"]
        if qsha != target["question_set_sha256"]:
            problems.append(
                f"question set {qsha} is not the one frozen against "
                f"({target['question_set_sha256']})"
            )

    if problems:
        raise SystemExit("INVALID: " + "; ".join(problems))
    return target


def require_target_for_mapped(run_dir, questions_path, arms):
    """The guard the runner calls before it evaluates anything mapped.

    Kept here rather than in the runner so that the rule and its enforcement
    live in one file and are tested together.
    """
    mapped = [a for a in arms if a.startswith(MAPPED_PREFIX)]
    if not mapped:
        return None
    target = do_check(run_dir, questions_path)
    if target.get("threshold") is None:
        raise SystemExit(
            f"BLOCKED: {target.get('verdict_if_no_headroom')} -- the native "
            "reference did not beat the floor, so there is no headroom for a "
            "mapped arm to recover and no threshold to clear."
        )
    return target


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)

    f = sub.add_parser("freeze", help="compute and write the target, once")
    f.add_argument("--run-dir", required=True)
    f.add_argument("--questions", required=True)
    f.add_argument("--expect-question-sha256", default="")

    c = sub.add_parser("check", help="verify the references have not moved")
    c.add_argument("--run-dir", required=True)
    c.add_argument("--questions", default="")

    args = ap.parse_args()

    if args.cmd == "freeze":
        t = do_freeze(args.run_dir, args.questions, args.expect_question_sha256)
        print(
            f"floor {t['floor_correct']}/{t['questions']}, "
            f"native at 200 updates {t['native200_correct']}/{t['questions']}"
        )
        if t["threshold"] is None:
            print(f"  {t['verdict_if_no_headroom']}: no headroom, no threshold")
        else:
            print(
                f"  threshold {t['threshold']}/{t['questions']} "
                f"(frozen {t['frozen_utc']})"
            )
        return 0

    t = do_check(args.run_dir, args.questions)
    print(f"target intact: threshold {t['threshold']}, frozen {t['frozen_utc']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
