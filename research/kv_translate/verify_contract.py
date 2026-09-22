# SPDX-License-Identifier: GPL-2.0
"""Check the staged inputs against the frozen contract before training.

A contract that is written and never read is a description. These are the
inputs both arms must share, so a mismatch here is not a slow run or a bad
score: it is two arms that differ by something other than the thing under
test, reported in fields that would all look correct.

Run on the machine that will train, against the files it will actually open.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--contract", required=True)
    ap.add_argument("--staged", required=True, help="directory holding the inputs")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    c = json.load(open(args.contract))
    checked, problems = {}, []
    for name, rec in c["inputs"].items():
        want = rec["sha256"]
        local = os.path.join(args.staged, os.path.basename(rec["path"]))
        if not os.path.exists(local):
            problems.append(f"{name}: absent at {local}")
            checked[name] = {"present": False}
            continue
        got = sha(local)
        ok = got == want
        checked[name] = {
            "present": True,
            "path": local,
            "expected": want,
            "got": got,
            "matches": ok,
        }
        if not ok:
            problems.append(f"{name}: hash {got[:16]} != contract {want[:16]}")

    out = {
        "contract": "contract_verification_v1",
        "experiment_id": c.get("experiment_id"),
        "checked": checked,
        "problems": problems,
        "passed": not problems,
        "note": (
            "input identity only. Example and document counts are asserted by "
            "the trainer against the same contract, because they are only "
            "knowable once the corpus has been read."
        ),
    }
    json.dump(out, open(args.out, "w"), indent=2, sort_keys=True)
    for k, v in sorted(checked.items()):
        print(f"  {k:<18s} {'ok' if v.get('matches') else 'MISMATCH/ABSENT'}")
    if problems:
        print(
            f"BLOCKED_INPUTS: {len(problems)} problems: {problems[:3]}", file=sys.stderr
        )
        return 1
    print("contract verified")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
