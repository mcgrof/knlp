#!/usr/bin/env python3
"""Find the code tasks whose tests work on this machine, before any model runs.

For every task this runs the repository's own function through the task's
tests, then the same function with its body removed. A task is kept when the
first passes and the second fails. Results are appended as they finish, so an
interrupted scan resumes where it stopped.

Repositories fetch their test dependencies the first time their tests run.
That first run, ``--warm``, is given the network and uses the repository as
its authors left it. Every later run has no network.
"""

from __future__ import annotations

import argparse
import collections
import concurrent.futures as cf
import json
import os
import sys

from . import code_tasks, sandbox


def warm(task, args):
    """One networked run in the master copy, so dependencies are fetched once."""
    master = os.path.join(args.source, code_tasks.repository(task))
    return code_tasks.run_tests(
        task, master, args.python, readable=args.readable, network=True, timeout=900
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--tasks", required=True)
    ap.add_argument("--source", required=True, help="root of the repositories")
    ap.add_argument(
        "--python", required=True, help="interpreter of the test environment"
    )
    ap.add_argument("--scratch", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--readable", action="append", default=[])
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--warm", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    if not sandbox.available():
        print("no sandbox is available; refusing to run tests unconfined")
        return 2
    os.makedirs(args.scratch, exist_ok=True)
    tasks = code_tasks.load(args.tasks)
    if args.limit:
        tasks = tasks[: args.limit]

    if args.warm:
        first = {}
        for t in tasks:
            first.setdefault(code_tasks.repository(t), t)
        done = {}
        if os.path.exists(args.out):
            with open(args.out) as f:
                done = {json.loads(x)["repository"]: 1 for x in f if x.strip()}
        todo = [(r, t) for r, t in first.items() if r not in done]
        with cf.ThreadPoolExecutor(args.workers) as ex, open(args.out, "a") as f:
            futs = {ex.submit(warm, t, args): r for r, t in todo}
            for fu in cf.as_completed(futs):
                r = fu.result()
                f.write(
                    json.dumps(
                        {
                            "repository": futs[fu],
                            "status": r["status"],
                            "seconds": sum(x.get("seconds", 0) for x in r["tests"]),
                            "output_tail": r["tests"][-1].get("output_tail", "")[
                                -1500:
                            ],
                        }
                    )
                    + "\n"
                )
                f.flush()
                print(futs[fu], r["status"], flush=True)
        return 0

    done = set()
    if os.path.exists(args.out):
        with open(args.out) as f:
            done = {json.loads(x)["namespace"] for x in f if x.strip()}
    todo = [t for t in tasks if t["namespace"] not in done]
    counts = collections.Counter()
    with cf.ThreadPoolExecutor(args.workers) as ex, open(args.out, "a") as f:
        futs = {
            ex.submit(
                code_tasks.eligibility,
                t,
                args.source,
                args.python,
                scratch=args.scratch,
                readable=args.readable,
            ): t
            for t in todo
        }
        for i, fu in enumerate(cf.as_completed(futs), 1):
            t, r = futs[fu], fu.result()
            counts[r["reason"]] += 1
            f.write(
                json.dumps(
                    {
                        "namespace": t["namespace"],
                        "repository": code_tasks.repository(t),
                        "eligible": r["eligible"],
                        "reason": r["reason"],
                        "reference_seconds": sum(
                            x.get("seconds", 0) for x in r["reference"]["tests"]
                        ),
                        "output_tail": (
                            ""
                            if r["eligible"]
                            else r["reference"]["tests"][-1].get("output_tail", "")[
                                -800:
                            ]
                        ),
                    }
                )
                + "\n"
            )
            f.flush()
            if i % 50 == 0:
                print(i, len(todo), dict(counts), flush=True)
    print("SCAN_DONE", dict(counts), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
