#!/usr/bin/env python3
"""Hold the scorers to every question of the workload before a model answers.

For each document question the dataset's answer and each of its aliases must
score as correct, and an empty answer, a wrong answer and an answer that
never closes its reasoning must not. For each code task the repository's own
function, written back as a model would write it, must be parsed and must
pass, and the function with its body removed must fail. A reply with no
function in it must be refused by the parser.

The receipt lists every check. Anything that did not behave as intended
makes the command fail.
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import os
import sys
import textwrap

from . import answers, code_tasks, sandbox


def document_checks(query):
    out = []
    kw = dict(stopped=True, new_tokens=4, max_new_tokens=48)
    for a in query["accepted_answers"]:
        if answers.normalise(a):
            got = answers.score_document(a, query["accepted_answers"], **kw)
            out.append(("accepted_answer_scores_correct", got["correct"] is True))
    wrong = "zzz not the answer zzz"
    for name, text in (
        ("wrong", wrong),
        ("empty", ""),
        ("unterminated_thinking", "<think>" + wrong),
    ):
        got = answers.score_document(text, query["accepted_answers"], **kw)
        out.append((f"{name}_scores_wrong", got["correct"] is False))
    got = answers.score_document("", query["accepted_answers"], **kw)
    out.append(("empty_is_unhealthy", "empty" in got["health_events"]))
    got = answers.score_document(wrong, query["accepted_answers"], **kw)
    out.append(("wrong_but_well_formed_is_healthy", got["health_events"] == []))
    return out


def code_checks(task, args):
    name = code_tasks.function_name(task)
    sig = textwrap.dedent(code_tasks.signature(args.source, task))
    ref = code_tasks.reference_body(args.source, task)
    reply = "```python\n" + sig + textwrap.indent(textwrap.dedent(ref), "    ") + "```"
    body, how = code_tasks.extract_body(reply, name)
    out = [("reference_reply_is_parsed", body is not None)]
    kw = dict(scratch=args.scratch, readable=args.readable)
    if body is not None:
        r = code_tasks.score(task, body, args.source, args.python, **kw)
        out.append(("reference_reply_passes", r["status"] == sandbox.PASS))
    r = code_tasks.score(task, code_tasks.REMOVED_BODY, args.source, args.python, **kw)
    out.append(("removed_body_fails", r["status"] in (sandbox.FAIL, sandbox.TIMEOUT)))
    for label, text in (("prose", "I cannot write this function."), ("empty", "")):
        b, _ = code_tasks.extract_body(text, name)
        out.append((f"{label}_reply_is_refused", b is None))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--records", required=True)
    ap.add_argument("--tasks", required=True)
    ap.add_argument("--source", required=True)
    ap.add_argument("--python", required=True)
    ap.add_argument("--scratch", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--readable", action="append", default=[])
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    os.makedirs(args.scratch, exist_ok=True)
    tasks = {t["namespace"]: t for t in code_tasks.load(args.tasks)}
    with open(args.records) as f:
        records = [json.loads(x) for x in f if x.strip()]
    jobs = []
    for r in records:
        for q in r["fixed_queries"]:
            jobs.append((r["domain"], q))

    def work(job):
        domain, q = job
        if domain == "document":
            checks = document_checks(q)
        else:
            checks = code_checks(tasks[q["gold_or_test_fixture_id"]], args)
        return {
            "query_id": q["query_id"],
            "domain": domain,
            "fixture": q["gold_or_test_fixture_id"],
            "checks": {k: bool(v) for k, v in checks},
            "all_as_intended": all(v for _, v in checks),
        }

    with cf.ThreadPoolExecutor(args.workers) as ex:
        rows = list(ex.map(work, jobs))
    bad = [r for r in rows if not r["all_as_intended"]]
    receipt = {
        "queries": len(rows),
        "checks": sum(len(r["checks"]) for r in rows),
        "queries_with_a_check_that_failed": [r["query_id"] for r in bad],
        "sandbox": "bubblewrap, no network, read-only file system, empty home",
        "test_limits": {
            "seconds_per_test": code_tasks.TEST_TIMEOUT,
            "resident_bytes": code_tasks.TEST_MEMORY,
        },
        "rows": rows,
    }
    with open(args.out, "w") as f:
        json.dump(receipt, f, indent=2, sort_keys=True)
    print(
        json.dumps(
            {
                k: receipt[k]
                for k in ("queries", "checks", "queries_with_a_check_that_failed")
            }
        )
    )
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
