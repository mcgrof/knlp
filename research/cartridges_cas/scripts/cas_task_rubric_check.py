#!/usr/bin/env python3
"""Check a short-answer task file against synthetic answers.

For every task in each TASKS file (the cas_eval_batched.py tasks
format), score synthetic answers with the evaluator's fixed rule:

- correct: the task's own reference must pass;
- incomplete: for each required group, the reference with every
  alternative of that group removed must fail;
- negated: for each required group, the reference with "not " inserted
  before every occurrence of its alternatives (other than alternatives
  that are themselves negations) must fail, unless the reference also
  states the group through an alternative that is itself a negation
  ("no evidence"), where the answer still asserts the absence;
- contradictory: where the task can detect it (forbidden alternatives
  or exclusive_numbers), the reference followed by a forbidden
  alternative or by "or 99" must fail.

It prints one line per task with any failed check and a summary, and
exits nonzero if any check fails. Tasks with no contradiction guard are
listed: a lexical rule cannot tell a contradictory answer for them.

Env: TASKS (comma list of task files).
"""

import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cas_eval_batched import NEGATIONS, rubric_correct, rubric_hits


def check(task):
    ref = task["reference"]
    fails = []
    if not rubric_correct(ref, task):
        fails.append("reference fails")
    for g, grp in enumerate(task["required"]):
        alts = grp["any"] if isinstance(grp, dict) else grp
        a = " ".join(ref.lower().split())
        for alt in sorted(alts, key=len, reverse=True):
            a = re.sub(re.escape(" ".join(alt.lower().split())), "[x]", a)
        if rubric_correct(a, task):
            fails.append(f"incomplete group {g} passes")
        a = " ".join(ref.lower().split())
        starts = sorted(
            {
                h
                for alt in alts
                if alt.split()[0] not in NEGATIONS
                for h in rubric_hits(a, alt, negation=False)
            },
            reverse=True,
        )
        absence = any(
            rubric_hits(a, alt, negation=False)
            for alt in alts
            if alt.split()[0] in NEGATIONS
        )
        if starts and not absence:
            neg = a
            for h in starts:
                neg = neg[:h] + "not " + neg[h:]
            if rubric_correct(neg, task):
                fails.append(f"negated group {g} passes")
    guarded = True
    if task.get("forbidden"):
        if rubric_correct(ref + " Alternatively, " + task["forbidden"][0], task):
            fails.append("forbidden addition passes")
    elif task.get("exclusive_numbers") is not None:
        if rubric_correct(ref + " or 99", task):
            fails.append("extra number passes")
    else:
        guarded = False
    return fails, guarded


def main():
    bad = 0
    for path in os.environ["TASKS"].split(","):
        tasks = json.load(open(path))
        unguarded = []
        for t in tasks:
            fails, guarded = check(t)
            if fails:
                bad += 1
                print(f"FAIL {path} row {t['row']}: {'; '.join(fails)}")
            if not guarded:
                unguarded.append(t["row"])
        print(
            f"RUBRIC_CHECK {path} tasks={len(tasks)} "
            f"families={len({t.get('family', t['row']) for t in tasks})} "
            f"no_contradiction_guard={len(unguarded)}"
        )
    print(f"RUBRIC_CHECK_RESULT {'PASS' if bad == 0 else 'FAIL'} failing_tasks={bad}")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
