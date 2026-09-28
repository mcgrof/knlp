#!/usr/bin/env python3
"""Score saved generations on the processor, after the card is released.

Every generation was saved raw, so scoring needs no model and can be run
again if a parser turns out to be wrong. A document answer is compared with
the accepted answers. A code answer has its function body taken out of the
reply and is run through the task's tests in the sandbox.

One row is written per generation: whether it is correct, how the answer was
parsed, and which health failures it shows.
"""

from __future__ import annotations

import argparse
import collections
import concurrent.futures as cf
import json
import os
import sys

from . import answers, code_tasks, sandbox


def load_jsonl(path):
    with open(path) as f:
        return [json.loads(x) for x in f if x.strip()]


def visible_text(row):
    """The reply without the end-of-turn marker the model may have emitted."""
    text = row["text"]
    for stop in ("<|im_end|>", "<|endoftext|>"):
        text = text.split(stop)[0]
    return text


def score_document(row, query):
    got = answers.score_document(
        visible_text(row),
        query["accepted_answers"],
        stopped=row["stopped"],
        new_tokens=row["new_tokens"],
        max_new_tokens=row["max_new_tokens"],
    )
    return {"scorer": "exact_match_with_aliases", **got}


def score_code(row, task, args):
    text = visible_text(row)
    name = code_tasks.function_name(task)
    body, how = code_tasks.extract_body(text, name)
    events = answers.health(
        text,
        stopped=row["stopped"],
        new_tokens=row["new_tokens"],
        max_new_tokens=row["max_new_tokens"],
        parsed=body is not None,
    )
    out = {
        "scorer": "all_prescribed_tests_pass",
        "parser": how,
        "health_events": events,
    }
    if body is None:
        return {**out, "correct": False, "tests": "not_run"}
    r = code_tasks.score(
        task,
        body,
        args.source,
        args.python,
        scratch=args.scratch,
        readable=args.readable,
    )
    if r["status"] == sandbox.BROKEN:
        # the sandbox failing is not the model failing
        return {**out, "correct": None, "tests": r["status"]}
    return {
        **out,
        "correct": bool(r["correct"]),
        "tests": r["status"],
        "tests_run": len(r["tests"]),
        "test_seconds": round(sum(t.get("seconds", 0) for t in r["tests"]), 2),
        "failure_tail": (
            "" if r["correct"] else r["tests"][-1].get("output_tail", "")[-600:]
        ),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--generations", required=True)
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
    queries = {}
    for r in load_jsonl(args.records):
        for q in r["fixed_queries"]:
            queries[q["query_id"]] = q
    tasks = {t["namespace"]: t for t in code_tasks.load(args.tasks)}
    rows = load_jsonl(args.generations)
    key = lambda r: (r["query_id"], r["route"], r.get("mapper") or "")  # noqa: E731
    done = set()
    if os.path.exists(args.out):
        done = {key(r) for r in load_jsonl(args.out)}
    todo = [r for r in rows if key(r) not in done]
    dup = [k for k, n in collections.Counter(map(key, rows)).items() if n > 1]
    if dup:
        print(
            f"{len(dup)} generations appear more than once; refusing to guess which counts"
        )
        return 1

    def work(row):
        q = queries[row["query_id"]]
        if row["domain"] == "document":
            s = score_document(row, q)
        else:
            s = score_code(row, tasks[q["gold_or_test_fixture_id"]], args)
        keep = ("identity", "context_id", "source_group_id", "domain", "length_bucket",
                "query_id", "route", "mapper", "stopped", "new_tokens", "max_new_tokens")  # fmt: skip
        return {**{k: row.get(k) for k in keep}, **s}

    counts = collections.Counter()
    with cf.ThreadPoolExecutor(args.workers) as ex, open(args.out, "a") as f:
        for s in ex.map(work, todo):
            f.write(json.dumps(s, sort_keys=True) + "\n")
            f.flush()
            counts[(s["domain"], s["route"], s["correct"])] += 1
    for k, v in sorted(counts.items(), key=str):
        print(k, v)
    broken = sum(v for k, v in counts.items() if k[2] is None)
    if broken:
        print(f"{broken} answers could not be tested because the sandbox failed")
        return 1
    print("SCORING_DONE", len(todo))
    return 0


if __name__ == "__main__":
    sys.exit(main())
