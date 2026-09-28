#!/usr/bin/env python3
"""Build the development workload and count what is left for confirmation.

Two things are produced, both on the processor and both before any model
runs. The first is the development workload itself: every prefix and suffix
as token arrays, with the record that ties each to its source. The second is
an inventory of how many more independent contexts the sources can give,
since a confirmation cohort that cannot be built should be known before a
card is rented, not after.

Independence is decided by source group. A document context owns the
Wikipedia titles of the passages placed in it, and no other context may use
them. A code context owns its repository. One repository gives one context
however many tasks it holds.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import os
import sys

import numpy as np

from . import answers, code_tasks, contexts

LENGTHS = (4096, 16384, 32768)
DEVELOPMENT = {4096: 8, 16384: 16, 32768: 8}
# the confirmation envelope keeps a quarter, a half and a quarter
CYCLE = (4096, 16384, 16384, 32768)
DOCUMENT_NEW_TOKENS = 48
CODE_NEW_TOKENS = 768
CODE_BODY_LIMIT = 512
SPECIAL_FLOOR = 151643


def order_key(seed, name):
    return hashlib.sha256(f"{seed}|{name}".encode()).hexdigest()


def sha_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 22), b""):
            h.update(block)
    return h.hexdigest()


def save_tokens(path, tokens):
    arr = np.asarray(tokens, dtype=np.int32)
    np.save(path, arr)
    return contexts.digest(tokens)


def plain(tokens, head):
    """No control token may appear in a prefix after its opening."""
    return all(t < SPECIAL_FLOOR for t in tokens[head:])


# ---- documents ------------------------------------------------------------


class DocumentPool:
    """Questions in a fixed order, handing out passages nobody has used."""

    def __init__(self, questions, encode, seed):
        self.q = sorted(questions, key=lambda q: order_key(seed, q["id"]))
        self.encode = encode
        self.seed = seed
        self.used_titles = set()
        self.spent = set()
        self.cursor = 0
        self.lengths = {}

    def tokens_in(self, p):
        key = (p["title"], len(p["paragraph_text"]))
        if key not in self.lengths:
            self.lengths[key] = len(self.encode(contexts.passage(p)))
        return self.lengths[key]

    def free(self, q):
        return q["id"] not in self.spent and not any(
            p["title"] in self.used_titles
            for p in q["paragraphs"]
            if p["is_supporting"]
        )

    def take_queries(self):
        first = None
        for i in range(self.cursor, len(self.q)):
            q = self.q[i]
            if not self.free(q):
                continue
            if first is None:
                first = q
                continue
            if contexts.titles_of(q) & contexts.titles_of(first):
                continue
            if answers.normalise(q["answer"]) == answers.normalise(first["answer"]):
                continue
            self.spent |= {first["id"], q["id"]}
            return [first, q]
        return None

    def filler(self, taken, want):
        """Unused passages from questions that are asked of no context."""
        out, have = [], 0
        while have < want and self.cursor < len(self.q):
            q = self.q[self.cursor]
            self.cursor += 1
            if q["id"] in self.spent:
                continue
            self.spent.add(q["id"])
            for p in q["paragraphs"]:
                if p["title"] in self.used_titles or p["title"] in taken:
                    continue
                taken.add(p["title"])
                out.append(p)
                have += self.tokens_in(p)
        return out

    def build(self, length, name):
        queries = self.take_queries()
        if queries is None:
            return None
        clean = []
        for q in queries:
            keep = [
                p
                for p in q["paragraphs"]
                if p["is_supporting"] or p["title"] not in self.used_titles
            ]
            clean.append(dict(q, paragraphs=keep))
        taken = set().union(*(contexts.titles_of(q) for q in clean))
        # a question can hold two passages of one article; only one is placed
        counted, own = set(), 0
        for q in clean:
            for p in q["paragraphs"]:
                if p["title"] not in counted:
                    counted.add(p["title"])
                    own += self.tokens_in(p)
        fill = self.filler(taken, max(0, length - own) + 1024)
        try:
            ctx = contexts.document_context(
                clean, fill, length, self.encode, f"{self.seed}|{name}"
            )
        except contexts.Unbuildable:
            return None
        self.used_titles |= {x["key"].split(" #")[0] for x in ctx["layout"]}
        self.used_titles |= taken & set().union(*(contexts.titles_of(q) for q in clean))
        return ctx


# ---- code -----------------------------------------------------------------


def eligible_by_repository(tasks, verdicts, source, encode):
    ok = {v["namespace"] for v in verdicts if v["eligible"]}
    by = collections.defaultdict(list)
    dropped = collections.Counter()
    for t in tasks:
        if t["namespace"] not in ok:
            dropped["tests_unusable_here"] += 1
            continue
        body = code_tasks.reference_body(source, t)
        if len(encode(body)) > CODE_BODY_LIMIT:
            dropped["reference_longer_than_the_answer_limit"] += 1
            continue
        by[code_tasks.repository(t)].append(t)
    return by, dropped


def task_pairs(tasks, seed, limit=12):
    """Pairs to try, tasks that need their repository first."""
    ranked = sorted(
        tasks,
        key=lambda t: (
            not any(
                t["dependency"].get(k)
                for k in ("cross_file", "intra_file", "intra_class")
            ),
            order_key(seed, t["namespace"]),
        ),
    )
    seen, out = set(), []
    for i, a in enumerate(ranked):
        for b in ranked[i + 1 :]:
            names = {
                a["namespace"].rsplit(".", 1)[-1],
                b["namespace"].rsplit(".", 1)[-1],
            }
            if len(names) < 2:
                continue
            key = (a["namespace"], b["namespace"])
            if key not in seen:
                seen.add(key)
                out.append([a, b])
            if len(out) >= limit:
                return out
    return out


def build_code(repo, tasks, source, length, encode, seed):
    rdir = os.path.join(source, repo)
    for pair in task_pairs(tasks, seed):
        sig = {t["namespace"]: code_tasks.signature(source, t) for t in pair}
        try:
            ctx = contexts.code_context(pair, rdir, repo, length, encode, sig)
        except (contexts.Unbuildable, OSError):
            continue
        head = len(encode(contexts.CODE_HEAD))
        if not plain(ctx["prefix"], head):
            continue
        ctx["tasks"] = pair
        return ctx
    return None


def allot(feasible, want, order):
    """Give each repository to one length, scarcest length first."""
    left = dict(want)
    out = {}
    for length in sorted(want, key=lambda n: sum(n in f for f in feasible.values())):
        for repo in order:
            if left[length] == 0:
                break
            if repo in out or length not in feasible[repo]:
                continue
            out[repo] = length
            left[length] -= 1
    return out, left


def capacity(feasible, taken):
    """Most further groups in the quarter, half, quarter proportion."""
    rest = {r: f for r, f in feasible.items() if r not in taken and f}
    best = 0
    for units in range(len(rest) // 4, 0, -1):
        got, left = allot(
            rest,
            {4096: units, 16384: 2 * units, 32768: units},
            sorted(rest),
        )
        if not any(left.values()):
            best = units
            break
    return {"groups": 4 * best, "by_length": {4096: best, 16384: 2 * best, 32768: best}}


# ---- writing --------------------------------------------------------------


def write_context(out, cid, group, domain, length, ctx, dataset, extra):
    tdir = os.path.join(out, "tokens")
    os.makedirs(tdir, exist_ok=True)
    pfile = os.path.join("tokens", f"{cid}.prefix.npy")
    pdig = save_tokens(os.path.join(out, pfile), ctx["prefix"])
    queries = []
    for i, q in enumerate(ctx["queries"]):
        qid = f"{cid}.q{i}"
        sfile = os.path.join("tokens", f"{qid}.suffix.npy")
        sdig = save_tokens(os.path.join(out, sfile), q["suffix"])
        rec = {
            "query_id": qid,
            "suffix_token_file": sfile,
            "suffix_tokens": len(q["suffix"]),
            "suffix_tokens_sha256": sdig,
            "full_prompt_tokens_sha256": contexts.digest(ctx["prefix"] + q["suffix"]),
            "gold_or_test_fixture_id": q["record"],
        }
        if "accepted" in q:
            rec["accepted_answers"] = q["accepted"]
        queries.append(rec)
    return {
        "context_id": cid,
        "source_group_id": group,
        "domain": domain,
        "length_bucket": length,
        "actual_prefix_tokens": len(ctx["prefix"]),
        "prefix_token_file": pfile,
        "prefix_tokens_sha256": pdig,
        "dataset": dataset,
        "dataset_record_id": [q["record"] for q in ctx["queries"]],
        "evidence_pieces": sum(1 for x in ctx["layout"] if x["evidence"]),
        "pieces": len(ctx["layout"]),
        "evidence_whole": not any(x["cut"] for x in ctx["layout"] if x["evidence"]),
        "fixed_queries": queries,
        **extra,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--documents", nargs="+", required=True)
    ap.add_argument("--tasks", required=True)
    ap.add_argument("--eligibility", required=True)
    ap.add_argument("--source", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--revision", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", default="20260928")
    args = ap.parse_args()
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.tokenizer, revision=args.revision)

    def encode(text):
        return tok.encode(text, add_special_tokens=False)

    os.makedirs(args.out, exist_ok=True)
    records, inventory = [], {"seed": args.seed}

    # documents
    questions = []
    for path in args.documents:
        with open(path) as f:
            questions += [json.loads(x) for x in f if x.strip()]
    questions = [q for q in questions if q.get("answerable", True)]
    pool = DocumentPool(questions, encode, args.seed)
    head = len(encode(contexts.DOCUMENT_HEAD))
    n = 0
    for length, count in DEVELOPMENT.items():
        for i in range(count):
            cid = f"dev-doc-{length}-{i:02d}"
            ctx = pool.build(length, cid)
            if ctx is None or not plain(ctx["prefix"], head):
                print(f"could not build {cid}")
                return 1
            records.append(
                write_context(
                    args.out,
                    cid,
                    f"doc-group-{n:05d}",
                    "document",
                    length,
                    ctx,
                    "musique",
                    {},
                )
            )
            n += 1
    built = collections.Counter()
    while True:
        done = 0
        for length in CYCLE:
            ctx = pool.build(length, f"capacity-{sum(built.values())}")
            if ctx is None:
                break
            built[length] += 1
            done += 1
        if done < len(CYCLE):
            break
    units = min(built[4096], built[16384] // 2, built[32768])
    inventory["document"] = {
        "questions_available": len(questions),
        "development_contexts": sum(DEVELOPMENT.values()),
        "further_contexts_buildable": {
            "groups": 4 * units,
            "by_length": {4096: units, 16384: 2 * units, 32768: units},
        },
        "rule": "no Wikipedia title is placed in more than one context",
    }

    # code
    tasks = code_tasks.load(args.tasks)
    with open(args.eligibility) as f:
        verdicts = [json.loads(x) for x in f if x.strip()]
    by, dropped = eligible_by_repository(tasks, verdicts, args.source, encode)
    repos = {r: t for r, t in by.items() if len(t) >= 2}
    feasible, cache = {}, {}
    for repo, ts in sorted(repos.items()):
        feasible[repo] = set()
        for length in LENGTHS:
            ctx = build_code(repo, ts, args.source, length, encode, args.seed)
            if ctx is not None:
                feasible[repo].add(length)
                cache[(repo, length)] = ctx
    order = sorted(feasible, key=lambda r: order_key(args.seed, r))
    chosen, short = allot(feasible, DEVELOPMENT, order)
    for length in LENGTHS:
        picked = [r for r in order if chosen.get(r) == length]
        for i, repo in enumerate(picked):
            ctx = cache[(repo, length)]
            records.append(
                write_context(
                    args.out,
                    f"dev-code-{length}-{i:02d}",
                    "repo:" + repo,
                    "code",
                    length,
                    ctx,
                    "deveval",
                    {"repository": repo},
                )
            )
    inventory["code"] = {
        "tasks_in_benchmark": len(tasks),
        "repositories_in_benchmark": len({code_tasks.repository(t) for t in tasks}),
        "tasks_dropped": dict(dropped),
        "repositories_with_two_usable_tasks": len(repos),
        "repositories_buildable_at": {
            n: sum(n in f for f in feasible.values()) for n in LENGTHS
        },
        "development_contexts": len(chosen),
        "development_short_of_allocation": {k: v for k, v in short.items() if v},
        "further_contexts_buildable": capacity(feasible, set(chosen)),
        "rule": "one repository gives one context, at one length",
    }

    with open(os.path.join(args.out, "records.jsonl"), "w") as f:
        for r in records:
            f.write(json.dumps(r, sort_keys=True) + "\n")
    inventory["records_sha256"] = sha_file(os.path.join(args.out, "records.jsonl"))
    inventory["contexts"] = len(records)
    inventory["queries"] = sum(len(r["fixed_queries"]) for r in records)
    inventory["source_groups"] = len({r["source_group_id"] for r in records})
    inventory["suffix_tokens_max"] = max(
        q["suffix_tokens"] for r in records for q in r["fixed_queries"]
    )
    with open(os.path.join(args.out, "ELIGIBILITY.json"), "w") as f:
        json.dump(inventory, f, indent=2, sort_keys=True)
    print(json.dumps(inventory, indent=1, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
