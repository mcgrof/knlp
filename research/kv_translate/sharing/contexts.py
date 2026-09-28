#!/usr/bin/env python3
"""Build reusable prefixes of an exact length from real documents and code.

A context is a prefix of exactly N tokens, shared by two fixed questions that
are asked after it. The prefix is what a serving system would cache. Nothing
in it is filler: a document prefix is passages from an open question
answering corpus, and a code prefix is the source files of one repository.

Every prefix keeps the evidence its questions need. The passages that
support a document question, and the files a code task depends on, are
placed whole. What is cut to reach the exact length is always something
else, at the end.

Tokens are produced by a function the caller passes in, so the construction
can be tested without a model's tokenizer.
"""

from __future__ import annotations

import hashlib
import os
import random
import textwrap

DOCUMENT_HEAD = (
    "<|im_start|>user\nRead the passages below. Questions about them follow.\n\n"
)
DOCUMENT_ASK = (
    "\n\nQuestion: {question}\nAnswer with the exact answer only, in a few "
    "words, with no explanation."
)
CODE_HEAD = (
    "<|im_start|>user\nBelow are source files from one Python repository. "
    "A request about them follows.\n\n"
)
CODE_ASK = (
    "\n\nWrite the function `{name}` in `{path}`. Its body was removed from "
    "the file above.\n\nSignature:\n```python\n{signature}```\n\n"
    "What it must do:\n{functionality}\n{arguments}\n\n"
    "Reply with the complete function, starting with its signature, in one "
    "```python code block and nothing else."
)
# the template's end of the user turn and start of the answer, thinking off
TURN = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
REMOVED = "# (body removed)\n"


class Unbuildable(ValueError):
    """The source cannot give a prefix of this length that keeps its evidence."""


def digest(tokens):
    h = hashlib.sha256()
    for t in tokens:
        h.update(int(t).to_bytes(4, "little"))
    return h.hexdigest()


def seeded(*parts):
    return random.Random(hashlib.sha256("|".join(map(str, parts)).encode()).digest())


def fill(head, required, optional, length, encode, rng=None):
    """Whole required pieces, then optional ones, then one cut piece, to length.

    ``required`` and ``optional`` are lists of (key, text). Required pieces
    are never cut. Returns the prefix tokens and the layout that was used.
    """
    tokens = list(encode(head))
    pieces = [(k, list(encode(t)), True) for k, t in required]
    need = len(tokens) + sum(len(p[1]) for p in pieces)
    if need > length:
        raise Unbuildable(
            f"the evidence alone is {need} tokens, over a prefix of {length}"
        )
    tail = None
    for k, t in optional:
        if need == length:
            break
        ids = list(encode(t))
        if need + len(ids) <= length:
            pieces.append((k, ids, False))
            need += len(ids)
            continue
        tail = (k, ids[: length - need])
        need = length
    if need < length:
        raise Unbuildable(
            f"the source holds {need} tokens, short of a prefix of {length}"
        )
    if rng is not None:
        rng.shuffle(pieces)
    layout = []
    for k, ids, req in pieces:
        layout.append(
            {
                "key": k,
                "start": len(tokens),
                "tokens": len(ids),
                "evidence": req,
                "cut": False,
            }
        )
        tokens += ids
    if tail is not None:
        layout.append(
            {
                "key": tail[0],
                "start": len(tokens),
                "tokens": len(tail[1]),
                "evidence": False,
                "cut": True,
            }
        )
        tokens += tail[1]
    assert len(tokens) == length, (len(tokens), length)
    return tokens, layout


# ---- documents ------------------------------------------------------------


def passage(p):
    return f"Title: {p['title']}\n{p['paragraph_text'].strip()}\n\n"


def accepted_answers(q):
    return [q["answer"], *q.get("answer_aliases", [])]


def titles_of(q):
    return {p["title"] for p in q["paragraphs"]}


def document_context(queries, filler, length, encode, seed):
    """One document prefix for its fixed questions.

    ``queries`` are the questions asked of this prefix; their supporting
    passages are the evidence. ``filler`` is an ordered list of passages from
    other questions, used only after the queries' own passages run out.
    """
    rng = seeded("document", seed)
    seen, required, own = set(), [], []
    # evidence first and all of it: an article can support a question with
    # two of its passages, and both are needed
    for q in queries:
        for p in q["paragraphs"]:
            if p["is_supporting"]:
                n = sum(1 for k, _ in required if k.split(" #")[0] == p["title"])
                key = p["title"] if n == 0 else f"{p['title']} #{n + 1}"
                required.append((key, passage(p)))
                seen.add(p["title"])
    for q in queries:
        for p in q["paragraphs"]:
            if not p["is_supporting"] and p["title"] not in seen:
                seen.add(p["title"])
                own.append((p["title"], passage(p)))
    rng.shuffle(own)
    rest = [(p["title"], passage(p)) for p in filler if p["title"] not in seen]
    tokens, layout = fill(DOCUMENT_HEAD, required, own + rest, length, encode, rng)
    return {
        "prefix": tokens,
        "layout": layout,
        "queries": [
            {
                "record": q["id"],
                "suffix": list(
                    encode(DOCUMENT_ASK.format(question=q["question"].strip()) + TURN)
                ),
                "accepted": accepted_answers(q),
            }
            for q in queries
        ],
    }


# ---- code -----------------------------------------------------------------

SKIPPED_DIRS = {
    "tests", "test", "testing", "docs", "doc", "examples", "example", "build",
    "dist", "benchmarks", "benchmark", "scripts", "tools", "node_modules",
}  # fmt: skip


def is_library_file(rel):
    """Source a developer would read, not tests that state the answer."""
    parts = rel.split(os.sep)
    name = parts[-1]
    if not name.endswith(".py") or name.startswith("tmp_"):
        return False
    if name.startswith("test_") or name.endswith("_test.py"):
        return False
    if name in ("conftest.py", "setup.py"):
        return False
    return not any(p in SKIPPED_DIRS or p.startswith(".") for p in parts[:-1])


def library_files(repo_dir):
    out = []
    for root, dirs, files in os.walk(repo_dir):
        dirs[:] = sorted(
            d for d in dirs if not d.startswith(".") and not d.endswith(".egg-info")
        )
        for fn in sorted(files):
            rel = os.path.relpath(os.path.join(root, fn), repo_dir)
            if is_library_file(rel):
                out.append(rel)
    return out


def file_of(namespace, files):
    """The source file that defines a dotted name, by its longest module path."""
    parts = namespace.split(".")
    have = set(files)
    for n in range(len(parts), 0, -1):
        stem = os.path.join(*parts[:n])
        for cand in (stem + ".py", os.path.join(stem, "__init__.py")):
            if cand in have:
                return cand
            # repositories that keep their package under src/
            alt = os.path.join("src", cand)
            if alt in have:
                return alt
    return None


def without_bodies(lines, tasks):
    """A file's lines with the named functions' bodies taken out."""
    out = list(lines)
    for t in sorted(tasks, key=lambda t: -t["body_position"][0]):
        a, b = t["body_position"]
        out[a - 1 : b] = [" " * t["indent"] + REMOVED]
    return out


def code_context(tasks, repo_dir, repo_rel, length, encode, signatures):
    """One repository prefix for its fixed tasks.

    The files the tasks are written in, and the files that define what they
    depend on, are the evidence and are placed whole and first. The rest of
    the repository follows in path order.
    """
    files = library_files(repo_dir)
    by_file = {}
    for t in tasks:
        rel = os.path.relpath(t["completion_path"], repo_rel)
        by_file.setdefault(rel, []).append(t)
    needed = list(by_file)
    for t in tasks:
        for kind in ("intra_file", "intra_class", "cross_file"):
            for ns in t["dependency"].get(kind, []):
                f = file_of(ns, files)
                if f and f not in needed:
                    needed.append(f)

    def text(rel):
        with open(os.path.join(repo_dir, rel), encoding="utf-8", errors="replace") as f:
            lines = f.readlines()
        if rel in by_file:
            lines = without_bodies(lines, by_file[rel])
        body = "".join(lines)
        if not body.endswith("\n"):
            body += "\n"
        return f"### File: {rel}\n{body}\n"

    for rel in needed:
        if not os.path.exists(os.path.join(repo_dir, rel)):
            raise Unbuildable(f"{rel} is named by a task and is not in the repository")
    required = [(rel, text(rel)) for rel in needed]
    optional = [(rel, text(rel)) for rel in files if rel not in needed]
    tokens, layout = fill(CODE_HEAD, required, optional, length, encode)
    return {
        "prefix": tokens,
        "layout": layout,
        "queries": [
            {
                "record": t["namespace"],
                "suffix": list(
                    encode(
                        CODE_ASK.format(
                            name=t["namespace"].rsplit(".", 1)[-1],
                            path=os.path.relpath(t["completion_path"], repo_rel),
                            signature=textwrap.dedent(signatures[t["namespace"]]),
                            functionality=t["requirement"]["Functionality"].strip(),
                            arguments=t["requirement"]["Arguments"].strip(),
                        )
                        + TURN
                    )
                ),
            }
            for t in tasks
        ],
    }
