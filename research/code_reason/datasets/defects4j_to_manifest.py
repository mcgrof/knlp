#!/usr/bin/env python3
"""Convert Defects4J bugs into locked fault-localization manifest rows.

Execution-free end to end: no JDK, no test runs. For each bug it reads the
buggy/fixed commit SHAs (active-bugs.csv), the loaded-class scope
(loaded_classes/<id>.src -- Defects4J's own precomputed scope, so we do not
reinvent "fit in context"), the gold file (modified_classes), and the failing
test names (trigger_tests, name only -- NO stack trace). Buggy source for each
loaded class is extracted from the project git bundle at the buggy SHA. Gold
fault regions are computed from OUR OWN buggy->fixed diff (never Defects4J's
reverse .src.patch, whose direction silently rigs the labels) and parsed by
defects4j.parse_gold_hunks.

Output per bug: a repo snapshot the RepoReader can read + a manifest row whose
gold carries the buggy-line hunk regions. A lock file freezes the selected bug
IDs, SHAs, snapshot content hashes, and the selection rule so the set cannot be
gardened after seeing model output.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
from defects4j import parse_gold_hunks  # noqa: E402

# project name -> git bundle basename under <repos>/project_repos/
BUNDLE = {
    "Lang": "commons-lang.git",
    "Math": "commons-math.git",
    "Cli": "commons-cli.git",
    "Csv": "commons-csv.git",
    "Codec": "commons-codec.git",
    "Gson": "gson.git",
    "Compress": "commons-compress.git",
    "JxPath": "commons-jxpath.git",
}

# Approx chars-per-token for the fit-in-context budget (documented heuristic).
CHARS_PER_TOKEN = 4


def _git(repo, *args):
    # source files are not always valid UTF-8 (e.g. a (c) latin-1 byte)
    return subprocess.run(
        ["git", "--git-dir", repo, *args],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )


def _src_root(d4j, project, sha):
    """Source dir for a revision from dir-layout.csv (default src/main/java)."""
    path = os.path.join(d4j, "framework", "projects", project, "dir-layout.csv")
    with open(path) as fh:
        for row in csv.reader(fh):
            if row and row[0] == sha:
                return row[1]
    return "src/main/java"


def _class_to_path(src_root, cls):
    # inner classes (Foo$Bar) live in the outer class file
    top = cls.split("$", 1)[0]
    return f"{src_root}/{top.replace('.', '/')}.java"


def _read_list(path):
    if not os.path.exists(path):
        return []
    with open(path) as fh:
        return [ln.strip() for ln in fh if ln.strip()]


def _trigger_tests(d4j, project, bug_id):
    """Failing test method names only (strip the stack trace)."""
    path = os.path.join(
        d4j, "framework", "projects", project, "trigger_tests", str(bug_id)
    )
    tests = []
    if os.path.exists(path):
        with open(path, errors="replace") as fh:
            for ln in fh:
                if ln.startswith("--- "):
                    tests.append(ln[4:].strip())
    return tests


def build_bug(d4j, repos, project, bug_id, snap_root, token_budget):
    """Return (manifest_row, lock_entry) or None if it doesn't fit context."""
    proj_dir = os.path.join(d4j, "framework", "projects", project)
    repo = os.path.join(repos, "project_repos", BUNDLE[project])

    with open(os.path.join(proj_dir, "active-bugs.csv")) as fh:
        rows = {r["bug.id"]: r for r in csv.DictReader(fh)}
    if str(bug_id) not in rows:
        return None
    rec = rows[str(bug_id)]
    buggy_sha = rec["revision.id.buggy"]
    fixed_sha = rec["revision.id.fixed"]

    src_root = _src_root(d4j, project, buggy_sha)
    loaded = _read_list(os.path.join(proj_dir, "loaded_classes", f"{bug_id}.src"))
    modified = _read_list(os.path.join(proj_dir, "modified_classes", f"{bug_id}.src"))
    if not loaded or not modified:
        return None

    # extract buggy source for every loaded class into the snapshot
    snap = os.path.join(snap_root, f"{project}-{bug_id}")
    files, total_chars = {}, 0
    for cls in sorted(set(loaded)):
        rel = _class_to_path(src_root, cls)
        got = _git(repo, "show", f"{buggy_sha}:{rel}")
        if got.returncode != 0:
            continue
        files[rel] = got.stdout
        total_chars += len(got.stdout)
    if not files:
        return None
    if total_chars > token_budget * CHARS_PER_TOKEN:
        return None  # does not fit context -- excluded by the pre-registered rule

    # gold: OUR buggy->fixed diff for each modified class, a-side (buggy) lines
    gold_hunks, gold_files = [], []
    for cls in sorted(set(modified)):
        rel = _class_to_path(src_root, cls)
        diff = _git(repo, "diff", buggy_sha, fixed_sha, "--", rel).stdout
        hunks = parse_gold_hunks(diff)
        gold_hunks.extend(hunks)
        if hunks:
            gold_files.append(rel)
    if not gold_hunks:
        return None
    # the gold file must be in scope (else the task is unfair / mis-scoped)
    if not all(h["file"] in files for h in gold_hunks):
        return None

    # write the snapshot
    for rel, text in files.items():
        dst = os.path.join(snap, rel)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        with open(dst, "w") as fh:
            fh.write(text)

    tests = _trigger_tests(d4j, project, bug_id)
    task_id = f"d4j-{project}-{bug_id}"
    row = {
        "task_id": task_id,
        "task_type": "fault_localization",
        "dataset": "defects4j",
        "repo": f"{project}-{bug_id}",
        "language": "java",
        "source_commit": buggy_sha,
        "payload": {
            "failing_tests": tests,
            "scope_files": sorted(files.keys()),
            "question": (
                f"The test(s) {', '.join(tests) or '(unnamed)'} fail on this "
                f"buggy code. Localize the fault: return up to 5 ranked "
                f"(file, line_start, line_end) regions, most likely first."
            ),
        },
        "gold": {
            "hunks": gold_hunks,
            "files": sorted(set(gold_files)),
            "n_hunks": len(gold_hunks),
        },
    }
    snap_hash = hashlib.sha1(
        json.dumps({k: files[k] for k in sorted(files)}, sort_keys=True).encode()
    ).hexdigest()
    lock = {
        "task_id": task_id,
        "project": project,
        "bug_id": int(bug_id),
        "buggy_sha": buggy_sha,
        "fixed_sha": fixed_sha,
        "scope_files": sorted(files.keys()),
        "scope_chars": total_chars,
        "snapshot_sha1": snap_hash,
        "gold_hunks": gold_hunks,
    }
    return row, lock


def build_set(d4j, repos, selection, out_dir, token_budget=60000):
    """selection: list of (project, bug_id). Writes manifest + lock + snapshots."""
    snap_root = os.path.join(out_dir, "repos")
    os.makedirs(snap_root, exist_ok=True)
    rows, locks, skipped = [], [], []
    for project, bug_id in selection:
        res = build_bug(d4j, repos, project, bug_id, snap_root, token_budget)
        if res is None:
            skipped.append(f"{project}-{bug_id}")
            continue
        row, lock = res
        rows.append(row)
        locks.append(lock)
    with open(os.path.join(out_dir, "manifest.jsonl"), "w") as fh:
        for r in rows:
            fh.write(json.dumps(r, sort_keys=True) + "\n")
    lock_doc = {
        "dataset": "defects4j",
        "task": "fault_localization",
        "selection_rule": (
            "pre-registered (project,bug_id) list; included iff loaded-class "
            f"scope <= {token_budget} tokens (~{CHARS_PER_TOKEN} chars/token), "
            "gold file(s) in scope, gold hunks non-empty; gold = a-side of our "
            "own buggy->fixed diff (not Defects4J's reverse patch)"
        ),
        "token_budget": token_budget,
        "chars_per_token": CHARS_PER_TOKEN,
        "n_selected": len(rows),
        "n_skipped": len(skipped),
        "skipped": skipped,
        "bugs": locks,
    }
    with open(os.path.join(out_dir, "manifest_lock.json"), "w") as fh:
        json.dump(lock_doc, fh, indent=2, sort_keys=True)
    return rows, lock_doc


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--d4j", required=True, help="Defects4J checkout (metadata)")
    ap.add_argument("--repos", required=True, help="extracted defects4j-repos dir")
    ap.add_argument("--out", required=True)
    ap.add_argument("--project", default="Lang")
    ap.add_argument("--bugs", default="", help="comma bug ids; empty = scan 1..N")
    ap.add_argument("--scan", type=int, default=0, help="scan bug ids 1..scan")
    ap.add_argument("--token-budget", type=int, default=60000)
    ap.add_argument("--limit", type=int, default=0, help="cap selected bugs")
    args = ap.parse_args()

    if args.bugs:
        sel = [(args.project, int(b)) for b in args.bugs.split(",")]
    else:
        sel = [(args.project, i) for i in range(1, args.scan + 1)]
    rows, lock = build_set(args.d4j, args.repos, sel, args.out, args.token_budget)
    if args.limit and len(rows) > args.limit:
        print(f"[d4j] note: {len(rows)} fit; use manifest as-is or re-lock")
    print(
        f"[d4j] selected {lock['n_selected']} / {len(sel)} "
        f"(skipped {lock['n_skipped']}) -> {args.out}/manifest.jsonl"
    )


if __name__ == "__main__":
    main()
