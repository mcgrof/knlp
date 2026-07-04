#!/usr/bin/env python3
"""Confirm each synthetic C item's construction label with sanitizers.

Compiles every item under UBSan+ASan (abort on first error), runs its driver,
and checks the observed behavior against the generator's construction proof:

  * buggy item  -> the first sanitizer diagnostic must be the planted category
                   at (or adjacent to) the planted first-fault line.
  * clean item  -> the program runs to completion with no diagnostic and prints
                   the generator's predicted final value.

Sanitizers CONFIRM construction; they do not define labels. Any item where the
sanitizer disagrees with construction is marked confirmed=False and excluded
from the gate/eval -- a mismatch means the generator, not the model, is wrong.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import tempfile

# sanitizer diagnostic text -> our category
_SIG = [
    ("signed integer overflow", "signed-overflow"),
    ("shift exponent", "shift-UB"),
    ("left shift of", "shift-UB"),
    ("shift", "shift-UB"),
    ("out of bounds", "OOB-read"),
    ("heap-buffer-overflow", "OOB-read"),
    ("stack-buffer-overflow", "OOB-read"),
    ("global-buffer-overflow", "OOB-read"),
    ("heap-use-after-free", "use-after-free"),
    ("use-after-free", "use-after-free"),
    ("double-free", "double-free"),
    ("attempting double-free", "double-free"),
    ("null pointer", "null-deref"),
    ("load of null", "null-deref"),
    ("SEGV", "null-deref"),
]


def _classify(stderr):
    low = stderr.lower()
    for needle, cat in _SIG:
        if needle.lower() in low:
            return cat
    return None


def _first_line(stderr, cfile):
    base = os.path.basename(cfile)
    m = re.search(re.escape(base) + r":(\d+)", stderr)
    return int(m.group(1)) if m else None


def check_item(d, cc="gcc"):
    meta = json.load(open(os.path.join(d, "meta.json")))
    cfile = os.path.join(d, "prog.c")
    with tempfile.TemporaryDirectory() as td:
        binp = os.path.join(td, "a.out")
        comp = subprocess.run(
            [
                cc,
                "-O0",
                "-g",
                "-fsanitize=undefined,address",
                "-fno-sanitize-recover=all",
                cfile,
                "-o",
                binp,
            ],
            capture_output=True,
            text=True,
        )
        if comp.returncode != 0:
            return {
                "confirmed": False,
                "why": "compile-failed",
                "stderr": comp.stderr[:300],
            }
        env = dict(os.environ, ASAN_OPTIONS="detect_leaks=0:halt_on_error=1")
        run = subprocess.run(
            [binp], capture_output=True, text=True, env=env, timeout=10
        )
    out = run.stdout.strip()
    err = run.stderr
    if meta["expected"] == "clean":
        if run.returncode == 0 and not _classify(err):
            ok = str(meta["final_value"]) == out
            return {
                "confirmed": ok,
                "why": "clean-ok" if ok else f"value {out}!={meta['final_value']}",
                "observed_value": out,
            }
        return {"confirmed": False, "why": "clean-but-faulted", "stderr": err[:300]}
    # buggy: expect a diagnostic of the planted category near the planted line
    cat = _classify(err)
    line = _first_line(err, cfile)
    want_cat = meta["category"]
    want_line = meta["first_fault_line"][0]
    cat_ok = cat == want_cat
    line_ok = line is not None and abs(line - want_line) <= 2
    return {
        "confirmed": bool(cat_ok and line_ok),
        "why": f"cat={cat}(want {want_cat}) line={line}(want {want_line})",
        "observed_category": cat,
        "observed_line": line,
    }


def run(items_dir, cc="gcc"):
    idx = json.load(open(os.path.join(items_dir, "index.json")))
    n_ok = 0
    results = {}
    for tid in idx["items"]:
        d = os.path.join(items_dir, tid)
        r = check_item(d, cc=cc)
        results[tid] = r
        meta = json.load(open(os.path.join(d, "meta.json")))
        meta["confirmed"] = r["confirmed"]
        meta["oracle"] = r
        json.dump(
            meta, open(os.path.join(d, "meta.json"), "w"), indent=2, sort_keys=True
        )
        n_ok += 1 if r["confirmed"] else 0
    json.dump(
        results,
        open(os.path.join(items_dir, "oracle.json"), "w"),
        indent=2,
        sort_keys=True,
    )
    return n_ok, len(idx["items"]), results


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--items", required=True)
    ap.add_argument("--cc", default="gcc")
    args = ap.parse_args()
    ok, tot, results = run(args.items, cc=args.cc)
    print(f"[oracle] confirmed {ok}/{tot}")
    for tid, r in results.items():
        if not r["confirmed"]:
            print(f"  MISMATCH {tid}: {r['why']}")


if __name__ == "__main__":
    main()
