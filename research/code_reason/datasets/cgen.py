#!/usr/bin/env python3
"""Synthetic C first-fault benchmark generator (contamination-resistant).

Each item is a self-contained C program: a function under test plus a `main`
driver that calls it with ONE concrete, hard-coded input. Run under that input,
the program is either

  * clean  -- returns/prints a well-defined final value, no UB reached; or
  * buggy  -- reaches exactly one planted undefined-behavior event first, at a
              known line and category.

Ground truth is by CONSTRUCTION: the generator plants (or withholds) a specific
defect on the path the concrete input takes and records a machine-readable
proof (category, first-fault line, trace, and the clean final value). Sanitizers
are used only to CONFIRM that construction (see cgate oracle), never to define
the label.

Two design rules make it hard to game:
  1. Every buggy item has a surface-matched SAFE sibling from the same family --
     near-identical text, differing in the one constant/operator that decides
     safety -- so a "bug-smell" classifier cannot separate them.
  2. Families are held out between calibration and evaluation, and identifiers,
     constants, and driver inputs are randomized per item, so there is nothing
     to memorize and no fixed template tell.

The narrowed v1 category set (per the design review) drops strict-aliasing,
data races, deadlocks, and uninitialized reads -- those need MSan/TSan and are
ambiguous to grade -- keeping: signed-overflow, shift-UB, OOB read/write,
use-after-free, double-free, null-deref, and clean.
"""

from __future__ import annotations

import argparse
import json
import os
import random

# ---- small helpers for building line-numbered source -----------------------


class Prog:
    """Accumulates source lines and remembers the fault line by label."""

    def __init__(self):
        self.lines = []
        self.marks = {}

    def add(self, text="", mark=None):
        self.lines.append(text)
        if mark:
            self.marks[mark] = len(self.lines)  # 1-indexed line number
        return len(self.lines)

    def source(self):
        return "\n".join(self.lines) + "\n"


def _names(rng, n):
    pool = [
        "acc",
        "buf",
        "idx",
        "cur",
        "tmp",
        "val",
        "cnt",
        "ptr",
        "arr",
        "sum",
        "res",
        "node",
        "data",
        "step",
        "lim",
        "off",
    ]
    rng.shuffle(pool)
    return pool[:n]


# ---- families --------------------------------------------------------------
#
# Each family(rng, buggy) -> (source, meta). meta carries the construction
# proof. The safe and buggy siblings share a seed so their surface text matches
# except for the deciding token.


def fam_signed_overflow(rng, buggy):
    a, s = _names(rng, 2)
    n = rng.randint(6, 12)
    base = rng.randint(2_000_000_000, 2_100_000_000)
    step = rng.randint(20_000_000, 90_000_000)
    # clean sibling uses a small base so the same loop never overflows
    clean_base = rng.randint(100, 900)
    p = Prog()
    p.add("#include <stdio.h>")
    p.add("")
    p.add(f"int accumulate(int start, int step, int count) {{")
    p.add(f"    int {a} = start;")
    p.add(f"    for (int {s} = 0; {s} < count; {s}++) {{")
    p.add(f"        {a} = {a} + step;", mark="fault")
    p.add(f"    }}")
    p.add(f"    return {a};")
    p.add("}")
    p.add("")
    p.add("int main(void) {")
    used_base = base if buggy else clean_base
    p.add(f"    int r = accumulate({used_base}, {step}, {n});")
    p.add('    printf("%d\\n", r);')
    p.add("    return 0;")
    p.add("}")
    if buggy:
        meta = _meta(
            "signed-overflow",
            p.marks["fault"],
            [
                f"start={used_base}, step={step}, count={n}",
                f"accumulator grows past INT_MAX during the additions",
                f"signed int overflow at the marked add",
            ],
        )
    else:
        val = clean_base + step * n
        meta = _clean_meta(val)
    return p.source(), meta


def fam_shift_ub(rng, buggy):
    x, s = _names(rng, 2)
    val = rng.randint(1, 255)
    good_shift = rng.randint(1, 20)
    bad_shift = rng.randint(32, 48)  # >= width of int -> UB
    p = Prog()
    p.add("#include <stdio.h>")
    p.add("")
    p.add("unsigned compute(unsigned v, int width) {")
    p.add(f"    unsigned {x} = v;")
    p.add(f"    int {s} = width;")
    p.add(f"    {x} = {x} << {s};", mark="fault")
    p.add(f"    return {x};")
    p.add("}")
    p.add("")
    p.add("int main(void) {")
    sh = bad_shift if buggy else good_shift
    p.add(f"    unsigned r = compute({val}u, {sh});")
    p.add('    printf("%u\\n", r);')
    p.add("    return 0;")
    p.add("}")
    if buggy:
        meta = _meta(
            "shift-UB",
            p.marks["fault"],
            [
                f"v={val}, width={sh}",
                f"shift count {sh} >= bit width of unsigned int (32)",
                "left shift by >= width is undefined",
            ],
        )
    else:
        meta = _clean_meta((val << good_shift) & 0xFFFFFFFF)
    return p.source(), meta


def fam_oob_index(rng, buggy):
    arr, i = _names(rng, 2)
    n = rng.randint(5, 9)
    vals = [rng.randint(1, 20) for _ in range(n)]
    # index computed with branch-dependent arithmetic
    k = rng.randint(1, 3)
    good_idx = n - 1
    bad_idx = n + rng.randint(0, 2)  # out of bounds
    p = Prog()
    p.add("#include <stdio.h>")
    p.add("")
    p.add(f"int pick(const int *{arr}, int count, int k) {{")
    p.add(f"    int {i} = count - 1;")
    p.add(f"    if (k > 0) {{")
    p.add(f"        {i} = {i} + k;")
    p.add(f"    }}")
    p.add(f"    return {arr}[{i}];", mark="fault")
    p.add("}")
    p.add("")
    p.add("int main(void) {")
    p.add(f"    int a[{n}] = {{{', '.join(str(v) for v in vals)}}};")
    # buggy: k>0 pushes index past end; safe: k=0 keeps it at count-1
    used_k = (bad_idx - (n - 1)) if buggy else 0
    p.add(f"    int r = pick(a, {n}, {used_k});")
    p.add('    printf("%d\\n", r);')
    p.add("    return 0;")
    p.add("}")
    if buggy:
        meta = _meta(
            "OOB-read",
            p.marks["fault"],
            [
                f"count={n}, k={used_k}",
                f"idx = (count-1) + k = {n - 1 + used_k} >= {n}",
                "array read past the last valid index",
            ],
        )
    else:
        meta = _clean_meta(vals[good_idx])
    return p.source(), meta


def fam_use_after_free(rng, buggy):
    ptr, v = _names(rng, 2)
    val = rng.randint(10, 99)
    p = Prog()
    p.add("#include <stdio.h>")
    p.add("#include <stdlib.h>")
    p.add("")
    p.add("int run(int seed, int mode) {")
    p.add(f"    int *{ptr} = malloc(sizeof(int));")
    p.add(f"    *{ptr} = seed;")
    if buggy:
        p.add(f"    free({ptr});")
        p.add(f"    int {v} = *{ptr};", mark="fault")
        p.add(f"    return {v};")
    else:
        p.add(f"    int {v} = *{ptr};")
        p.add(f"    free({ptr});")
        p.add(f"    return {v};")
    p.add("}")
    p.add("")
    p.add("int main(void) {")
    p.add(f"    int r = run({val}, 1);")
    p.add('    printf("%d\\n", r);')
    p.add("    return 0;")
    p.add("}")
    if buggy:
        meta = _meta(
            "use-after-free",
            p.marks["fault"],
            [
                f"seed={val}",
                "the heap cell is freed on the line above",
                "the dereference reads freed memory",
            ],
        )
    else:
        meta = _clean_meta(val)
    return p.source(), meta


def fam_double_free(rng, buggy):
    ptr = _names(rng, 1)[0]
    n = rng.randint(3, 7)
    p = Prog()
    p.add("#include <stdio.h>")
    p.add("#include <stdlib.h>")
    p.add("")
    p.add("int run(int n, int twice) {")
    p.add(f"    int *{ptr} = malloc(n * sizeof(int));")
    p.add(f"    for (int j = 0; j < n; j++) {ptr}[j] = j;")
    p.add(f"    int total = 0;")
    p.add(f"    for (int j = 0; j < n; j++) total += {ptr}[j];")
    p.add(f"    free({ptr});")
    if buggy:
        p.add(f"    free({ptr});", mark="fault")
    p.add(f"    return total;")
    p.add("}")
    p.add("")
    p.add("int main(void) {")
    p.add(f"    int r = run({n}, {'1' if buggy else '0'});")
    p.add('    printf("%d\\n", r);')
    p.add("    return 0;")
    p.add("}")
    if buggy:
        meta = _meta(
            "double-free",
            p.marks["fault"],
            [
                f"n={n}",
                "the pointer is already freed one line above",
                "freeing the same pointer twice is undefined",
            ],
        )
    else:
        meta = _clean_meta(sum(range(n)))
    return p.source(), meta


def fam_null_deref(rng, buggy):
    ptr, v = _names(rng, 2)
    thresh = rng.randint(5, 15)
    # buggy input makes the helper return NULL; safe input keeps it non-NULL
    bad_in = rng.randint(0, thresh - 1)
    good_in = rng.randint(thresh, thresh + 10)
    payload = rng.randint(20, 80)
    p = Prog()
    p.add("#include <stdio.h>")
    p.add("#include <stdlib.h>")
    p.add("")
    p.add(f"static int *maybe(int x, int thr, int payload) {{")
    p.add(f"    if (x < thr) return NULL;")
    p.add(f"    int *{ptr} = malloc(sizeof(int));")
    p.add(f"    *{ptr} = payload;")
    p.add(f"    return {ptr};")
    p.add("}")
    p.add("")
    p.add("int consume(int x, int thr, int payload) {")
    p.add(f"    int *{ptr} = maybe(x, thr, payload);")
    if buggy:
        p.add(f"    int {v} = *{ptr};", mark="fault")
        p.add(f"    return {v};")
    else:
        p.add(f"    if ({ptr} == NULL) return -1;")
        p.add(f"    int {v} = *{ptr};")
        p.add(f"    return {v};")
    p.add("}")
    p.add("")
    p.add("int main(void) {")
    used = bad_in if buggy else good_in
    p.add(f"    int r = consume({used}, {thresh}, {payload});")
    p.add('    printf("%d\\n", r);')
    p.add("    return 0;")
    p.add("}")
    if buggy:
        meta = _meta(
            "null-deref",
            p.marks["fault"],
            [
                f"x={used} < thr={thresh}",
                "maybe() returns NULL for this input",
                "the pointer is dereferenced without a NULL check",
            ],
        )
    else:
        meta = _clean_meta(payload)
    return p.source(), meta


def fam_safe_decoy(rng, buggy):
    """Alias-looking-but-safe: allocates, frees, indexes -- all correct.

    Only ever emits a CLEAN item; it exists to give the safe class items that
    have the surface shape (malloc/free/indexing) of a bug so the model cannot
    pass by pattern-matching suspicious idioms. `buggy` is ignored.
    """
    ptr = _names(rng, 1)[0]
    n = rng.randint(4, 8)
    vals = [rng.randint(1, 9) for _ in range(n)]
    p = Prog()
    p.add("#include <stdio.h>")
    p.add("#include <stdlib.h>")
    p.add("")
    p.add("int run(int n) {")
    p.add(f"    int *{ptr} = malloc(n * sizeof(int));")
    p.add(f"    for (int j = 0; j < n; j++) {ptr}[j] = j + 1;")
    p.add(f"    int last = {ptr}[n - 1];")
    p.add(f"    free({ptr});")
    p.add(f"    return last;")
    p.add("}")
    p.add("")
    p.add("int main(void) {")
    p.add(f"    int r = run({n});")
    p.add('    printf("%d\\n", r);')
    p.add("    return 0;")
    p.add("}")
    return p.source(), _clean_meta(n)


FAMILIES = {
    "signed_overflow": fam_signed_overflow,
    "shift_ub": fam_shift_ub,
    "oob_index": fam_oob_index,
    "use_after_free": fam_use_after_free,
    "double_free": fam_double_free,
    "null_deref": fam_null_deref,
    "safe_decoy": fam_safe_decoy,
}


def _meta(category, line, trace):
    return {
        "expected": "ub",
        "category": category,
        "first_fault_line": [line, line],
        "trace": trace,
        "final_value": None,
    }


def _clean_meta(final_value):
    return {
        "expected": "clean",
        "category": "none",
        "first_fault_line": None,
        "trace": ["no undefined behavior on this input"],
        "final_value": int(final_value),
    }


# ---- surface-only view (for the leakage probe) -----------------------------


def surface_view(source):
    """Signatures + brace skeleton only -- what surface_only sees.

    Keeps includes, function-definition headers, and closing braces; drops ALL
    statement bodies, including main()'s driver call. This is deliberate: the
    driver's concrete constants (a huge accumulator base, an out-of-range shift,
    an index offset) are exactly what decides safety, so leaving them in would
    let surface_only "cheat." With bodies and the call removed, a buggy item and
    its safe sibling have byte-identical surfaces, so surface_only cannot beat
    chance -- which is the property the gate checks for.
    """
    types = ("int ", "unsigned ", "static ", "void ", "char ", "long ")
    out = []
    for ln in source.splitlines():
        st = ln.strip()
        is_header = st.endswith("{") and "(" in st and st.startswith(types)
        if st.startswith("#include") or is_header or st == "}":
            out.append(ln)
    return "\n".join(out) + "\n"


# ---- generation ------------------------------------------------------------


def _emit(out_dir, tid, fam, source, meta, pair_id, items):
    d = os.path.join(out_dir, tid)
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, "prog.c"), "w") as fh:
        fh.write(source)
    with open(os.path.join(d, "surface.txt"), "w") as fh:
        fh.write(surface_view(source))
    meta.update({"task_id": tid, "family": fam, "pair_id": pair_id, "confirmed": None})
    with open(os.path.join(d, "meta.json"), "w") as fh:
        json.dump(meta, fh, indent=2, sort_keys=True)
    items.append(tid)


def build(out_dir, n_items, families, seed, comments=False):
    """Emit surface-matched safe/buggy sibling pairs.

    Each pair draws its identifiers, constants, and driver input from ONE seed
    and calls the family twice -- buggy and safe -- so the two siblings differ
    only in the single deciding token (a constant or a comparison). A bug-smell
    classifier cannot separate them; only tracing the concrete input can. Every
    family's random draws happen before the buggy branch, so the same seed
    yields matched text. safe_decoy emits two distinct clean items (it has no
    buggy form) to give the safe class bug-shaped-but-correct decoys.
    """
    os.makedirs(out_dir, exist_ok=True)
    fam_names = list(families)
    items = []
    n_pairs = n_items // 2
    for pi in range(n_pairs):
        fam = fam_names[pi % len(fam_names)]
        pair_id = f"p{pi:04d}-{fam}"
        pseed = seed * 100003 + pi
        if fam == "safe_decoy":
            for k, sub in enumerate([pseed, pseed + 1]):
                src, meta = FAMILIES[fam](random.Random(sub), False)
                _emit(
                    out_dir,
                    f"c-{pi:04d}{chr(97 + k)}-{fam}-clean",
                    fam,
                    src,
                    meta,
                    pair_id,
                    items,
                )
            continue
        srcb, metab = FAMILIES[fam](random.Random(pseed), True)
        srcs, metas = FAMILIES[fam](random.Random(pseed), False)
        _emit(out_dir, f"c-{pi:04d}a-{fam}-ub", fam, srcb, metab, pair_id, items)
        _emit(out_dir, f"c-{pi:04d}b-{fam}-clean", fam, srcs, metas, pair_id, items)
    with open(os.path.join(out_dir, "index.json"), "w") as fh:
        json.dump(
            {"seed": seed, "families": fam_names, "items": items, "n_pairs": n_pairs},
            fh,
            indent=2,
        )
    return items


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=80)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument(
        "--families",
        default=",".join(FAMILIES),
        help="comma list; hold some out for eval vs calibration",
    )
    args = ap.parse_args()
    fams = [f for f in args.families.split(",") if f in FAMILIES]
    items = build(args.out, args.n, fams, args.seed)
    print(f"[cgen] wrote {len(items)} items to {args.out} (families: {fams})")


if __name__ == "__main__":
    main()
