#!/usr/bin/env python3
"""Defects4J gold-hunk parsing for fault localization (execution-free).

Defects4J ships a curated `patches/<id>.src.patch` per bug that transforms the
BUGGY source into the FIXED source (the `a/` side is buggy, `b/` is fixed).
For fault localization the gold answer is the set of BUGGY-file regions the fix
touches, grouped by hunk, in BUGGY line numbers -- exactly what the Agentic
Code Reasoning paper matches predictions against (Top-N with "All"/"Any" hunk
coverage).

`parse_gold_hunks` extracts, per hunk, the a-side (buggy) line span of the
removed lines. A pure-insertion hunk (fix only adds lines) has no removed line,
so the fault anchors to the buggy line at the insertion boundary. The rule is
fixed and pre-registered here so it cannot be tuned per result.
"""

from __future__ import annotations

import re

_HUNK = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")
_FILE_A = re.compile(r"^--- a/(.+)$")
_FILE_B = re.compile(r"^\+\+\+ b/(.+)$")


def parse_gold_hunks(patch_text):
    """Return [{file, line_start, line_end, kind}] in buggy line numbers.

    One entry per hunk that changes the buggy file. `kind` is "modify" when
    the hunk removes buggy lines, "insert" when the fix only adds lines (the
    fault anchors to the boundary buggy line).
    """
    gold = []
    cur_file = None
    lines = patch_text.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i]
        mb = _FILE_B.match(line)
        if mb:
            cur_file = mb.group(1)
            i += 1
            continue
        mh = _HUNK.match(line)
        if mh and cur_file is not None:
            a_start = int(mh.group(1))
            a_line = a_start
            removed = []
            insert_anchor = None  # buggy line just before the first insertion
            i += 1
            while (
                i < len(lines)
                and not _HUNK.match(lines[i])
                and not lines[i].startswith("--- a/")
            ):
                body = lines[i]
                if body.startswith("-") and not body.startswith("---"):
                    removed.append(a_line)
                    a_line += 1
                elif body.startswith("+") and not body.startswith("+++"):
                    if insert_anchor is None:
                        # last buggy line consumed before this insertion
                        insert_anchor = max(a_line - 1, 1)
                else:
                    a_line += 1
                i += 1
            # One region per hunk (paper groups predictions by hunk): the
            # a-side span covering every buggy line the hunk changes -- removed
            # lines plus the anchor of any inserted block.
            involved = list(removed)
            if insert_anchor is not None:
                involved.append(insert_anchor)
            if not involved:
                involved = [a_start]
            gold.append(
                {
                    "file": cur_file,
                    "line_start": min(involved),
                    "line_end": max(involved),
                    "kind": "modify" if removed else "insert",
                }
            )
            continue
        i += 1
    return gold


def _self_test():
    # Real Lang-20: two single-line replacements at buggy lines 3298 and 3383.
    lang20 = (
        "diff --git a/src/main/java/org/apache/commons/lang3/StringUtils.java "
        "b/src/main/java/org/apache/commons/lang3/StringUtils.java\n"
        "--- a/src/main/java/org/apache/commons/lang3/StringUtils.java\n"
        "+++ b/src/main/java/org/apache/commons/lang3/StringUtils.java\n"
        "@@ -3295,7 +3295,7 @@ public class StringUtils {\n"
        "             return EMPTY;\n"
        "         }\n"
        "         \n"
        "-        StringBuilder buf = new StringBuilder(noOfItems * 16);\n"
        "+        StringBuilder buf = new StringBuilder(1);\n"
        "         \n"
        "         for (int i = startIndex; i < endIndex; i++) {\n"
        "             if (i > startIndex) {\n"
        "@@ -3380,7 +3380,7 @@ public class StringUtils {\n"
        "             return EMPTY;\n"
        "         }\n"
        "\n"
        "-        StringBuilder buf = new StringBuilder(noOfItems * 16);\n"
        "+        StringBuilder buf = new StringBuilder(2);\n"
        "         \n"
        "         for (int i = startIndex; i < endIndex; i++) {\n"
        "             if (i > startIndex) {\n"
    )
    g = parse_gold_hunks(lang20)
    assert [(x["line_start"], x["line_end"]) for x in g] == [
        (3298, 3298),
        (3383, 3383),
    ], g
    assert all(x["file"].endswith("StringUtils.java") for x in g)

    # Pure-insertion hunk: fix adds a guard; fault anchors to boundary line.
    ins = (
        "--- a/Foo.java\n+++ b/Foo.java\n"
        "@@ -10,3 +10,4 @@\n"
        " a();\n"
        "+ guard();\n"
        " b();\n"
        " c();\n"
    )
    gi = parse_gold_hunks(ins)
    assert gi == [
        {"file": "Foo.java", "line_start": 10, "line_end": 10, "kind": "insert"}
    ], gi

    # Multi-line removal block: contiguous removed span.
    block = (
        "--- a/Bar.java\n+++ b/Bar.java\n"
        "@@ -94,6 +94,2 @@\n"
        " ctx0\n"
        "- r1\n"
        "- r2\n"
        "- r3\n"
        " ctx1\n"
        " ctx2\n"
    )
    gb = parse_gold_hunks(block)
    assert gb == [
        {"file": "Bar.java", "line_start": 95, "line_end": 97, "kind": "modify"}
    ], gb

    # Mixed hunk (buggy->fixed like real Lang-1): an inserted block plus two
    # modified lines. Region spans the insertion anchor through the last
    # removed buggy line.
    mixed = (
        "--- a/N.java\n+++ b/N.java\n"
        "@@ -464,11 +464,20 @@\n"
        " ctxA\n"
        " ctxB\n"
        " if (pfxLen > 0) {\n"
        "+ char firstSigDigit = 0;\n"
        "+ for (...) { }\n"
        " final int hexDigits = ...;\n"
        "- if (hexDigits > 16) {\n"
        "+ if (hexDigits > 16 || x) {\n"
        " return big;\n"
        " }\n"
        "- if (hexDigits > 8) {\n"
        "+ if (hexDigits > 8 || x) {\n"
        " return createInteger;\n"
    )
    gm = parse_gold_hunks(mixed)
    # ctxA=464 ctxB=465 if-pfx=466 (anchor); hexDigits=467; removed at 468,471
    assert gm == [
        {"file": "N.java", "line_start": 466, "line_end": 471, "kind": "modify"}
    ], gm
    print("[defects4j] parse_gold_hunks self-test PASS: modify+insert+block+mixed")


if __name__ == "__main__":
    import sys

    if "--self-test" in sys.argv:
        _self_test()
