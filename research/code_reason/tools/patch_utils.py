#!/usr/bin/env python3
"""Unified-diff parsing (no repo execution, pure text).

Used by the patch-equivalence task and the manifest builders. Produces
stable per-patch hashes for the manifest and the affected file/line ranges
the agent should focus on.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field

_HUNK = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")
_FILE = re.compile(r"^\+\+\+ [ab]/(.+?)\s*$")


@dataclass
class Hunk:
    old_start: int
    old_count: int
    new_start: int
    new_count: int
    lines: list = field(default_factory=list)


@dataclass
class FileDiff:
    path: str
    hunks: list = field(default_factory=list)


def patch_hash(text):
    return hashlib.sha1(text.encode("utf-8", "replace")).hexdigest()


def parse_unified_diff(text):
    files, cur = [], None
    for line in text.splitlines():
        mf = _FILE.match(line)
        if mf:
            cur = FileDiff(path=mf.group(1))
            files.append(cur)
            continue
        mh = _HUNK.match(line)
        if mh and cur is not None:
            cur.hunks.append(
                Hunk(
                    int(mh.group(1)),
                    int(mh.group(2) or 1),
                    int(mh.group(3)),
                    int(mh.group(4) or 1),
                )
            )
            continue
        if cur is not None and cur.hunks and line and line[0] in "+- ":
            cur.hunks[-1].lines.append(line)
    return files


def affected_files(files):
    return sorted({f.path for f in files})


def affected_line_ranges(files):
    """{path: [(new_start, new_end)]} of changed regions."""
    out = {}
    for f in files:
        rngs = []
        for h in f.hunks:
            end = h.new_start + max(h.new_count, 1) - 1
            rngs.append((h.new_start, end))
        if rngs:
            out[f.path] = rngs
    return out
