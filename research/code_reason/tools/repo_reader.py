#!/usr/bin/env python3
"""Sandboxed read-only repository access for the agent.

Every path is resolved and confined to the repository root; attempts to
escape via `..` or absolute paths are rejected. Provides the three tools the
paper agent needs: read a file line range, list files by glob, and grep.
No writes, no execution.
"""

from __future__ import annotations

import glob as globmod
import os
import re

_NOISE = {".git", "__pycache__", "node_modules", ".tox", "build", "dist", ".venv"}


class PathEscape(ValueError):
    pass


class RepoReader:
    def __init__(self, repo_root, max_bytes=200_000):
        self.repo_root = os.path.realpath(repo_root)
        self.max_bytes = max_bytes

    def _resolve(self, rel):
        full = os.path.realpath(os.path.join(self.repo_root, rel))
        if full != self.repo_root and not full.startswith(self.repo_root + os.sep):
            raise PathEscape(f"path escapes repo root: {rel}")
        return full

    def read(self, path, line_start=None, line_end=None):
        """Read [line_start, line_end] (1-indexed, inclusive). None = whole."""
        full = self._resolve(path)
        if not os.path.isfile(full):
            return {"path": path, "error": "not a file"}
        with open(full, "r", errors="replace") as fh:
            lines = fh.readlines()
        n = len(lines)
        lo = 1 if line_start is None else max(1, line_start)
        hi = n if line_end is None else min(n, line_end)
        chunk = lines[lo - 1 : hi]
        text = "".join(chunk)[: self.max_bytes]
        return {
            "path": path,
            "line_start": lo,
            "line_end": hi,
            "total_lines": n,
            "text": text,
        }

    def list_files(self, glob="**/*", limit=2000):
        # glob module understands ** (recursive); fnmatch does not.
        pattern = os.path.join(self.repo_root, glob)
        out = []
        for full in globmod.glob(pattern, recursive=True):
            if not os.path.isfile(full):
                continue
            rel = os.path.relpath(full, self.repo_root)
            if _NOISE & set(rel.split(os.sep)):
                continue
            out.append(rel)
            if len(out) >= limit:
                break
        return sorted(out)

    def grep(self, pattern, glob="**/*", max_hits=500, ignore_case=False):
        flags = re.IGNORECASE if ignore_case else 0
        try:
            rx = re.compile(pattern, flags)
        except re.error as exc:
            return {"pattern": pattern, "error": f"bad regex: {exc}"}
        hits = []
        for rel in self.list_files(glob):
            full = self._resolve(rel)
            try:
                with open(full, "r", errors="replace") as fh:
                    for i, line in enumerate(fh, 1):
                        if rx.search(line):
                            hits.append(
                                {
                                    "file": rel,
                                    "line": i,
                                    "text": line.rstrip("\n")[:400],
                                }
                            )
                            if len(hits) >= max_hits:
                                return {
                                    "pattern": pattern,
                                    "hits": hits,
                                    "truncated": True,
                                }
            except (OSError, UnicodeError):
                continue
        return {"pattern": pattern, "hits": hits, "truncated": False}
