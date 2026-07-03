#!/usr/bin/env python3
"""Base class and helpers for the augmented certificate addendums.

An addendum is an optional, independently-ablatable analysis that layers on
top of the paper certificate. Each one is gated three ways: the addendum
system must be on, the addendum's own Kconfig flag must be set, and it must
be applicable to the task's language (Coccinelle is C-family only, for
example). The `envelope` method resolves all three and returns an
addendum-schema record -- enabled with output, or disabled with a reason and
a fallback -- so a run records *why* each addendum did or did not fire.

Addendums stay execution-free: they read the repo, parse ASTs, or (only when
the git-history flag is set) inspect history. They never execute target code.
"""

from __future__ import annotations

import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "tools"))
from patch_utils import parse_unified_diff, affected_files  # noqa: E402

_EXT_LANG = {
    ".py": "python",
    ".java": "java",
    ".c": "c",
    ".h": "c",
    ".cpp": "cpp",
    ".cc": "cpp",
    ".hpp": "cpp",
    ".cxx": "cpp",
}
C_FAMILY = {"c", "cpp"}


def language_of(task):
    lang = task.get("language")
    if lang and lang not in ("unknown", "mixed"):
        return lang
    files = task_files(task)
    langs = {_EXT_LANG.get(os.path.splitext(f)[1]) for f in files}
    langs.discard(None)
    if len(langs) == 1:
        return langs.pop()
    return lang or "unknown"


def task_files(task):
    """Repo-relative files a task touches, from its payload/gold."""
    p = task.get("payload", {}) or {}
    out = []
    for key in ("patch_reference", "patch_candidate"):
        if p.get(key):
            out += affected_files(parse_unified_diff(p[key]))
    for key in ("candidate_files", "context_files"):
        if isinstance(p.get(key), list):
            out += p[key]
    gold = task.get("gold", {}) or {}
    if gold.get("file"):
        out.append(gold["file"])
    # de-dup, preserve order
    seen, files = set(), []
    for f in out:
        if f not in seen:
            seen.add(f)
            files.append(f)
    return files


class Addendum:
    name = "base"
    flag = None  # Kconfig flag that enables it
    fallback = None  # name of the addendum to fall back to when inapplicable

    def applicable(self, flags, language):
        """Return (ok, reason). Default: always applicable."""
        return True, ""

    def run(self, task, reader, cert):
        """Produce the addendum output object. Override."""
        return {}

    def envelope(self, flags, language, task, reader, cert):
        """Resolve gating -> addendum-schema record, or None if not emitted."""
        if flags.get("CONFIG_CODE_REASON_ADDENDUMS") not in (True, "y"):
            return None
        rec = {"name": self.name, "enabled": False, "language": language}
        if flags.get(self.flag) not in (True, "y"):
            rec["reason"] = "flag disabled"
            return rec
        ok, reason = self.applicable(flags, language)
        if not ok:
            rec["reason"] = reason
            rec["fallback"] = self.fallback
            return rec
        rec["enabled"] = True
        rec["output"] = self.run(task, reader, cert)
        return rec
