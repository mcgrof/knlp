#!/usr/bin/env python3
"""Lightweight, read-only symbol index (regex, no execution).

A cheap language-adaptive index of top-level definitions so the agent can
jump to symbols without importing or running the target. Not a full parser
(that is the AST addendum's job); this is the grep-grade fallback.
"""

from __future__ import annotations

import os
import re

_DEFS = {
    "python": re.compile(r"^\s*(?:def|class)\s+([A-Za-z_]\w*)"),
    "java": re.compile(
        r"^\s*(?:public|private|protected|static|final|\s)*"
        r"(?:class|interface|enum|[A-Za-z_<>\[\]]+)\s+([A-Za-z_]\w*)\s*\("
    ),
    "cpp": re.compile(r"^[A-Za-z_][\w:<>\*&\s]*\s+([A-Za-z_]\w*)\s*\("),
    "c": re.compile(r"^[A-Za-z_][\w\*&\s]*\s+([A-Za-z_]\w*)\s*\("),
}

_LANG_BY_EXT = {
    ".py": "python",
    ".java": "java",
    ".cpp": "cpp",
    ".cc": "cpp",
    ".cxx": "cpp",
    ".hpp": "cpp",
    ".h": "c",
    ".c": "c",
}


def lang_of(path):
    for ext, lang in _LANG_BY_EXT.items():
        if path.endswith(ext):
            return lang
    return None


def index_file(reader, rel):
    lang = lang_of(rel)
    if not lang:
        return []
    pat = _DEFS[lang]
    res = reader.read(rel)
    out = []
    if "text" not in res:
        return out
    for i, line in enumerate(res["text"].splitlines(), 1):
        m = pat.match(line)
        if m:
            out.append({"symbol": m.group(1), "file": rel, "line": i, "lang": lang})
    return out


def build_index(reader, glob="**/*"):
    idx = {}
    for rel in reader.list_files(glob):
        for e in index_file(reader, rel):
            idx.setdefault(e["symbol"], []).append(
                {"file": e["file"], "line": e["line"], "lang": e["lang"]}
            )
    return idx


def _self_test():
    import sys
    import tempfile

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from repo_reader import RepoReader

    d = tempfile.mkdtemp()
    open(os.path.join(d, "m.py"), "w").write(
        "def foo():\n    pass\n\nclass Bar:\n    def baz(self):\n        pass\n"
    )
    open(os.path.join(d, "n.c"), "w").write(
        "int add(int a, int b) {\n  return a+b;\n}\n"
    )
    idx = build_index(RepoReader(d), "**/*")
    assert "foo" in idx and "Bar" in idx and "baz" in idx, list(idx)
    assert idx["foo"][0]["lang"] == "python"
    assert "add" in idx and idx["add"][0]["lang"] == "c", idx.get("add")
    print("[static_index] self-test PASS: python + c symbols indexed")


if __name__ == "__main__":
    import sys as _sys

    if "--self-test" in _sys.argv:
        _self_test()
