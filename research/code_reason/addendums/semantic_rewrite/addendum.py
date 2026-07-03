#!/usr/bin/env python3
"""semantic_rewrite addendum: language-adaptive normalized rewrite.

Produces a normalized view of the affected code so two patches can be
compared past cosmetic differences. For Python it uses `ast.dump` of the
parsed tree (structure, not text); for other languages it falls back to a
whitespace/comment-normalized token view. Always applicable -- it is the
fallback other addendums point at -- but it adapts its method to the
language.
"""

from __future__ import annotations

import ast
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from base import Addendum, task_files  # noqa: E402


def _normalize_text(text):
    lines = []
    for line in text.splitlines():
        line = re.sub(r"#.*$", "", line)
        line = re.sub(r"//.*$", "", line)
        line = " ".join(line.split())
        if line:
            lines.append(line)
    return lines


class SemanticRewriteAddendum(Addendum):
    name = "semantic_rewrite"
    flag = "CONFIG_CODE_REASON_ADDENDUM_SEMANTIC_REWRITE"

    def run(self, task, reader, cert):
        files = {}
        for rel in task_files(task):
            rec = reader.read(rel)
            if "error" in rec:
                continue
            if rel.endswith(".py"):
                try:
                    files[rel] = {
                        "method": "ast_dump",
                        "normalized": ast.dump(ast.parse(rec["text"])),
                    }
                    continue
                except SyntaxError:
                    pass
            files[rel] = {
                "method": "text_normalize",
                "normalized_lines": _normalize_text(rec["text"]),
            }
        return {"files": files}
