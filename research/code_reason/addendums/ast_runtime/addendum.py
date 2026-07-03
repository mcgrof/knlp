#!/usr/bin/env python3
"""ast_runtime addendum: static AST / runtime-shape assessment.

Parses the affected files with Python's `ast` module (parsing is not
execution) and reports the shape a runtime would see: functions with their
argument counts, classes, and import surface. Python-only; on other languages
it disables itself and points at the semantic_rewrite fallback.
"""

from __future__ import annotations

import ast
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from base import Addendum, task_files, C_FAMILY  # noqa: E402,F401


class AstRuntimeAddendum(Addendum):
    name = "ast_runtime"
    flag = "CONFIG_CODE_REASON_ADDENDUM_AST_RUNTIME"
    fallback = "semantic_rewrite"

    def applicable(self, flags, language):
        if language != "python":
            return False, f"ast_runtime is python-only (language={language})"
        return True, ""

    def run(self, task, reader, cert):
        files = {}
        for rel in task_files(task):
            rec = reader.read(rel)
            if "error" in rec:
                continue
            try:
                tree = ast.parse(rec["text"])
            except SyntaxError as exc:
                files[rel] = {"parse_error": str(exc)}
                continue
            funcs, classes, imports = [], [], []
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    funcs.append(
                        {
                            "name": node.name,
                            "args": len(node.args.args),
                            "lineno": node.lineno,
                        }
                    )
                elif isinstance(node, ast.ClassDef):
                    classes.append({"name": node.name, "lineno": node.lineno})
                elif isinstance(node, (ast.Import, ast.ImportFrom)):
                    imports.append(getattr(node, "module", None) or "import")
            files[rel] = {
                "functions": funcs,
                "classes": classes,
                "imports": sorted(set(i for i in imports if i)),
            }
        return {"files": files}
