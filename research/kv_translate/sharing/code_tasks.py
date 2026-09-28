#!/usr/bin/env python3
"""Code tasks scored by running a repository's own tests.

The tasks come from DevEval (Li et al., 2024): functions taken out of real
repositories, each with the tests its authors wrote for it. A task is correct
when every one of its prescribed tests passes with the generated function in
place, and wrong otherwise. Passing some of the tests is not partial credit.

A task is only usable if its tests can tell a right answer from a wrong one
here, on this machine: the repository's own function must pass them, and the
same function with its body removed must fail them. Both are checked before
any model is asked anything.
"""

from __future__ import annotations

import ast
import json
import os
import re
import shutil
import tempfile
import textwrap

from . import sandbox

REMOVED_BODY = 'raise NotImplementedError("body removed")\n'
TEST_TIMEOUT = 120
TEST_MEMORY = 6 * 2**30
FENCE = re.compile(r"```[ \t]*([A-Za-z0-9_+-]*)[ \t]*\n(.*?)```", re.S)
THINK = re.compile(r"<think>.*?</think>", re.S)


def load(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def repository(task):
    return task["project_path"]


def function_name(task):
    return task["namespace"].rsplit(".", 1)[-1]


def read_lines(root, task):
    with open(os.path.join(root, task["completion_path"]), encoding="utf-8") as f:
        return f.readlines()


def signature(root, task):
    a, b = task["signature_position"]
    return "".join(read_lines(root, task)[a - 1 : b])


def reference_body(root, task):
    a, b = task["body_position"]
    return "".join(read_lines(root, task)[a - 1 : b])


def reindent(code, indent):
    """Strip whatever indentation the code came with and apply the task's."""
    return textwrap.indent(textwrap.dedent(code), " " * indent)


def with_body(lines, task, body):
    """The completion file's lines with the function's body replaced."""
    a, b = task["body_position"]
    text = reindent(body, task["indent"])
    if not text.endswith("\n"):
        text += "\n"
    return lines[: a - 1] + [text] + lines[b:]


def extract_body(output, name):
    """The function body a model wrote, or the reason there is none.

    The model is asked for the whole function in one fenced block. The body
    is whatever follows the signature of the function with the task's name.
    Returns (body, outcome). A body of None is a malformed answer: it scores
    as wrong and is also counted as a generation health failure.
    """
    text = THINK.sub("", output)
    if "<think>" in text:
        return None, "unterminated_thinking"
    blocks = [
        b for lang, b in FENCE.findall(text) if lang.lower() in ("", "python", "py")
    ]
    if not blocks:
        if "```" in text:
            return None, "unterminated_fence"
        blocks = [text]
    for block in blocks:
        src = textwrap.dedent(block)
        try:
            tree = ast.parse(src)
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if (
                isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                and node.name == name
                and node.body
            ):
                lines = src.splitlines(keepends=True)
                first = node.body[0]
                # a leading docstring or decorator-free statement starts the body
                start = first.lineno - 1
                body = "".join(lines[start : node.end_lineno])
                if body.strip():
                    return body, "function"
    if not text.strip():
        return None, "empty"
    return None, "no_function_with_that_name"


def run_tests(task, repo_dir, python, *, readable, network=False, timeout=TEST_TIMEOUT):
    """Run the task's tests in the given copy of its repository."""
    results = []
    for test in task["tests"]:
        r = sandbox.run(
            [python, "setup.py", "pytest", "--addopts", test],
            writable=repo_dir,
            cwd=repo_dir,
            timeout=timeout,
            memory_bytes=TEST_MEMORY,
            readable=readable,
            network=network,
            hide=("/data",),
            env={"PATH": os.path.dirname(python) + ":/usr/bin:/bin"},
        )
        results.append({"test": test, **r})
        if r["status"] != sandbox.PASS:
            break
    status = sandbox.PASS
    for r in results:
        if r["status"] != sandbox.PASS:
            status = r["status"]
            break
    return {"status": status, "tests": results}


def score(task, body, source_root, python, *, scratch, readable, network=False):
    """Pass or fail for one body, run in a disposable copy of the repository.

    ``body`` of None runs the repository as its authors left it.
    """
    master = os.path.join(source_root, repository(task))
    work = tempfile.mkdtemp(prefix="repo-", dir=scratch)
    copy = os.path.join(work, "repo")
    try:
        shutil.copytree(master, copy, symlinks=True)
        if body is not None:
            path = os.path.join(
                copy, os.path.relpath(task["completion_path"], repository(task))
            )
            with open(path, encoding="utf-8") as f:
                lines = f.readlines()
            with open(path, "w", encoding="utf-8") as f:
                f.writelines(with_body(lines, task, body))
        out = run_tests(task, copy, python, readable=readable, network=network)
    finally:
        shutil.rmtree(work, ignore_errors=True)
    out["correct"] = out["status"] == sandbox.PASS
    return out


def eligibility(task, source_root, python, **kw):
    """Can this task's tests tell the repository's function from none at all?"""
    ref = score(task, None, source_root, python, **kw)
    if not ref["correct"]:
        return {
            "eligible": False,
            "reason": f"reference_{ref['status']}",
            "reference": ref,
        }
    gone = score(task, REMOVED_BODY, source_root, python, **kw)
    if gone["status"] == sandbox.BROKEN:
        return {
            "eligible": False,
            "reason": "sandbox_error",
            "reference": ref,
            "removed": gone,
        }
    if gone["correct"]:
        return {
            "eligible": False,
            "reason": "tests_pass_without_the_function",
            "reference": ref,
            "removed": gone,
        }
    return {"eligible": True, "reason": "ok", "reference": ref, "removed": gone}
