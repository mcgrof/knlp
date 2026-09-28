# SPDX-License-Identifier: GPL-2.0
"""Code tasks: the parser, the body replacement, and tests that can fail."""

from __future__ import annotations

import os
import sys

import pytest

from research.kv_translate.sharing import code_tasks, sandbox

LIB = '''\
def helper(x):
    return x + 1


def target(a, b):
    """Add, through the helper."""
    return helper(a) + b - 1


def other(a):
    return a * 2
'''

TESTS = """\
from pkg.core import target, other


def test_target():
    assert target(2, 3) == 5


def test_other():
    assert other(4) == 8
"""

SETUP = """\
from setuptools import setup
setup(name="pkg", version="0", packages=["pkg"])
"""


@pytest.fixture
def repo(tmp_path):
    root = tmp_path / "src"
    r = root / "Topic" / "proj"
    (r / "pkg").mkdir(parents=True)
    (r / "tests").mkdir()
    (r / "pkg" / "__init__.py").write_text("")
    (r / "pkg" / "core.py").write_text(LIB)
    (r / "pkg" / "extra.py").write_text("VALUE = 1\n" * 40)
    (r / "tests" / "test_core.py").write_text(TESTS)
    (r / "setup.py").write_text(SETUP)
    task = {
        "namespace": "pkg.core.target",
        "project_path": "Topic/proj",
        "completion_path": "Topic/proj/pkg/core.py",
        "signature_position": [5, 5],
        "body_position": [6, 7],
        "indent": 4,
        "tests": ["tests/test_core.py::test_target"],
        "dependency": {
            "intra_file": ["pkg.core.helper"],
            "intra_class": [],
            "cross_file": [],
        },
        "requirement": {
            "Functionality": "Add two numbers.",
            "Arguments": ":param a: int",
        },
    }
    return str(root), task


def test_the_body_is_replaced_at_the_tasks_lines_and_indent(repo):
    root, task = repo
    lines = code_tasks.read_lines(root, task)
    out = "".join(code_tasks.with_body(lines, task, "return a + b\n"))
    assert "    return a + b\n" in out
    assert "through the helper" not in out
    assert "def other(a):" in out and "def helper(x):" in out
    assert code_tasks.signature(root, task) == "def target(a, b):\n"
    assert "return helper(a) + b - 1" in code_tasks.reference_body(root, task)


@pytest.mark.parametrize(
    "output,outcome,has_body",
    [
        ("```python\ndef target(a, b):\n    return a + b\n```", "function", True),
        (
            "Sure.\n```python\ndef target(a, b):\n    '''doc'''\n    return a + b\n```\nDone.",
            "function",
            True,
        ),
        (
            "<think>hm</think>```py\ndef target(a, b):\n    return a + b\n```",
            "function",
            True,
        ),
        ("def target(a, b):\n    return a + b\n", "function", True),
        (
            "```python\ndef other(a):\n    return a\n```",
            "no_function_with_that_name",
            False,
        ),
        (
            "```python\ndef target(a, b):\n    return a +\n```",
            "no_function_with_that_name",
            False,
        ),
        (
            "```python\ndef target(a, b):\n    return a + b\n",
            "unterminated_fence",
            False,
        ),
        ("", "empty", False),
        ("<think>never ends", "unterminated_thinking", False),
    ],
)
def test_the_parser_finds_the_function_or_says_why_not(output, outcome, has_body):
    body, how = code_tasks.extract_body(output, "target")
    assert how == outcome
    assert (body is not None) is has_body
    if has_body:
        assert "return a + b" in body and "def target" not in body


needs_sandbox = pytest.mark.skipif(
    not sandbox.available(), reason="bubblewrap is not installed"
)


def run_in(repo, body):
    root, task = repo
    scratch = os.path.join(os.path.dirname(root), "scratch")
    os.makedirs(scratch, exist_ok=True)
    return code_tasks.score(
        task, body, root, sys.executable, scratch=scratch, readable=[]
    )


@needs_sandbox
def test_the_reference_passes_and_wrong_answers_do_not(repo):
    pytest.importorskip("ptr", reason="the repositories run setup.py pytest")
    assert run_in(repo, None)["correct"] is True
    assert run_in(repo, "return helper(a) + b - 1\n")["correct"] is True
    assert run_in(repo, "return a - b\n")["correct"] is False
    assert run_in(repo, code_tasks.REMOVED_BODY)["correct"] is False
    assert run_in(repo, "return a +\n")["status"] == sandbox.FAIL


@needs_sandbox
def test_a_task_whose_tests_cannot_fail_is_not_eligible(repo):
    pytest.importorskip("ptr", reason="the repositories run setup.py pytest")
    root, task = repo
    scratch = os.path.join(os.path.dirname(root), "scratch")
    os.makedirs(scratch, exist_ok=True)
    kw = dict(scratch=scratch, readable=[])
    assert code_tasks.eligibility(task, root, sys.executable, **kw)["eligible"] is True
    blind = dict(task, tests=["tests/test_core.py::test_other"])
    got = code_tasks.eligibility(blind, root, sys.executable, **kw)
    assert got["eligible"] is False
    assert got["reason"] == "tests_pass_without_the_function"
