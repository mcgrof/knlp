# SPDX-License-Identifier: GPL-2.0
"""Prefixes of an exact length that keep their evidence whole."""

from __future__ import annotations

import os

import pytest

from research.kv_translate.sharing import code_tasks, contexts


def chars(text):
    return [ord(c) for c in text]


# ---- prefixes of an exact length ------------------------------------------


def question(qid, n=6, support=(0, 1)):
    return {
        "id": qid,
        "question": f"what about {qid}?",
        "answer": f"ans-{qid}",
        "answer_aliases": [f"alias-{qid}"],
        "paragraphs": [
            {
                "title": f"{qid}-t{i}",
                "paragraph_text": f"text {qid} {i} " + "x" * 30,
                "is_supporting": i in support,
            }
            for i in range(n)
        ],
    }


def test_a_document_prefix_is_exact_and_keeps_its_evidence_whole():
    qs = [question("a"), question("b")]
    filler = [p for q in (question("c"), question("d")) for p in q["paragraphs"]]
    ctx = contexts.document_context(qs, filler, 900, chars, seed=1)
    assert len(ctx["prefix"]) == 900
    text = "".join(map(chr, ctx["prefix"]))
    for q in qs:
        for p in q["paragraphs"]:
            if p["is_supporting"]:
                assert contexts.passage(p) in text
    ev = [x for x in ctx["layout"] if x["evidence"]]
    assert len(ev) == 4 and not any(x["cut"] for x in ev)
    assert [x["cut"] for x in ctx["layout"]].count(True) <= 1
    assert ctx["layout"][-1]["start"] + ctx["layout"][-1]["tokens"] == 900
    assert ctx["queries"][0]["accepted"] == ["ans-a", "alias-a"]
    assert "".join(map(chr, ctx["queries"][1]["suffix"])).endswith(contexts.TURN)


def test_the_same_seed_builds_the_same_prefix_and_another_seed_does_not():
    qs = [question("a"), question("b")]
    filler = [p for p in question("c")["paragraphs"]]
    one = contexts.document_context(qs, filler, 700, chars, seed=1)
    two = contexts.document_context(qs, filler, 700, chars, seed=1)
    other = contexts.document_context(qs, filler, 700, chars, seed=2)
    assert one["prefix"] == two["prefix"]
    assert contexts.digest(one["prefix"]) != contexts.digest(other["prefix"])


def test_a_source_too_small_or_evidence_too_large_is_refused():
    qs = [question("a"), question("b")]
    with pytest.raises(contexts.Unbuildable, match="short of"):
        contexts.document_context(qs, [], 5000, chars, seed=1)
    with pytest.raises(contexts.Unbuildable, match="evidence alone"):
        contexts.document_context(qs, [], 120, chars, seed=1)


def test_a_passage_is_never_placed_twice():
    a, b = question("a"), question("b")
    b["paragraphs"][3]["title"] = a["paragraphs"][3]["title"]
    ctx = contexts.document_context([a, b], a["paragraphs"], 500, chars, seed=3)
    keys = [x["key"] for x in ctx["layout"]]
    assert len(keys) == len(set(keys))


def test_two_supporting_passages_of_one_article_are_both_kept():
    a, b = question("a"), question("b")
    a["paragraphs"][1]["title"] = a["paragraphs"][0]["title"]
    ctx = contexts.document_context(
        [a, b], question("c")["paragraphs"], 800, chars, seed=4
    )
    text = "".join(map(chr, ctx["prefix"]))
    assert contexts.passage(a["paragraphs"][0]) in text
    assert contexts.passage(a["paragraphs"][1]) in text
    assert sum(x["evidence"] for x in ctx["layout"]) == 4


def test_tests_and_build_files_are_not_library_source():
    keep = ["pkg/core.py", "src/pkg/util.py", "pkg/sub/__init__.py"]
    drop = [
        "tests/test_core.py", "pkg/test_util.py", "pkg/util_test.py", "setup.py",
        "conftest.py", "docs/conf.py", "examples/demo.py", ".eggs/x/y.py",
        "pkg/tmp_core.py", "pkg/data.json",
    ]  # fmt: skip
    assert all(contexts.is_library_file(p) for p in keep)
    assert not any(contexts.is_library_file(p) for p in drop)


def test_a_dotted_name_resolves_to_the_file_that_defines_it():
    files = ["pkg/__init__.py", "pkg/utils/type_util.py", "src/other/mod.py"]
    assert (
        contexts.file_of("pkg.utils.type_util.is_bool", files)
        == "pkg/utils/type_util.py"
    )
    assert contexts.file_of("pkg.Thing.method", files) == "pkg/__init__.py"
    assert contexts.file_of("other.mod.f", files) == "src/other/mod.py"
    assert contexts.file_of("nowhere.f", files) is None


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


def test_a_code_prefix_holds_the_tasks_file_with_the_body_gone(repo):
    root, task = repo
    rdir = os.path.join(root, "Topic/proj")
    sig = {task["namespace"]: code_tasks.signature(root, task)}
    ctx = contexts.code_context([task], rdir, "Topic/proj", 420, chars, sig)
    text = "".join(map(chr, ctx["prefix"]))
    assert len(ctx["prefix"]) == 420
    assert "### File: pkg/core.py" in text
    assert "def target(a, b):" in text and contexts.REMOVED.strip() in text
    assert "return helper(a) + b - 1" not in text
    assert "def helper(x):" in text
    assert "test_target" not in text
    assert ctx["layout"][0]["key"] == "pkg/core.py" and ctx["layout"][0]["evidence"]
    ask = "".join(map(chr, ctx["queries"][0]["suffix"]))
    assert "def target(a, b):" in ask and "Add two numbers." in ask
