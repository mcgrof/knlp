# SPDX-License-Identifier: GPL-2.0
"""Check that a script's entry point cannot raise NameError when it is run.

A ``__main__`` guard placed above the definitions it calls imports and passes
review cleanly, then raises NameError the moment the script is executed. That
happened once in this program on a rented card, where the cheapest possible
failure became one of the most expensive.

So this asserts two things about a module: the guard is the last statement in
the file, and every free name ``main()`` reads is bound somewhere above it.
Getting the second right means accounting for every way Python binds a name
inside a function -- assignment, ``with ... as``, ``for`` targets,
comprehension variables, ``except ... as``, parameters, walrus, imports,
globals and nested definitions -- because a binding form this does not know
about reads as a missing name and turns the check into noise.
"""

from __future__ import annotations

import ast
import builtins


def _bound_inside(fn):
    """Every name the function body itself binds."""
    names = set()

    def add_target(t):
        if isinstance(t, ast.Name):
            names.add(t.id)
        elif isinstance(t, (ast.Tuple, ast.List)):
            for e in t.elts:
                add_target(e)
        elif isinstance(t, ast.Starred):
            add_target(t.value)

    for a in fn.args.args + fn.args.kwonlyargs + fn.args.posonlyargs:
        names.add(a.arg)
    if fn.args.vararg:
        names.add(fn.args.vararg.arg)
    if fn.args.kwarg:
        names.add(fn.args.kwarg.arg)

    for node in ast.walk(fn):
        if isinstance(node, ast.Assign):
            for t in node.targets:
                add_target(t)
        elif isinstance(node, (ast.AnnAssign, ast.AugAssign)):
            add_target(node.target)
        elif isinstance(node, ast.NamedExpr):
            add_target(node.target)
        elif isinstance(node, (ast.For, ast.AsyncFor)):
            add_target(node.target)
        elif isinstance(node, ast.comprehension):
            add_target(node.target)
        elif isinstance(node, (ast.With, ast.AsyncWith)):
            for item in node.items:
                if item.optional_vars is not None:
                    add_target(item.optional_vars)
        elif isinstance(node, ast.ExceptHandler):
            if node.name:
                names.add(node.name)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            for a in node.names:
                names.add((a.asname or a.name).split(".")[0])
        elif isinstance(node, (ast.Global, ast.Nonlocal)):
            names.update(node.names)
    return names


def assert_entry_point_is_sound(module, entry="main"):
    """Raise AssertionError unless ``module`` can actually be executed."""
    path = module.__file__
    tree = ast.parse(open(path, encoding="utf-8").read())

    assert tree.body, f"{path} is empty"
    last = tree.body[-1]
    assert isinstance(last, ast.If), (
        f"{path}: the last statement must be the __main__ guard, not "
        f"{type(last).__name__}; anything defined after it is not available to it"
    )

    top = {"__name__", "__doc__", "__file__", "__builtins__"}
    fn = None
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            top.add(node.name)
            if node.name == entry:
                fn = node
        elif isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name):
                    top.add(t.id)
        elif isinstance(node, (ast.AnnAssign, ast.AugAssign)):
            if isinstance(node.target, ast.Name):
                top.add(node.target.id)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            for a in node.names:
                top.add((a.asname or a.name).split(".")[0])

    assert fn is not None, f"{path} has no top-level {entry}()"

    used = {
        n.id
        for n in ast.walk(fn)
        if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)
    }
    missing = used - top - _bound_inside(fn) - set(dir(builtins))
    assert not missing, (
        f"{path}: {entry}() reads names that are not bound above it: "
        f"{sorted(missing)}"
    )

    guard_names = {
        n.id
        for n in ast.walk(last)
        if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)
    }
    unbound = guard_names - top - set(dir(builtins))
    assert (
        not unbound
    ), f"{path}: the __main__ guard reads unbound names {sorted(unbound)}"
