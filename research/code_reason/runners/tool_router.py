#!/usr/bin/env python3
"""Route model tool calls to the sandboxed read-only tools.

The model is offered exactly the execution-free tools: read a file range,
list files, grep, a static symbol index, and a guarded shell whose commands
pass through SafeShell (so a shell tool can never run target code or tests).
`tool_specs` is the schema list handed to the model; `dispatch` runs one call
and returns a JSON-able result. Submit tools are NOT dispatched here -- the
agent loop intercepts them as the final answer -- but they appear in
`tool_specs` so the model knows how to finish.
"""

from __future__ import annotations

import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "tools"))
from repo_reader import RepoReader  # noqa: E402
from safe_shell import SafeShell, ExecutionPolicy  # noqa: E402
from static_index import build_index  # noqa: E402

from model_client import SUBMIT_ANSWER, SUBMIT_CERTIFICATE  # noqa: E402


def tool_specs(semiformal=False, include_shell=True):
    """Anthropic-style tool schema list offered to the model."""
    specs = [
        {
            "name": "read_file",
            "description": "Read a file's line range (1-indexed, inclusive).",
            "input_schema": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "line_start": {"type": "integer"},
                    "line_end": {"type": "integer"},
                },
                "required": ["path"],
            },
        },
        {
            "name": "list_files",
            "description": "List repository files matching a glob.",
            "input_schema": {
                "type": "object",
                "properties": {"glob": {"type": "string"}},
            },
        },
        {
            "name": "grep",
            "description": "Search files for a regex pattern.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "pattern": {"type": "string"},
                    "glob": {"type": "string"},
                    "ignore_case": {"type": "boolean"},
                },
                "required": ["pattern"],
            },
        },
        {
            "name": "symbol_index",
            "description": "Static symbol -> file:line index (no execution).",
            "input_schema": {
                "type": "object",
                "properties": {"glob": {"type": "string"}},
            },
        },
    ]
    if include_shell:
        specs.append(
            {
                "name": "shell",
                "description": (
                    "Run one read-only command (cat/grep/ls/find/...). "
                    "Execution, tests, installs, git history are refused."
                ),
                "input_schema": {
                    "type": "object",
                    "properties": {"command": {"type": "string"}},
                    "required": ["command"],
                },
            }
        )
    submit = {
        "name": SUBMIT_CERTIFICATE if semiformal else SUBMIT_ANSWER,
        "description": (
            "Submit the final semi-formal certificate."
            if semiformal
            else "Submit the final answer object."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "answer": {"type": "object"},
                "certificate": {"type": "object"},
            },
            "required": ["answer"] if not semiformal else ["answer", "certificate"],
        },
    }
    specs.append(submit)
    return specs


class ToolRouter:
    def __init__(self, reader, shell=None):
        self.reader = reader
        self.shell = shell
        self._index = None
        self.calls = 0

    @classmethod
    def for_repo(cls, repo_root, flags=None):
        reader = RepoReader(repo_root)
        policy = ExecutionPolicy.from_flags(flags or {})
        return cls(reader, SafeShell(policy, repo_root))

    def dispatch(self, name, args):
        self.calls += 1
        args = args or {}
        try:
            if name == "read_file":
                return self.reader.read(
                    args["path"], args.get("line_start"), args.get("line_end")
                )
            if name == "list_files":
                return {"files": self.reader.list_files(args.get("glob", "**/*"))}
            if name == "grep":
                return self.reader.grep(
                    args["pattern"],
                    args.get("glob", "**/*"),
                    ignore_case=args.get("ignore_case", False),
                )
            if name == "symbol_index":
                if self._index is None:
                    self._index = build_index(self.reader, args.get("glob", "**/*"))
                return {"symbols": self._index}
            if name == "shell":
                if self.shell is None:
                    return {"error": "shell disabled"}
                return self.shell.run(args.get("command", ""))
            return {"error": f"unknown tool: {name}"}
        except KeyError as exc:
            return {"error": f"missing argument {exc}"}
        except Exception as exc:  # pragma: no cover - defensive
            return {"error": f"{type(exc).__name__}: {exc}"}


def _self_test():
    import tempfile

    d = tempfile.mkdtemp()
    with open(os.path.join(d, "m.py"), "w") as fh:
        fh.write("def foo():\n    return 1\n")
    r = ToolRouter.for_repo(d, {})
    assert r.dispatch("list_files", {"glob": "**/*.py"})["files"] == ["m.py"]
    assert r.dispatch("grep", {"pattern": "def "})["hits"][0]["line"] == 1
    assert "foo" in r.dispatch("symbol_index", {})["symbols"]
    assert r.dispatch("read_file", {"path": "m.py"})["total_lines"] == 2
    # shell: read-only allowed, execution refused
    assert r.dispatch("shell", {"command": "ls"})["allowed"] is True
    blocked = r.dispatch("shell", {"command": "pytest m.py"})
    assert blocked["allowed"] is False and "test execution" in blocked["reason"]
    assert r.dispatch("bogus", {})["error"].startswith("unknown tool")
    print("[tool_router] self-test PASS (read tools + shell guard + unknown)")


if __name__ == "__main__":
    if "--self-test" in sys.argv:
        _self_test()
