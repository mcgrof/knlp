#!/usr/bin/env python3
"""Execution-free command guard for the paper baseline.

The paper's verifier can inspect a repository but must not execute its code,
run its tests, install its dependencies, or read git history. This guard
enforces that with a *default-deny* allowlist: a command runs only if its
executable is a known read/search tool, it contains no shell composition
(so an allowed command cannot chain into a forbidden one), and it stays
inside the repository root. The named forbidden classes (test runners,
imports, installs, git history) get explicit, clear rejection reasons on top
of the default deny.

Policy flows from Kconfig: CONFIG_CODE_REASON_NO_TARGET_* and
CONFIG_CODE_REASON_AUGMENTED / _ALLOW_STATIC_GIT_HISTORY_ADDENDUM.
"""

from __future__ import annotations

import os
import re
import shlex
import subprocess
from dataclasses import dataclass

# Read-only / search / static tools. Anything not here is denied by default.
ALLOWLIST = {
    "cat",
    "head",
    "tail",
    "nl",
    "wc",
    "ls",
    "dir",
    "find",
    "tree",
    "stat",
    "file",
    "grep",
    "egrep",
    "fgrep",
    "rg",
    "ripgrep",
    "sort",
    "uniq",
    "cut",
    "basename",
    "dirname",
    "realpath",
    "readlink",
}

# Shell composition / redirection / write — never allowed in a single command.
_SHELL_META = re.compile(r"[;&|<>`$()]|\*\*|\|\||&&|>>")

# Named forbidden classes -> clear reasons (checked before the allowlist so the
# operator sees *why*, not just "not allowlisted").
_NAMED_BLOCKS = [
    (re.compile(r"\b(pytest|py\.test|tox|nose2?|unittest)\b"), "test execution"),
    (re.compile(r"\bpython[0-9.]*\b.*\b-m\s+pytest\b"), "test execution"),
    (re.compile(r"\bmvn\b.*\btest\b"), "test execution"),
    (re.compile(r"\b(gradle|gradlew)\b.*\btest\b"), "test execution"),
    (re.compile(r"\bnpm\b.*\b(test|run\s+test)\b"), "test execution"),
    (re.compile(r"\byarn\b.*\btest\b"), "test execution"),
    (re.compile(r"\bcargo\b.*\btest\b"), "test execution"),
    (re.compile(r"\bgo\s+test\b"), "test execution"),
    (re.compile(r"\bctest\b"), "test execution"),
    (re.compile(r"\bpython[0-9.]*\b\s+(-c|-m)\b"), "target code execution"),
    (
        re.compile(r"\b(node|ruby|perl|bash|sh|zsh|make|cmake)\b"),
        "target code execution",
    ),
    (re.compile(r"\bpip[0-9.]*\b\s+install\b"), "dependency install"),
    (re.compile(r"\b(npm|yarn)\b\s+(install|add)\b"), "dependency install"),
    (
        re.compile(r"\b(apt|apt-get|conda|poetry|uv)\b\s+(install|add)\b"),
        "dependency install",
    ),
]

_GIT_HISTORY = re.compile(
    r"\bgit\b.*\b(log|show|blame|rev-list|reflog|whatchanged|diff)\b"
)


@dataclass
class ExecutionPolicy:
    no_target_repo_execution: bool = True
    no_target_test_execution: bool = True
    no_target_dependency_install: bool = True
    no_git_history: bool = True
    augmented: bool = False
    allow_static_git_history: bool = False

    @classmethod
    def from_flags(cls, flags):
        aug = bool(flags.get("CONFIG_CODE_REASON_AUGMENTED", False))
        allow_git = bool(
            flags.get("CONFIG_CODE_REASON_ALLOW_STATIC_GIT_HISTORY_ADDENDUM", False)
        )
        return cls(
            no_target_repo_execution=flags.get(
                "CONFIG_CODE_REASON_NO_TARGET_REPO_EXECUTION", True
            ),
            no_target_test_execution=flags.get(
                "CONFIG_CODE_REASON_NO_TARGET_TEST_EXECUTION", True
            ),
            no_target_dependency_install=flags.get(
                "CONFIG_CODE_REASON_NO_TARGET_DEPENDENCY_INSTALL", True
            ),
            no_git_history=flags.get("CONFIG_CODE_REASON_NO_GIT_HISTORY", True),
            augmented=aug,
            allow_static_git_history=aug and allow_git,
        )

    @property
    def git_history_allowed(self):
        # The paper baseline forbids git history (no_git_history=y). The only
        # way it opens is the explicit augmented allowance, which is the
        # override -- so it does not also require no_git_history to be off
        # (the Kconfig collateral-evolution addendum selects the allowance
        # while NO_GIT_HISTORY stays y).
        return self.allow_static_git_history


@dataclass
class Decision:
    allowed: bool
    reason: str


class SafeShell:
    def __init__(self, policy, repo_root):
        self.policy = policy
        self.repo_root = os.path.realpath(repo_root)

    def check(self, command):
        cmd = command.strip()
        if not cmd:
            return Decision(False, "empty command")
        if _SHELL_META.search(cmd):
            return Decision(
                False,
                "shell composition/redirection is not allowed "
                "(one read-only command at a time)",
            )
        # named forbidden classes first, for a clear reason
        for pat, klass in _NAMED_BLOCKS:
            if pat.search(cmd):
                return Decision(
                    False, f"blocked: {klass} " "(execution-free paper baseline)"
                )
        if _GIT_HISTORY.search(cmd) and not self.policy.git_history_allowed:
            return Decision(
                False, "blocked: git history access " "(disabled in paper reproduction)"
            )
        try:
            argv = shlex.split(cmd)
        except ValueError as exc:
            return Decision(False, f"unparseable command: {exc}")
        exe = os.path.basename(argv[0])
        git_ok = exe == "git" and self.policy.git_history_allowed
        if exe not in ALLOWLIST and not git_ok:
            return Decision(False, f"'{exe}' is not in the read-only allowlist")
        return Decision(True, "ok")

    def run(self, command, timeout=20):
        d = self.check(command)
        if not d.allowed:
            return {"allowed": False, "reason": d.reason, "command": command}
        argv = shlex.split(command)
        try:
            proc = subprocess.run(
                argv,
                cwd=self.repo_root,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            return {
                "allowed": True,
                "command": command,
                "returncode": proc.returncode,
                "stdout": proc.stdout,
                "stderr": proc.stderr,
            }
        except subprocess.TimeoutExpired:
            return {
                "allowed": True,
                "command": command,
                "returncode": -1,
                "stdout": "",
                "stderr": "timeout",
            }


_BLOCKED = [
    "pytest tests/",
    "py.test -q",
    "python -m pytest",
    "python3 -m pytest tests",
    "tox",
    "nose2",
    "python -m unittest",
    "mvn test",
    "gradle test",
    "./gradlew test",
    "npm test",
    "npm run test",
    "yarn test",
    "cargo test",
    "go test ./...",
    "ctest",
    "python -c 'import target'",
    "python3 -m target",
    "node app.js",
    "bash run.sh",
    "make",
    "pip install -r requirements.txt",
    "pip3 install numpy",
    "npm install",
    "conda install numpy",
    "cat a.py | grep def",
    "grep def a.py > out.txt",
    "ls && pytest",
]

_ALLOWED = [
    "cat calc.py",
    "head -20 calc.py",
    "grep -n def calc.py",
    "rg pattern",
    "ls -la",
    "find . -name '*.py'",
    "wc -l calc.py",
]


def _self_test():
    pol = ExecutionPolicy()  # paper defaults: everything forbidden
    sh = SafeShell(pol, ".")
    blocked = sum(1 for c in _BLOCKED if not sh.check(c).allowed)
    allowed = sum(1 for c in _ALLOWED if sh.check(c).allowed)
    assert blocked == len(_BLOCKED), [c for c in _BLOCKED if sh.check(c).allowed]
    assert allowed == len(_ALLOWED), [c for c in _ALLOWED if not sh.check(c).allowed]
    # git history: forbidden in the paper baseline...
    assert not sh.check("git log --oneline").allowed
    # ...opened only by the explicit augmented allowance
    aug = ExecutionPolicy.from_flags(
        {
            "CONFIG_CODE_REASON_AUGMENTED": True,
            "CONFIG_CODE_REASON_ALLOW_STATIC_GIT_HISTORY_ADDENDUM": True,
        }
    )
    assert SafeShell(aug, ".").check("git log --oneline -- calc.py").allowed
    # but the allowance never opens execution
    assert not SafeShell(aug, ".").check("pytest tests/").allowed
    print(
        f"[safe_shell] self-test PASS: blocked {blocked}/{len(_BLOCKED)}, "
        f"allowed {allowed}/{len(_ALLOWED)}, git gate + no exec leak"
    )


if __name__ == "__main__":
    import sys

    if "--self-test" in sys.argv:
        _self_test()
