"""collateral_evolution addendum: mine history for related past changes.

Looks for prior commits that touched the same files, as evidence that a
change follows an established evolution pattern. This is the one addendum
that reads git history, so it is gated by the execution policy: it fires only
when the git-history flag is set (the Kconfig entry selects
ALLOW_STATIC_GIT_HISTORY_ADDENDUM). History is read through SafeShell, which
permits `git log` only under that same allowance -- it can never turn into
target execution.
"""

from __future__ import annotations

import os
import sys

_ADD_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ADD_ROOT)
sys.path.insert(0, os.path.join(os.path.dirname(_ADD_ROOT), "tools"))
from base import Addendum, task_files  # noqa: E402
from safe_shell import SafeShell, ExecutionPolicy  # noqa: E402


class CollateralEvolutionAddendum(Addendum):
    name = "collateral_evolution"
    flag = "CONFIG_CODE_REASON_ADDENDUM_COLLATERAL_EVOLUTION"

    def applicable(self, flags, language):
        if not ExecutionPolicy.from_flags(flags).git_history_allowed:
            return False, "git history not permitted by execution policy"
        return True, ""

    def run(self, task, reader, cert):
        policy = ExecutionPolicy.from_flags(
            {
                "CONFIG_CODE_REASON_AUGMENTED": True,
                "CONFIG_CODE_REASON_ALLOW_STATIC_GIT_HISTORY_ADDENDUM": True,
                "CONFIG_CODE_REASON_NO_GIT_HISTORY": False,
            }
        )
        shell = SafeShell(policy, reader.repo_root)
        history = {}
        for rel in task_files(task):
            res = shell.run(f"git log --oneline -n 10 -- {rel}")
            if res.get("allowed") and res.get("returncode") == 0:
                commits = [line for line in res.get("stdout", "").splitlines() if line]
                history[rel] = commits
            else:
                history[rel] = {"unavailable": res.get("reason", "no history")}
        return {"per_file_history": history}
