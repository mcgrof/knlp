"""coccinelle addendum: SmPL semantic-patch analysis for C-family code.

Coccinelle (`spatch`) matches semantic patches against C/C++ source. This
addendum is applicable only to C-family repositories; with the auto-disable
flag set (the default) it disables itself elsewhere and points at the
semantic_rewrite fallback rather than pretending to run. It never invokes
`spatch` as target execution -- Coccinelle is a static matcher -- but to
avoid a hard dependency on it being installed, the harness records the SmPL
plan it would apply and marks whether `spatch` is available.
"""

from __future__ import annotations

import os
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from base import Addendum, task_files, C_FAMILY  # noqa: E402


class CoccinelleAddendum(Addendum):
    name = "coccinelle"
    flag = "CONFIG_CODE_REASON_ADDENDUM_COCCINELLE"
    fallback = "semantic_rewrite"

    def applicable(self, flags, language):
        if language in C_FAMILY:
            return True, ""
        auto = flags.get(
            "CONFIG_CODE_REASON_ADDENDUM_COCCINELLE_AUTO_DISABLE_IF_UNSUPPORTED"
        ) in (True, "y")
        if auto:
            return False, f"coccinelle auto-disabled: non-C-family ({language})"
        return False, f"coccinelle needs C-family source (language={language})"

    def run(self, task, reader, cert):
        return {
            "spatch_available": shutil.which("spatch") is not None,
            "target_files": task_files(task),
            "smpl_plan": (
                "@@ expression E; @@ - E + E /* structural match plan; "
                "spatch not invoked by the harness */"
            ),
            "status": "planned",
        }
