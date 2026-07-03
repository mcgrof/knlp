#!/usr/bin/env python3
"""Registry of the augmented certificate addendums.

Holds the addendum instances and resolves, for a given set of config flags
and a task, which addendums fire. Each addendum decides its own gating
(system on, own flag on, applicable to the language) in `envelope`, so this
registry just collects the envelopes and validates them against
addendum.schema.json. `run_addendums` returns one record per registered
addendum whenever the addendum system is on -- enabled with output, or
disabled with a reason and a fallback -- so a run documents every addendum's
disposition, not only the ones that fired.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_ROOT, "tools"))

from base import language_of  # noqa: E402
from ast_runtime.addendum import AstRuntimeAddendum  # noqa: E402
from semantic_rewrite.addendum import SemanticRewriteAddendum  # noqa: E402
from coccinelle.addendum import CoccinelleAddendum  # noqa: E402
from collateral_evolution.addendum import CollateralEvolutionAddendum  # noqa: E402
from ab_vs_blb.addendum import AbVsBlbAddendum  # noqa: E402

_SCHEMA = os.path.join(_ROOT, "schemas", "addendum.schema.json")

try:
    import jsonschema  # type: ignore

    _HAVE_JSONSCHEMA = True
except Exception:  # pragma: no cover
    _HAVE_JSONSCHEMA = False


def _validate(obj):
    if _HAVE_JSONSCHEMA:
        with open(_SCHEMA) as fh:
            jsonschema.validate(obj, json.load(fh))


def registered():
    """The addendum instances, in a stable order."""
    return [
        AstRuntimeAddendum(),
        SemanticRewriteAddendum(),
        CoccinelleAddendum(),
        CollateralEvolutionAddendum(),
        AbVsBlbAddendum(),
    ]


def run_addendums(flags, task, reader, cert=None):
    """Return the addendum-schema records for every registered addendum.

    Empty list when the addendum system is off. Each record is schema-valid.
    """
    language = language_of(task)
    records = []
    for add in registered():
        rec = add.envelope(flags, language, task, reader, cert or {})
        if rec is None:
            continue
        _validate(rec)
        records.append(rec)
    return records


def _self_test():
    import sys as _sys

    _sys.path.insert(0, os.path.join(_ROOT, "tools"))
    from repo_reader import RepoReader

    repo = os.path.join(_ROOT, "datasets", "fixtures", "smoke", "repo")
    reader = RepoReader(repo)
    py_task = {
        "task_type": "fault_localization",
        "language": "python",
        "payload": {"candidate_files": ["calc.py"]},
        "gold": {"file": "calc.py", "line_start": 13, "line_end": 18},
    }

    # system off -> nothing
    assert run_addendums({"CONFIG_CODE_REASON_ADDENDUMS": False}, py_task, reader) == []

    # system on, ast + semantic on, coccinelle on (auto-disable), no git
    flags = {
        "CONFIG_CODE_REASON_AUGMENTED": True,
        "CONFIG_CODE_REASON_ADDENDUMS": True,
        "CONFIG_CODE_REASON_ADDENDUM_AST_RUNTIME": True,
        "CONFIG_CODE_REASON_ADDENDUM_SEMANTIC_REWRITE": True,
        "CONFIG_CODE_REASON_ADDENDUM_COCCINELLE": True,
        "CONFIG_CODE_REASON_ADDENDUM_COCCINELLE_AUTO_DISABLE_IF_UNSUPPORTED": True,
        "CONFIG_CODE_REASON_ADDENDUM_COLLATERAL_EVOLUTION": True,
        # collateral-evolution's Kconfig entry selects this allowance:
        "CONFIG_CODE_REASON_ALLOW_STATIC_GIT_HISTORY_ADDENDUM": True,
        "CONFIG_CODE_REASON_ADDENDUM_A_VS_BLB": True,
    }
    recs = {r["name"]: r for r in run_addendums(flags, py_task, reader)}
    assert len(recs) == 5, list(recs)
    # ast fires on python and finds mul
    ast_out = recs["ast_runtime"]
    assert ast_out["enabled"] and any(
        f["name"] == "mul" for f in ast_out["output"]["files"]["calc.py"]["functions"]
    )
    # semantic rewrite uses ast_dump for python
    assert (
        recs["semantic_rewrite"]["output"]["files"]["calc.py"]["method"] == "ast_dump"
    )
    # coccinelle auto-disabled on python, points at fallback
    cocc = recs["coccinelle"]
    assert not cocc["enabled"] and cocc["fallback"] == "semantic_rewrite"
    # collateral evolution: flag on selects git history -> enabled
    ce = recs["collateral_evolution"]
    assert ce["enabled"] and "per_file_history" in ce["output"]
    # ab_vs_blb enabled from flag
    assert recs["ab_vs_blb"]["enabled"]

    # C task -> coccinelle applicable
    c_task = {
        "task_type": "fault_localization",
        "language": "c",
        "payload": {"candidate_files": ["foo.c"]},
    }
    crecs = {r["name"]: r for r in run_addendums(flags, c_task, reader)}
    assert crecs["coccinelle"]["enabled"], crecs["coccinelle"]
    # ast_runtime disabled on C, points at fallback
    assert not crecs["ast_runtime"]["enabled"]
    assert crecs["ast_runtime"]["fallback"] == "semantic_rewrite"

    # git-history flag off -> collateral evolution disabled with reason
    noflag = dict(flags)
    noflag["CONFIG_CODE_REASON_ADDENDUM_COLLATERAL_EVOLUTION"] = False
    r2 = {r["name"]: r for r in run_addendums(noflag, py_task, reader)}
    assert not r2["collateral_evolution"]["enabled"]

    js = "with-schema" if _HAVE_JSONSCHEMA else "no-jsonschema (soft)"
    print(f"[addendums] self-test PASS ({js}): gating + language-adaptive + git gate")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()
    if args.self_test:
        _self_test()


if __name__ == "__main__":
    main()
