# SPDX-License-Identifier: GPL-2.0
"""Archive the code that actually ran, with its revision and its dirty state.

An earlier snapshot recorded a revision but not the runners that produced the
results beside it, so the archived source could not establish what the
measured boundary had been. The fix is not to promise better discipline but
to copy the closure of files the run imports, hash each one, and record the
working tree's uncommitted difference alongside the commit it sits on.

Run: python research/kv_translate/archive_source.py --out <dir>
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

# The import closure the timing path actually pulls in, resolved by reading
# the imports rather than by listing a directory and hoping.
FILES = [
    "research/kv_translate/time_first_next_token.py",
    "research/kv_translate/qualify_operator.py",
    "research/kv_translate/make_timing_fixtures.py",
    "research/kv_translate/dry_run_latency.py",
    "research/kv_translate/archive_source.py",
    "research/kv_translate/audit_eos_denominator.py",
    "research/kv_translate/drive_latency.sh",
    "research/kv_translate/drive_remote.sh",
    "research/kv_translate/measure_eos.sh",
    "research/kv_translate/verify_contract.py",
    "research/kv_translate/eos_receipt.py",
    "research/kv_translate/preflight_eos.py",
    "research/kv_translate/acquire_eos.sh",
    "research/kv_translate/eos_objective.py",
    "research/kv_translate/eos_gate.py",
    "research/kv_translate/build_recovery_manifest.py",
    "research/kv_translate/build_recovery_manifest_v2.py",
    "research/kv_translate/recovery.py",
    "research/kv_translate/recovery_gate.py",
    "research/kv_translate/analyse_arm_traces.py",
    "research/kv_translate/cartridge_preflight.py",
    "research/kv_translate/measure_latency.sh",
    "research/kv_translate/closure_receipt.py",
    "research/kv_translate/update_ledger.py",
    "research/kv_translate/render_latency_result.py",
    "research/kv_translate/freeze.py",
    "research/kv_translate/fit.py",
    "research/kv_translate/pairs.py",
    "research/kv_translate/run_a0.py",
    "research/kv_translate/run_a1.py",
    "research/kv_translate/refine.py",
    "research/kv_translate/tasks.py",
    "research/kv_translate/run_probe.py",
    "research/kv_translate/run_dev_gate.py",
    "research/kv_translate/objective_examples.py",
    "research/kv_translate/build_manifests.py",
    "research/kv_translate/rescore_gold.py",
    "research/cartridges_cas/scripts/cas_kv_rope.py",
    "tests/kv_translate/test_latency_fail_closed.py",
    "tests/kv_translate/test_gate_integrity.py",
    "tests/kv_translate/test_eos_objective.py",
    "tests/kv_translate/test_eos_gate_policy.py",
    "tests/kv_translate/test_recovery.py",
    "tests/kv_translate/test_recovery_gate.py",
    "tests/kv_translate/test_driver_flags.py",
    "tests/kv_translate/test_drive_remote.sh",
    "research/kv_translate/replay.py",
    "research/kv_translate/recovery_runner.py",
    "research/kv_translate/pick_candidate.py",
    "research/kv_translate/summarise_recovery.py",
    "research/kv_translate/decompose_recovery_cost.py",
    "research/kv_translate/make_cpu_fixture.py",
    "tests/kv_translate/test_replay_integration.py",
    "tests/kv_translate/test_recovery_runner.py",
    "tests/kv_translate/test_pick_candidate.py",
    # reached through run_a1, which the runner imports for its feature
    # layout; found by the closure check below, not by inspection
    "research/jacobian_kv/artifacts.py",
    "research/jacobian_kv/accounting.py",
    "research/jacobian_kv/evaluate.py",
    "research/jacobian_kv/__init__.py",
    "research/kv_translate/__init__.py",
    # reached from jacobian_kv.evaluate, under a different top-level package;
    # the closure walker found it only after it stopped assuming one
    "tools/kv/fp8_failure/statistics.py",
    "tools/__init__.py",
    "tools/kv/fp8_failure/__init__.py",
]

# Entry points whose import closure the archive must cover. The list above is
# written by hand and has drifted before: a runner was added, the archive kept
# passing, and the snapshot beside the results was missing the file that
# produced them. So the closure is now derived from the imports themselves and
# the hand-written list is checked against it.
ENTRY_POINTS = [
    "research/kv_translate/recovery_runner.py",
    "research/kv_translate/pick_candidate.py",
    "research/kv_translate/recovery_gate.py",
    "research/kv_translate/build_recovery_manifest_v2.py",
    "research/kv_translate/make_cpu_fixture.py",
    "research/kv_translate/time_first_next_token.py",
    "research/kv_translate/qualify_operator.py",
]


def local_imports(rel, seen):
    """Every in-tree module ``rel`` reaches, by reading its imports.

    Import statements are read from the syntax tree rather than executed, so
    this works without the run's dependencies installed and cannot be fooled
    by a module that imports something only on one branch.
    """
    import ast

    path = os.path.join(ROOT, rel)
    if rel in seen or not os.path.exists(path):
        return seen
    seen.add(rel)
    tree = ast.parse(open(path).read(), filename=path)
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            names.add(node.module)
            names.update(f"{node.module}.{a.name}" for a in node.names)
    for name in names:
        # Any dotted name that resolves to a file in this tree is in the
        # closure. Restricting this to one top-level package was the bug it
        # was written to prevent: a module under `research` imports one under
        # `tools`, the walker skipped it, the archive reported success, and
        # the rented card stopped on a missing module.
        parts = name.split(".")
        if not parts or not os.path.isdir(os.path.join(ROOT, parts[0])):
            continue
        # Importing a.b.c executes every package __init__ along the way, so
        # each one is part of what has to be present, not just the leaf.
        cands = []
        for i in range(1, len(parts) + 1):
            cands.append(os.path.join(*parts[:i], "__init__.py"))
        cands.append(os.path.join(*parts) + ".py")
        for cand in cands:
            if os.path.exists(os.path.join(ROOT, cand)):
                local_imports(cand, seen)
    return seen


def closure():
    seen = set()
    for e in ENTRY_POINTS:
        local_imports(e, seen)
    return seen


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def git(*a):
    return subprocess.run(
        ["git", "-C", ROOT, *a], capture_output=True, text=True
    ).stdout.strip()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    reached = closure()
    unlisted = sorted(reached - set(FILES))
    entries, missing = [], []
    for rel in FILES:
        src = os.path.join(ROOT, rel)
        if not os.path.exists(src):
            missing.append(rel)
            continue
        dst = os.path.join(args.out, rel)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copy2(src, dst)
        entries.append({"path": rel, "sha256": sha(src), "bytes": os.path.getsize(src)})

    # The uncommitted difference is part of what ran. Recording only the
    # commit would describe code that does not exist on disk.
    patch = git("diff", "HEAD", "--", *[e["path"] for e in entries])
    with open(os.path.join(args.out, "uncommitted.patch"), "w") as f:
        f.write(patch)

    man = {
        "contract": "source_snapshot_v2",
        "revision": git("rev-parse", "HEAD"),
        "revision_subject": git("log", "-1", "--format=%s"),
        "tree_is_clean_for_these_files": patch.strip() == "",
        "uncommitted_patch_bytes": len(patch),
        "files": sorted(entries, key=lambda e: e["path"]),
        "missing": missing,
        "entry_points": ENTRY_POINTS,
        "import_closure_size": len(reached),
        "unlisted_imports": unlisted,
        "note": (
            "The import closure of the timing path, each file hashed as copied. "
            "A snapshot that records a revision without the runners cannot "
            "establish what boundary was measured, which is the defect this "
            "replaces."
        ),
    }
    if missing:
        print(f"WARNING: {len(missing)} declared files absent: {missing}")
    if unlisted:
        print(
            f"WARNING: {len(unlisted)} modules the entry points import are not "
            f"archived: {unlisted}"
        )
    json.dump(man, open(os.path.join(args.out, "MANIFEST.json"), "w"), indent=2)
    print(f"archived {len(entries)} files at {man['revision'][:12]}")
    print(f"import closure: {len(reached)} modules, {len(unlisted)} unlisted")
    print(f"uncommitted patch: {len(patch)} bytes")
    return 1 if (missing or unlisted) else 0


if __name__ == "__main__":
    raise SystemExit(main())
