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
    "research/kv_translate/acquire_eos.sh",
    "research/kv_translate/eos_objective.py",
    "research/kv_translate/eos_gate.py",
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
    "tests/kv_translate/test_drive_remote.sh",
]


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
        "note": (
            "The import closure of the timing path, each file hashed as copied. "
            "A snapshot that records a revision without the runners cannot "
            "establish what boundary was measured, which is the defect this "
            "replaces."
        ),
    }
    if missing:
        print(f"WARNING: {len(missing)} declared files absent: {missing}")
    json.dump(man, open(os.path.join(args.out, "MANIFEST.json"), "w"), indent=2)
    print(f"archived {len(entries)} files at {man['revision'][:12]}")
    print(f"uncommitted patch: {len(patch)} bytes")
    return 1 if missing else 0


if __name__ == "__main__":
    raise SystemExit(main())
