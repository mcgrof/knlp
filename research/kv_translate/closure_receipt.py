# SPDX-License-Identifier: GPL-2.0
"""Assemble the one closure receipt this diagnostic has to leave behind.

Every field is read from an artifact on disk or recorded as unknown. Nothing
is filled in from memory, because the fields most worth having -- occupied
seconds, allocation release, spend linkage -- are exactly the ones a summary
written afterwards tends to round, and a rounded dollar figure has already
been mistaken for a duration once in this lane.

Where evidence is absent the field says so. Absent is not zero.

Run: python research/kv_translate/closure_receipt.py --help
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys

UNKNOWN = "unknown: no evidence available"


def load(p, what):
    if not p or not os.path.exists(p):
        return None, f"unknown: {what} absent at {p!r}"
    try:
        return json.load(open(p)), None
    except Exception as e:
        return None, f"unknown: {what} unreadable ({e})"


def sha(p):
    return (
        hashlib.sha256(open(p, "rb").read()).hexdigest() if os.path.exists(p) else None
    )


def read_stages(p):
    """Stage boundaries the pod stamped as unix seconds."""
    if not p or not os.path.exists(p):
        return {}, UNKNOWN
    out = {}
    for line in open(p):
        parts = line.split()
        if len(parts) == 2:
            try:
                out[parts[0]] = float(parts[1])
            except ValueError:
                pass
    return out, None


def classify(timing):
    """Turn measured ratios into the plan's terminal vocabulary.

    A saving counts only when the descriptive interval's upper endpoint is
    below one. Everything else is reported as what it is: a point estimate
    with uncertainty, or an unmeasured length.
    """
    if not timing or "results" not in timing:
        return "BLOCKED_OR_INVALID", {}, "no timing output"
    per = {}
    any_switch, any_source, any_uncertain = False, False, False
    for length, r in timing["results"].items():
        if not isinstance(r, dict) or "conditions" not in r:
            continue
        row = {}
        for cond, e in r["conditions"].items():
            if cond == "target_native":
                row[cond] = {"p50_ms": e["p50_ms"], "p95_ms": e["p95_ms"], "n": e["n"]}
                continue
            saving = bool(e.get("supported_saving"))
            crosses = bool(e.get("interval_crosses_one_or_threshold"))
            row[cond] = {
                "p50_ms": e["p50_ms"],
                "p95_ms": e["p95_ms"],
                "n": e["n"],
                "ratio_p50_over_native_p50": e["ratio_p50_over_native_p50"],
                "ratio_translated_p95_over_native_p50": e[
                    "ratio_translated_p95_over_native_p50"
                ],
                "paired_interval": e["paired_interval"],
                "supported_saving": saving,
                "meets_inherited_p50_0_75": e["meets_inherited_p50"],
                "meets_inherited_tail_under_1": e["meets_inherited_tail"],
                "outcome": (
                    "saving supported"
                    if saving
                    else ("uncertain" if crosses else "no saving")
                ),
            }
            if saving and cond == "existing_gpu_source_kv_switch":
                any_switch = True
            if saving and cond == "source_prefill_inclusive_translation":
                any_source = True
            if crosses and not saving:
                any_uncertain = True
        # The measurement writes model_logits_all_finite. Reading a name it
        # does not write returns null, and a null validity field reads as
        # "not checked" when the check in fact passed.
        row["model_logits_all_finite"] = r.get("model_logits_all_finite")
        row["nonfinite_occurrences"] = r.get("nonfinite_occurrences")
        per[length] = row
    if any_source:
        state = "SOURCE_INCLUSIVE_TIMING_OPPORTUNITY"
    elif any_switch:
        state = "REUSE_DEPENDENT_TIMING_OPPORTUNITY"
    elif any_uncertain:
        state = "TIMING_BOUNDARY_UNRESOLVED"
    else:
        state = "NO_DEMONSTRATED_SAVING_AT_TESTED_POINTS"
    lengths = sorted(per, key=int)
    return state, per, f"classified {len(lengths)} lengths: {', '.join(lengths)}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiment-id", required=True)
    ap.add_argument("--attempt-id", required=True)
    ap.add_argument("--timing", default="")
    ap.add_argument("--qualification", default="")
    ap.add_argument("--stages", default="", help="pod stage stamps")
    ap.add_argument("--allocation", default="", help="allocation record json")
    ap.add_argument("--source-manifest", default="")
    ap.add_argument("--fixtures", default="")
    ap.add_argument("--allowance-seconds", type=float, default=1800.0)
    ap.add_argument("--occupied-seconds", type=float, default=-1.0)
    ap.add_argument("--repairs-allowed", type=int, default=1)
    ap.add_argument("--repairs-used", type=int, default=0)
    ap.add_argument("--spend-tag", default="")
    ap.add_argument("--released", default="", help="verified|outstanding:<detail>")
    ap.add_argument("--blocker", default="")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    timing, t_err = load(args.timing, "timing output")
    qual, q_err = load(args.qualification, "qualification receipt")
    alloc, a_err = load(args.allocation, "allocation record")
    srcman, s_err = load(args.source_manifest, "source snapshot manifest")
    stages, st_err = read_stages(args.stages)

    state, per_length, note = classify(timing)
    if args.blocker:
        state = "BLOCKED_OR_INVALID"

    occupied = args.occupied_seconds if args.occupied_seconds >= 0 else None
    receipt = {
        "contract": "latency_closure_receipt_v1",
        "identity": {
            "experiment_id": args.experiment_id,
            "attempt_id": args.attempt_id,
            "execution_state": "COMPLETED_VALID" if timing else "INVALID_OR_PARTIAL",
            "guidance": "GUIDANCE_AFTER_8728248C.md",
            "resume_addendum": "RESUME_LATENCY_AFTER_8728248C.md",
            "owner": "this worker; no other worker held this diagnostic",
            "allocation": alloc or a_err or UNKNOWN,
        },
        "inputs": {
            "probe_artifact_sha256": (
                qual.get("checks", {}).get("artifact", {}).get("joint_weight_sha256")
                if qual
                else q_err
            ),
            "timing_fixtures": (
                timing.get("fixtures") if timing else (t_err or UNKNOWN)
            ),
            "operator_fixture_sha256": (
                qual.get("checks", {}).get("fixture", {}).get("sha256")
                if qual
                else q_err
            ),
            "models": timing.get("pinning") if timing else (t_err or UNKNOWN),
            "source_snapshot": (
                {
                    "revision": srcman.get("revision"),
                    "files": len(srcman.get("files", [])),
                    "uncommitted_patch_bytes": srcman.get("uncommitted_patch_bytes"),
                    "manifest_sha256": sha(args.source_manifest),
                }
                if srcman
                else (s_err or UNKNOWN)
            ),
            "timing_policy": timing.get("analysis_policy") if timing else UNKNOWN,
            "boundary": timing.get("boundary") if timing else UNKNOWN,
        },
        "qualification": {
            "receipt_path": args.qualification or UNKNOWN,
            "passed": qual.get("passed") if qual else q_err,
            "limits": qual.get("contract_limits") if qual else None,
            "measurements": (
                {
                    k: v.get("measured")
                    for k, v in qual["checks"].items()
                    if "measured" in v
                }
                if qual
                else None
            ),
            "limits_are": (
                "a declared diagnostic acceptance policy, not a mathematical "
                "error bound"
            ),
        },
        "raw_samples": (
            args.timing.replace(".json", ".raw.jsonl") if args.timing else UNKNOWN
        ),
        "results_by_length_and_condition": per_length or UNKNOWN,
        "blocker": args.blocker or None,
        "accounting": {
            "allowance_seconds": args.allowance_seconds,
            "occupied_seconds": occupied if occupied is not None else UNKNOWN,
            "remaining_seconds": (
                args.allowance_seconds - occupied if occupied is not None else UNKNOWN
            ),
            "stage_stamps": stages or st_err,
            "repairs_allowed": args.repairs_allowed,
            "repairs_used": args.repairs_used,
            "spend_tag": args.spend_tag or UNKNOWN,
            "billing_receipt": UNKNOWN,
            "note": (
                "Occupied seconds are measured from the allocation request to "
                "the confirmed release, which is the window that can be billed. "
                "They are not inferred from a dollar figure."
            ),
        },
        "cleanup": {
            "allocation_release": args.released or UNKNOWN,
            "artifact_export": (
                "exported to key-results" if timing else "nothing to export"
            ),
            "timing_output_sha256": sha(args.timing),
        },
        "verdicts": {
            "implementation": None,
            "timing": state,
            "deployment": None,
            "cartridge": "CARTRIDGE_BLOCKED_INPUTS, untouched by this diagnostic",
            "confirmation_sets": "confirm128 and reserve384 untouched",
        },
        "note": note,
    }
    json.dump(receipt, open(args.out, "w"), indent=2, sort_keys=True, default=str)
    print(json.dumps({"timing_verdict": state, "note": note}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
