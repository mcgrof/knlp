# SPDX-License-Identifier: GPL-2.0
"""Refuse to rent a machine for a cartridge pilot whose inputs are not there.

The pilot compares three initialisers for receiver training, and every one of
them needs something the archive may not contain. The failure this exists to
prevent is the ordinary one: a machine is provisioned, weights download, and
the run stops on an input that was absent before any of it started.

It also exists to prevent a subtler one. A natural-prefix truncation -- a
document run through a model with the first p tokens of its key/value cache
kept -- has the shape of a cartridge and is not one. Earlier artifacts in this
archive were exactly that, and a checker that tested for a file of the right
shape would have passed them. So a source payload is required to carry
positive evidence that an optimiser touched it.

Run: python research/kv_translate/cartridge_preflight.py --manifest <p>
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys

# Every item the plan's manifest must cover. A pilot missing any one of these
# cannot produce the comparison it is for.
REQUIRED = (
    "source_model_identity",
    "receiver_model_identity",
    "tokenizer_identity",
    "learned_source_cartridge",
    "original_untrained_source_state",
    "shared_capability_content",
    "train_dev_roles",
    "prefix_and_positional_contract",
    "kv_geometry_and_dtype",
    "runnable_trainer",
    "evaluator",
    "receiver_native_quality_reference",
)

# Items whose mere presence is not enough.
NEEDS_EVIDENCE = {
    "learned_source_cartridge": "evidence_of_optimisation",
    "receiver_native_quality_reference": "measured_positive_gain",
    "runnable_trainer": "performs_backward_and_step",
}


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    try:
        man = json.load(open(args.manifest))
    except Exception as e:
        print(f"CARTRIDGE_BLOCKED_INPUTS: manifest unreadable: {e}", file=sys.stderr)
        return 1

    items = man.get("items", {})
    missing, unverified, present = [], [], {}

    for req in REQUIRED:
        rec = items.get(req)
        if not rec or rec.get("status") != "qualified":
            missing.append(
                {
                    "requirement": req,
                    "status": (rec or {}).get("status", "absent"),
                    "what_is_there": (rec or {}).get("what_it_actually_is"),
                }
            )
            continue
        path = rec.get("path")
        if path and not os.path.exists(path):
            missing.append(
                {"requirement": req, "status": "path-does-not-exist", "path": path}
            )
            continue
        # A qualified status is a claim. Where the claim is the kind that has
        # been wrong before, demand the field that substantiates it.
        need = NEEDS_EVIDENCE.get(req)
        if need and not rec.get(need):
            unverified.append(
                {
                    "requirement": req,
                    "missing_evidence_field": need,
                    "why": (
                        "a payload of the right shape is not evidence of the "
                        "thing that produced it"
                    ),
                }
            )
            continue
        present[req] = {
            "path": path,
            "sha256": sha(path) if path and os.path.isfile(path) else None,
        }

    ok = not missing and not unverified
    report = {
        "contract": "cartridge_preflight_v1",
        "manifest": os.path.abspath(args.manifest),
        "required": list(REQUIRED),
        "qualified": present,
        "missing": missing,
        "present_but_unsubstantiated": unverified,
        "passed": ok,
        "state": "READY_FOR_PILOT" if ok else "CARTRIDGE_BLOCKED_INPUTS",
        "note": (
            "This gates a rental. It does not judge whether the pilot would "
            "succeed, only whether it could run at all."
        ),
    }
    if args.out:
        json.dump(report, open(args.out, "w"), indent=2, sort_keys=True)

    for req in REQUIRED:
        mark = (
            "ok  "
            if req in present
            else (
                "EVID" if any(u["requirement"] == req for u in unverified) else "MISS"
            )
        )
        print(f"  {mark} {req}")
    if ok:
        print("READY_FOR_PILOT: every required input is present and substantiated")
        return 0
    print(
        f"CARTRIDGE_BLOCKED_INPUTS: {len(missing)} absent, "
        f"{len(unverified)} present but unsubstantiated",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
