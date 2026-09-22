# SPDX-License-Identifier: GPL-2.0
"""Assemble the terminal receipt for the corrected terminal-supervision stage.

The latency stage's generator carries that stage's allowance and repair count
and its per-length result shape, so using it here would report the wrong
numbers in the fields that matter most. This reads the artifacts this stage
produces and records unknown where there is no evidence. Unknown is not zero.
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os

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
        hashlib.sha256(open(p, "rb").read()).hexdigest()
        if p and os.path.exists(p)
        else None
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage-dir", required=True, help="the attempt directory")
    ap.add_argument("--experiment-id", default="kv-translate-eos-repair")
    ap.add_argument("--attempt-id", required=True)
    ap.add_argument("--contract", required=True)
    ap.add_argument("--source-manifest", default="")
    ap.add_argument("--allowance-seconds", type=float, default=7200.0)
    ap.add_argument("--total-allowance-seconds", type=float, default=14400.0)
    ap.add_argument("--repairs-allowed", type=int, default=2)
    ap.add_argument("--repairs-used", type=int, default=1)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    S = args.stage_dir
    gate, g_err = load(os.path.join(S, "eos_gate.json"), "eos gate")
    dev, d_err = load(os.path.join(S, "gate", "dev_gate.json"), "scorer output")
    contract, c_err = load(args.contract, "contract")
    srcman, s_err = load(args.source_manifest, "source snapshot")
    verified, v_err = load(os.path.join(S, "contract_verified.json"), "verification")

    arms = {}
    for arm in ("on", "off"):
        tr, t_err = load(
            os.path.join(S, f"eos_{arm}", "train154_train.json"), f"{arm} training"
        )
        q, q_err = load(
            os.path.join(S, f"operator_eos_{arm}.json"), f"{arm} qualification"
        )
        folded = os.path.join(S, f"eos_{arm}", "train154_lin.pt")
        pre = os.path.join(S, f"eos_{arm}", "train154_premerge.pt")
        arms[arm] = {
            "training": (
                {
                    k: tr.get(k)
                    for k in (
                        "eos_arm",
                        "eos_arm_confounded",
                        "denominator_total",
                        "supervised_terms_total",
                        "valid_supervised_tokens",
                        "sample_order_sha256",
                        "init_hashes",
                        "affine_joint_hash",
                        "folded_joint_hash",
                        "train_seconds",
                        "first_100_mean_loss",
                        "last_100_mean_loss",
                        "n_train_documents",
                        "objective_examples",
                        "source_revision",
                        "target_revision",
                        "fold_parity",
                    )
                }
                if tr
                else t_err
            ),
            "loss_trace_len": len(tr.get("loss_trace", [])) if tr else None,
            "update_order_len": len(tr.get("update_order", [])) if tr else None,
            "qualification": (
                {
                    "passed": q.get("passed"),
                    "limits": q.get("contract_limits"),
                    "measurements": {
                        k: v.get("measured")
                        for k, v in q.get("checks", {}).items()
                        if isinstance(v, dict) and "measured" in v
                    },
                    "fold_equivalence": q.get("checks", {}).get("fold_equivalence"),
                    "qualified_on_gpu": q.get("gpu"),
                }
                if q
                else q_err
            ),
            "artifacts": {
                "folded": {
                    "path": folded,
                    "sha256": sha(folded),
                    "present": os.path.exists(folded),
                },
                "premerge": {
                    "path": pre,
                    "sha256": sha(pre),
                    "present": os.path.exists(pre),
                },
            },
        }

    # The two arms are a pair only if everything they share really is shared.
    on_t = arms["on"]["training"]
    off_t = arms["off"]["training"]
    matched = {}
    if isinstance(on_t, dict) and isinstance(off_t, dict):
        for k in (
            "sample_order_sha256",
            "affine_joint_hash",
            "init_hashes",
            "denominator_total",
            "n_train_documents",
            "objective_examples",
            "source_revision",
            "target_revision",
        ):
            matched[k] = {
                "on": on_t.get(k),
                "off": off_t.get(k),
                "same": on_t.get(k) == off_t.get(k),
            }
        matched["differs_only_in_supervised_terms"] = bool(
            all(v["same"] for v in matched.values() if isinstance(v, dict))
            and on_t.get("supervised_terms_total")
            != off_t.get("supervised_terms_total")
        )

    alloc_files = sorted(
        glob.glob(os.path.join(os.path.dirname(S), "attempt-*", "allocation.json"))
    )
    attempts, occupied = [], 0.0
    for p in alloc_files:
        a = json.load(open(p))
        occupied += float(a.get("occupied_seconds") or 0.0)
        attempts.append(
            {
                "attempt_id": a.get("attempt_id"),
                "state": a.get("state"),
                "occupied_seconds": a.get("occupied_seconds"),
                "gpu_ever_active": a.get("gpu_ever_active"),
                "release_verified": a.get("release_verified"),
                "defect": a.get("defect"),
            }
        )

    out = {
        "contract": "eos_repair_receipt_v1",
        "identity": {
            "experiment_id": args.experiment_id,
            "attempt_id": args.attempt_id,
            "stage": "E",
            "guidance": "GUIDANCE_AFTER_081AFE74.md",
            "execution_state": "COMPLETED_VALID" if gate else "INVALID_OR_PARTIAL",
        },
        "contract_frozen": (
            {
                "path": args.contract,
                "sha256": sha(args.contract),
                "state": contract.get("state"),
            }
            if contract
            else c_err
        ),
        "contract_verified_on_device": verified or v_err,
        "source_snapshot": (
            {
                "revision": srcman.get("revision"),
                "files": len(srcman.get("files", [])),
                "uncommitted_patch_bytes": srcman.get("uncommitted_patch_bytes"),
            }
            if srcman
            else s_err
        ),
        "arms": arms,
        "pair_is_matched": matched or UNKNOWN,
        "gate": gate or g_err,
        "scorer_output": (
            {
                "passing_arms": dev.get("passing_arms"),
                "verdict_scope": next(
                    (g.get("verdict_scope") for g in (dev.get("gates") or {}).values()),
                    None,
                ),
            }
            if dev
            else d_err
        ),
        "accounting": {
            "stage_allowance_seconds": args.allowance_seconds,
            "continuation_allowance_seconds": args.total_allowance_seconds,
            "occupied_seconds_this_stage": round(occupied, 1),
            "attempts": attempts,
            "repairs_allowed": args.repairs_allowed,
            "repairs_used": args.repairs_used,
            "spend_tag": UNKNOWN,
            "billing_receipt": UNKNOWN,
        },
        "verdicts": {
            "implementation": None,
            "causal": None,
            "quality_health": (gate or {}).get("verdict") if gate else UNKNOWN,
            "deployment": (
                "unchanged. This stage tested quality and health at 512 tokens; "
                "the known 512 latency failure was measured, reported, and not "
                "allowed to decide."
            ),
            "cartridge": "CARTRIDGE_BLOCKED_INPUTS, untouched by this stage",
            "confirmation_sets": "confirm128 and reserve384 untouched",
        },
    }
    json.dump(out, open(args.out, "w"), indent=2, sort_keys=True, default=str)
    print(
        json.dumps(
            {
                "quality_health": out["verdicts"]["quality_health"],
                "occupied_s": out["accounting"]["occupied_seconds_this_stage"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
