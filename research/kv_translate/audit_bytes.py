# SPDX-License-Identifier: GPL-2.0
"""Count what a translator actually occupies, by looking at its tensors.

The byte figures this lane has been quoting were computed from a parameter
count times an assumed two bytes each. That is an estimate of what a
deployment *could* store, not a measurement of what was stored or of what the
timed apply path held, and the two diverge: a residual network accounted at
31,266,816 bytes occupies 62,836,999 on disk, which is the signature of an
assumed half-precision tensor that was written in single.

A byte claim has to name which quantity it is. Four are counted here and kept
apart. The serialized container is what the file costs, overhead included. The
tensor payload is the sum of `numel * itemsize` over the tensors actually
present, by dtype, which is what a reader would have to hold. The serving
payload is the same sum after casting to the precision the apply path runs at,
which is what a deployment would hold. And the parameter-count estimate is the
figure previously reported, retained so the difference is visible rather than
quietly corrected.

Env: none beyond torch; CPU only.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from collections import Counter

import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

MIB = float(2**20)


def tensor_stats(obj, acc=None):
    """Walk an arbitrary loaded structure and total its tensors by dtype."""
    acc = acc if acc is not None else Counter()
    if torch.is_tensor(obj):
        acc[str(obj.dtype)] += obj.numel()
    elif isinstance(obj, dict):
        for v in obj.values():
            tensor_stats(v, acc)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            tensor_stats(v, acc)
    return acc


ITEMSIZE = {
    "torch.float64": 8,
    "torch.float32": 4,
    "torch.bfloat16": 2,
    "torch.float16": 2,
    "torch.int64": 8,
    "torch.int32": 4,
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms-dir", required=True)
    ap.add_argument("--h1-json", default="")
    ap.add_argument(
        "--serving-dtype",
        default="float32",
        help="the precision the measured apply path ran at",
    )
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    serving_item = ITEMSIZE["torch." + args.serving_dtype]
    reported = {}
    if args.h1_json:
        reported = json.load(open(args.h1_json)).get("arm_bytes", {})

    arms = {}
    for path in sorted(glob.glob(os.path.join(args.arms_dir, "*.pt"))):
        if path.endswith(".residual.pt"):
            continue
        name = os.path.basename(path)[:-3]
        blob = torch.load(path, map_location="cpu", weights_only=False)
        counts = tensor_stats(blob.get("blocks", blob))
        rec = {
            "affine_file": path,
            "affine_serialized_bytes": os.path.getsize(path),
            "affine_by_dtype": {
                k: {"numel": v, "bytes": v * ITEMSIZE.get(k, 0)}
                for k, v in sorted(counts.items())
            },
            "affine_payload_bytes": sum(
                v * ITEMSIZE.get(k, 0) for k, v in counts.items()
            ),
            "affine_numel": sum(counts.values()),
            "affine_serving_bytes": sum(counts.values()) * serving_item,
        }
        res = path[:-3] + ".residual.pt"
        if os.path.exists(res):
            rblob = torch.load(res, map_location="cpu", weights_only=False)
            rc = tensor_stats(rblob)
            rec.update(
                {
                    "residual_file": res,
                    "residual_serialized_bytes": os.path.getsize(res),
                    "residual_by_dtype": {
                        k: {"numel": v, "bytes": v * ITEMSIZE.get(k, 0)}
                        for k, v in sorted(rc.items())
                    },
                    "residual_payload_bytes": sum(
                        v * ITEMSIZE.get(k, 0) for k, v in rc.items()
                    ),
                    "residual_numel": sum(rc.values()),
                    "residual_serving_bytes": sum(rc.values()) * serving_item,
                    "residual_linear": bool(rblob.get("linear")),
                    "residual_hidden": rblob.get("hidden"),
                }
            )
        else:
            for k in (
                "residual_serialized_bytes",
                "residual_payload_bytes",
                "residual_numel",
                "residual_serving_bytes",
            ):
                rec[k] = 0

        rec["total_serialized_bytes"] = (
            rec["affine_serialized_bytes"] + rec["residual_serialized_bytes"]
        )
        rec["total_serving_bytes"] = (
            rec["affine_serving_bytes"] + rec["residual_serving_bytes"]
        )
        # The name the run used has a plus where the filename has an underscore.
        key = name.replace("_lin", "+lin").replace("_mlp", "+mlp")
        rec["reported_total_bytes"] = reported.get(key, {}).get("total_bytes")
        arms[key] = rec

    ceiling = 117_497_856
    print(
        "%-14s %10s %10s %10s %10s %7s"
        % ("arm", "reported", "payload", "serving", "on disk", "<=cap")
    )
    for k, r in sorted(arms.items()):
        rep = r["reported_total_bytes"]
        print(
            "%-14s %9s %10.1f %10.1f %10.1f %7s"
            % (
                k,
                ("%.1f" % (rep / MIB)) if rep else "-",
                (r["affine_payload_bytes"] + r["residual_payload_bytes"]) / MIB,
                r["total_serving_bytes"] / MIB,
                r["total_serialized_bytes"] / MIB,
                "yes" if r["total_serving_bytes"] <= ceiling else "no",
            )
        )

    out = {
        "serving_dtype": args.serving_dtype,
        "serving_itemsize": serving_item,
        "tensor_payload_ceiling_bytes": ceiling,
        "arms": arms,
        "note": (
            "payload is the sum over tensors as stored; serving is the same "
            "tensor count at the precision the measured apply ran at; on disk "
            "is the container including overhead; reported is the earlier "
            "parameter-count estimate at an assumed two bytes each"
        ),
    }
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2, sort_keys=True)
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
