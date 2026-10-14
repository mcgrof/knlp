"""Three-pair exploratory summaries with censored target crossings."""

import argparse
import json
from pathlib import Path

import numpy as np

from .data import sha


def crossing(curve, threshold, flops):
    for a, b in zip(curve, curve[1:]):
        if a["ce"] <= threshold and b["ce"] <= threshold:
            return {
                "censored": False,
                "step": b["step"],
                "tokens": b["tokens"],
                "training_seconds": b["training_seconds"],
                "end_to_end_seconds": b["end_to_end_seconds"],
                "dense_training_flops_estimate": b["step"] * flops,
                "rule": "confirmed at second consecutive scheduled evaluation",
            }
    return {
        "censored": True,
        "step": None,
        "tokens": None,
        "training_seconds": None,
        "end_to_end_seconds": None,
        "dense_training_flops_estimate": None,
    }


def summary(values):
    return {
        "pairs": values,
        "mean": float(np.mean(values)),
        "range": [min(values), max(values)],
    }


def run(root):
    root = Path(root)
    plan = json.loads((root / "PLAN.json").read_text())
    status = json.loads((root / "STATUS.json").read_text())
    runs = {
        p.parent.name: json.loads(p.read_text())
        for p in (root / "runs").glob("*/RESULT.json")
    }
    result = {
        "plan_sha256": sha(root / "PLAN.json"),
        "tier": plan["tier"],
        "cohort_status": status["status"],
        "cells": {},
        "comparisons": {},
        "missing": [],
        "invalid": [],
        "raw_result_sha256": {},
        "method": "All three paired differences, mean and range; exploratory; no p-value or confidence interval.",
        "flops_convention": "Multiply-add=2; train=3x forward dense linear, attention QK/AV, routing projection. Excludes elementwise norm/gain/CE/Sinkhorn/optimizer; estimate, not measured hardware instructions.",
    }
    for cell in plan["cells"]:
        key = cell["id"]
        finals = []
        pilots = []
        for template in cell["pilots"] + cell["finals"]:
            rid = template["id"]
            if rid not in runs:
                result["missing"].append(rid)
                continue
            r = runs[rid]
            result["raw_result_sha256"][rid] = sha(root / "runs" / rid / "RESULT.json")
            if template["stage"] == "pilot":
                points = (
                    (150, 200, 250) if plan["tier"] == "SCREEN" else (300, 400, 500)
                )
                ce = [v["ce"] for v in r["validation"] if v["step"] in points]
                pilots.append(
                    {
                        "id": rid,
                        "lr": template["lr"],
                        "status": r["status"],
                        "selection_ce": (
                            float(np.mean(ce)) if r["status"] == "complete" else None
                        ),
                        "seconds": r["end_to_end_seconds"],
                    }
                )
                if r["status"] != "complete":
                    result["invalid"].append(rid)
                continue
            if r["status"] != "complete":
                result["invalid"].append(rid)
                finals.append(
                    {"id": rid, "seed": template["seed"], "status": r["status"]}
                )
                continue
            start = 500 if plan["tier"] == "SCREEN" else 1000
            points = [v["ce"] for v in r["validation"] if v["step"] >= start]
            assert len(points) == (template["steps"] - start) // 100 + 1
            finals.append(
                {
                    "id": rid,
                    "seed": template["seed"],
                    "status": r["status"],
                    "primary_ce": float(np.mean(points)),
                    "endpoint_final_ce": r["final_report_ce"],
                    "curve": r["validation"],
                    "parameters": r["identities"]["parameters"],
                    "peak_allocated_bytes": r["peak_allocated_bytes"],
                    "training_seconds": r["training_seconds"],
                    "end_to_end_seconds": r["end_to_end_seconds"],
                    "targets": {
                        str(t): crossing(
                            r["validation"], t, r["optimizer_step_flops_estimate"]
                        )
                        for t in (3.0, 4.0)
                    },
                }
            )
        good = [r for r in finals if r["status"] == "complete"]
        result["cells"][key] = {
            "pilots": pilots,
            "finals": finals,
            "selected_lr": status["selected_lrs"].get(key),
            "primary_ce": (
                summary([r["primary_ce"] for r in good]) if len(good) == 3 else None
            ),
            "endpoint_final_ce": (
                summary([r["endpoint_final_ce"] for r in good])
                if len(good) == 3
                else None
            ),
            "parameters": good[0]["parameters"] if good else None,
            "tuning_seconds": sum(r["seconds"] for r in pilots),
        }

    def paired(a, b):
        aa = {
            r["seed"]: r
            for r in result["cells"][a]["finals"]
            if r["status"] == "complete"
        }
        bb = {
            r["seed"]: r
            for r in result["cells"][b]["finals"]
            if r["status"] == "complete"
        }
        if set(aa) != set(plan["final_seeds"]) or set(bb) != set(plan["final_seeds"]):
            return {"status": "incomplete_pairs", "nomination": False}
        primary = [
            aa[s]["primary_ce"] - bb[s]["primary_ce"] for s in plan["final_seeds"]
        ]
        endpoint = [
            aa[s]["endpoint_final_ce"] - bb[s]["endpoint_final_ce"]
            for s in plan["final_seeds"]
        ]
        speeds = {}
        for threshold in ("3.0", "4.0"):
            speeds[threshold] = []
            for seed in plan["final_seeds"]:
                left, right = (
                    aa[seed]["targets"][threshold],
                    bb[seed]["targets"][threshold],
                )
                item = {"seed": seed, "a": left, "b": right, "ratios": None}
                if not left["censored"] and not right["censored"]:
                    item["ratios"] = {
                        k: left[k] / right[k] if right[k] else None
                        for k in (
                            "step",
                            "tokens",
                            "training_seconds",
                            "end_to_end_seconds",
                            "dense_training_flops_estimate",
                        )
                    }
                speeds[threshold].append(item)
        return {
            "status": "three_pairs",
            "a_minus_b_primary": summary(primary),
            "a_minus_b_endpoint": summary(endpoint),
            "targets": speeds,
            "directional_nomination": all(v < 0 for v in primary)
            and float(np.mean(endpoint)) <= 0,
        }

    for d in (4, 16, 32):
        for a, b in (("M", "R"), ("L", "R"), ("ML", "M"), ("ML", "L")):
            result["comparisons"][f"{a}-{b}-D{d}"] = paired(f"{a}-D{d}", f"{b}-D{d}")
        cs = result["cells"]
        if all(cs[f"{a}-D{d}"]["primary_ce"] for a in ("R", "L", "M", "ML")):
            values = [
                cs[f"ML-D{d}"]["primary_ce"]["pairs"][i]
                - cs[f"M-D{d}"]["primary_ce"]["pairs"][i]
                - cs[f"L-D{d}"]["primary_ce"]["pairs"][i]
                + cs[f"R-D{d}"]["primary_ce"]["pairs"][i]
                for i in range(3)
            ]
            result["comparisons"][f"interaction-D{d}"] = summary(values)
    result["combination_nomination_D32"] = all(
        result["comparisons"][f"ML-{b}-D32"].get("directional_nomination", False)
        for b in ("M", "L")
    )
    if plan["tier"] == "CORE_CHANNELS":
        for a, b in (
            ("ML-O", "ML"),
            ("ML-O", "ML-S"),
            ("ML-O", "ML-C"),
            ("M-O", "M"),
            ("ML-O", "M-O"),
        ):
            result["comparisons"][f"{a}-{b}-D32"] = paired(f"{a}-D32", f"{b}-D32")
        result["ordered_nomination"] = all(
            result["comparisons"][f"ML-O-{b}-D32"].get("directional_nomination", False)
            for b in ("ML", "ML-S", "ML-C")
        )
    else:
        result["ordered_nomination"] = None
    (root / "ANALYSIS.json").write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {
                "cohort_status": result["cohort_status"],
                "missing": len(result["missing"]),
                "invalid": len(result["invalid"]),
                "combination_nomination_D32": result["combination_nomination_D32"],
                "ordered_nomination": result["ordered_nomination"],
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True)
    run(p.parse_args().root)
