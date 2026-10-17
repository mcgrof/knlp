# SPDX-License-Identifier: MIT
"""Recompute observed predecessor curves without extending or equating clocks."""

import argparse
from pathlib import Path
import statistics
from research.layerrope_beta.time_budget.protocol import freeze
from research.mo_variants.stage_c_common import read, sha, write


def summarize(runs, out):
    p = freeze()
    records = []
    for path in sorted(Path(runs).glob("final-*/RESULT.json")):
        result = read(path)
        job = result["job"]
        if job["arm"] not in p["rows"] or job.get("beta") not in (None, 0.0):
            continue
        quant_path = path.parent / "QUANTIZATION.json"
        quant = read(quant_path)
        seconds = {v["step"]: v["seconds"] for v in result["steps"]}
        cumulative = {0: 0.0}
        total = 0.0
        for step in sorted(seconds):
            total += seconds[step]
            cumulative[step] = total
        curves = [
            dict(
                step=v["step"],
                tokens=v["tokens"],
                ce=v["ce"],
                observed_training_seconds=cumulative[v["step"]],
                original_process_seconds=v["seconds"],
            )
            for v in result["curve"]
        ]
        cache = {v["name"]: v for v in quant["cache"]}
        weights = {v["name"]: v for v in quant["weight_activation_residual"]}
        # Actual prior timings on the old data/runtime; not calibrated new-path estimates.
        endpoint = sum(cache[c["name"]]["seconds"] for c in p["cache_cases"])
        endpoint += sum(weights[w["name"]]["seconds"] for w in p["weight_policies"])
        endpoint += sum(cache[n]["seconds"] for n in p["supporting_cache_names"])
        records.append(
            dict(
                row=job["arm"],
                seed=job["seed"],
                result=dict(path=str(path), sha256=sha(path)),
                quantization=dict(path=str(quant_path), sha256=sha(quant_path)),
                curves=curves,
                clean_ce=quant["clean"]["ce"],
                training_seconds=total,
                absolute_k3v2_ce=cache["symmetric-K3V2"]["ce"],
                v_bit_penalty=cache["symmetric-K3V2"]["ce"]
                - cache["symmetric-K3V3"]["ce"],
                reduced_20_evaluation_seconds_historical=endpoint,
                clean_full_evaluation_seconds_historical=cache["clean"]["seconds"],
            )
        )
    expected = {(r, s) for r in p["rows"] for s in (77311, 77312, 77313)}
    if len(records) != 12 or {(r["row"], r["seed"]) for r in records} != expected:
        raise ValueError("Incomplete historical core")
    summary = []
    for row in p["rows"]:
        data = [r for r in records if r["row"] == row]
        summary.append(
            dict(
                row=row,
                **{
                    k: statistics.mean(r[k] for r in data)
                    for k in (
                        "clean_ce",
                        "training_seconds",
                        "absolute_k3v2_ce",
                        "v_bit_penalty",
                        "reduced_20_evaluation_seconds_historical",
                        "clean_full_evaluation_seconds_historical",
                    )
                }
            )
        )
    result = dict(
        scope="Artifact-only three-seed fixed-token observations; no extrapolation, continuation or new clock equivalence",
        source_measured="17035f8402a7775c3f96f8c16d7a29281398155d",
        summary=summary,
        records=records,
    )
    write(out, result)
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--runs", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    summarize(a.runs, a.out)
