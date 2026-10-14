#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Audit FP32 seed fidelity and exact-neighbor recovery by target length."""

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from evaluate import load_embeddings, ranked_indices


def summarize(rows, seeds, target, document_count, query_count):
    """Require one matching result for every independent seed and tuple."""
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("expected seeds must be nonempty and unique")
    if not 0 < target <= 1:
        raise ValueError("fidelity target must be in (0, 1]")
    groups = defaultdict(list)
    for row in rows:
        if row["document_count"] != document_count or row["query_count"] != query_count:
            raise ValueError("metric denominators differ from the saved inputs")
        fidelity = row["exact_top10_candidate_recall"]
        if not np.isfinite(fidelity) or not 0 <= fidelity <= 1:
            raise ValueError("fidelity must be finite and in [0, 1]")
        groups[(row["repetitions"], row["k_sim"], row["projection_dim"])].append(row)
    if not groups:
        raise ValueError("no matching FP32 exact-rerank measurements")
    spacing = min(np.diff(sorted(seeds))) if len(seeds) > 1 else np.inf
    summaries = []
    for (rep, k, p), group in groups.items():
        if len(group) != len(seeds) or {r["seed"] for r in group} != set(seeds):
            raise ValueError("each tuple must contain exactly every expected seed")
        if spacing < rep:
            raise ValueError("repetition seed ranges overlap")
        if len({(r["dimension"], r["record_bytes"]) for r in group}) != 1:
            raise ValueError("dimensions or record sizes differ across seeds")
        values = [r["exact_top10_candidate_recall"] for r in group]
        summaries.append(
            {
                "repetitions": rep,
                "hash_bits": k,
                "projection_dimensions": p,
                "dimension": group[0]["dimension"],
                "record_bytes_fp32": group[0]["record_bytes"],
                **{
                    f'fidelity_seed_{r["seed"]}': r["exact_top10_candidate_recall"]
                    for r in group
                },
                "min_fidelity": min(values),
                "mean_fidelity": float(np.mean(values)),
                "max_fidelity": max(values),
                "passes_every_seed": min(values) >= target,
            }
        )
    return sorted(summaries, key=lambda r: r["min_fidelity"], reverse=True)


def write_csv(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def analyze(
    run_dir,
    output_dir,
    seeds=(1, 1001, 2001),
    target=0.95,
    candidate_budget=100,
    bootstrap_replicates=1000,
    bootstrap_seed=20261014,
):
    if output_dir.exists():
        raise ValueError("analysis output must be a new directory")
    if candidate_budget < 1 or bootstrap_replicates < 1:
        raise ValueError("candidate budget and bootstrap replicates must be positive")
    report = json.loads((run_dir / "metrics.json").read_text())
    rows = [
        r
        for r in report["metrics"]
        if r["method"] == "google-muvera"
        and r["stage"] == "exact-rerank"
        and r["format"] == "fp32"
        and r["candidate_budget"] == candidate_budget
    ]
    inputs = load_embeddings(run_dir)
    ndoc, nquery = len(inputs["docs"]), len(inputs["queries"])
    summaries = summarize(rows, seeds, target, ndoc, nquery)
    lengths = np.array([len(d) for d in inputs["docs"]])
    scores = np.load(run_dir / "exact_chamfer_scores.npy", mmap_mode="r")
    if scores.shape != (nquery, ndoc) or not np.isfinite(scores).all():
        raise ValueError(
            "exact-score table differs from input dimensions or is nonfinite"
        )
    oracle = np.array([ranked_indices(row, min(10, ndoc)) for row in scores])
    best = summaries[0]
    rep, k, p = (
        best[key] for key in ("repetitions", "hash_bits", "projection_dimensions")
    )
    target_lengths = lengths[oracle]
    bootstrap_indices = np.random.default_rng(bootstrap_seed).integers(
        0, nquery, size=(bootstrap_replicates, nquery)
    )
    target_rows, per_query = [], []
    bin_edges = [0, 32, 64, 128, 256, 512, 1024, max(1025, int(lengths.max()))]
    for seed in seeds:
        name = f"r{rep}-k{k}-p{p}-s{seed}"
        meta = json.loads(
            (run_dir / "representations" / f"{name}-fp32.json").read_text()
        )
        if meta["shape"] != [ndoc, best["dimension"]] or meta["format"] != "fp32":
            raise ValueError("representation metadata differs from the selected tuple")
        docs = np.memmap(
            run_dir / "representations" / meta["payload_file"],
            mode="r",
            dtype="<f4",
            shape=tuple(meta["shape"]),
        )
        qs = np.fromfile(run_dir / f"{name}-queries.f32.bin", dtype="<f4").reshape(
            nquery, meta["shape"][1]
        )
        if not np.isfinite(docs).all() or not np.isfinite(qs).all():
            raise ValueError("nonfinite FDE vectors")
        selected = [ranked_indices(docs @ q, candidate_budget) for q in qs]
        hits = np.array([[d in selected[i] for d in oracle[i]] for i in range(nquery)])
        if abs(float(hits.mean()) - best[f"fidelity_seed_{seed}"]) > 1e-12:
            raise ValueError("recomputed candidate fidelity differs from saved metrics")
        for i in range(nquery):
            per_query.append(
                {
                    "query_id": inputs["query_ids"][i],
                    "query_index": i,
                    "seed": seed,
                    "recovered_exact_top10": int(hits[i].sum()),
                }
            )
        for low, high in zip(bin_edges[:-1], bin_edges[1:]):
            mask = (target_lengths > low) & (target_lengths <= high)
            n, h = mask.sum(1), (mask & hits).sum(1)
            denominator = n.sum()
            denom_boot = n[bootstrap_indices].sum(1)
            valid = denom_boot > 0
            boot = h[bootstrap_indices].sum(1)[valid] / denom_boot[valid]
            target_rows.append(
                {
                    "repetitions": rep,
                    "hash_bits": k,
                    "projection_dimensions": p,
                    "seed": seed,
                    "candidate_budget": candidate_budget,
                    "length_lower_exclusive": low,
                    "length_upper_inclusive": high,
                    "target_neighbors": int(denominator),
                    "contributing_queries": int((n > 0).sum()),
                    "fidelity": float(h.sum() / denominator) if denominator else "",
                    "query_bootstrap_95_low": (
                        float(np.quantile(boot, 0.025)) if len(boot) else ""
                    ),
                    "query_bootstrap_95_high": (
                        float(np.quantile(boot, 0.975)) if len(boot) else ""
                    ),
                    "valid_bootstrap_replicates": int(valid.sum()),
                }
            )
    verdict = {
        "fidelity_target": target,
        "candidate_budget": candidate_budget,
        "document_count": ndoc,
        "query_count": nquery,
        "configuration_count": len(summaries),
        "seeds": list(seeds),
        "passed_configurations": [r for r in summaries if r["passes_every_seed"]],
        "strongest_min_seed_tuple": best,
        "held_out_confirmation": False,
        "ssd_gate_passed": any(r["passes_every_seed"] for r in summaries),
        "length_bootstrap": {
            "unit": "query",
            "replicates": bootstrap_replicates,
            "seed": bootstrap_seed,
            "aggregation": "each exact top10 target document contributes to its own bin",
        },
    }
    output_dir.mkdir(parents=True)
    write_csv(output_dir / "fidelity-summary.csv", summaries)
    write_csv(output_dir / "target-length-diagnostics.csv", target_rows)
    write_csv(output_dir / "best-tuple-per-query.csv", per_query)
    (output_dir / "VERDICT.json").write_text(json.dumps(verdict, indent=2) + "\n")
    return verdict


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 1001, 2001])
    parser.add_argument("--fidelity-target", type=float, default=0.95)
    parser.add_argument("--candidate-budget", type=int, default=100)
    parser.add_argument("--bootstrap-replicates", type=int, default=1000)
    parser.add_argument("--bootstrap-seed", type=int, default=20261014)
    args = parser.parse_args(argv)
    verdict = analyze(
        args.run_dir.expanduser().resolve(),
        args.output_dir.expanduser().resolve(),
        args.seeds,
        args.fidelity_target,
        args.candidate_budget,
        args.bootstrap_replicates,
        args.bootstrap_seed,
    )
    print(json.dumps(verdict, indent=2), flush=True)


if __name__ == "__main__":
    main()
