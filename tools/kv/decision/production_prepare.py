# SPDX-License-Identifier: MIT
"""Freeze a fixed-scale worker protocol after both complete reference panels.

This prepares files only. It does not allocate a GPU or establish that the
selected optimized backend supports or qualifies either compressed recipe.
Dynamic per-token/head scale storage is outside this launcher's interface.
"""

import argparse
import json
from pathlib import Path
import shutil
import subprocess

from .production import (
    ARMS,
    GEOMETRY,
    MODEL,
    NUMERICAL_BOUNDS,
    RESOURCE_POLICY,
    REVISION,
    SOURCES,
    SUPPORT_POLICY,
    read_inputs,
    sha,
    transferred_scales,
)
from .production_analysis import feasibility
from .production_qualify import CASES, EXTRA_LONG_CASES
from .production_plan import PILOT_RULES


def reference_gate(analysis, evidence, input_digest):
    if set(analysis["models"]) != {"qwen3", "qwen25"} or set(evidence["models"]) != {
        "qwen3",
        "qwen25",
    }:
        raise ValueError("Both complete reference models must precede worker selection")
    for name in ("qwen3", "qwen25"):
        result, audit = analysis["models"][name], evidence["models"][name]
        if any(
            result[key] != audit[key]
            for key in ("model", "revision", "source", "input_sha256")
        ):
            raise ValueError("Reference analysis and evidence identities differ")
        if audit["retained_tokens_decode_exactly"] is not True or audit["counts"] != {
            arm: {"nll": 128, "retrieval": 400, "gsm8k": 1319}
            for arm in ("N", "AF", "RF", "SF", "AD", "RD", "SD")
        }:
            raise ValueError(
                "Incomplete reference evidence cannot select a worker policy"
            )
        if set(result["candidates"]) != {"AF", "RF", "SF", "AD", "RD", "SD"}:
            raise ValueError("All six reference candidate certificates are required")
    qwen = analysis["models"]["qwen3"]
    if (qwen["model"], qwen["revision"], qwen["input_sha256"]) != (
        MODEL,
        REVISION,
        input_digest,
    ):
        raise ValueError(
            "Worker inputs differ from the qualified reference checkpoint/panel"
        )
    return {
        "reference_candidates": {
            arm: {
                objective: qwen["candidates"][arm][objective + "_certificate"]
                for objective in ("preservation", "aggregate_math")
            }
            for arm in ("AF", "SF", "SD")
        },
        "optimized_symmetric_policy": "fixed",
        "dynamic_symmetric_support": "Not expressed by this pinned worker's scalar layer-scale interface; no dynamic worker quality or timing claim",
        "scope": "Fixed-scale deployment comparison. If SD alone qualifies in the reference, its missing paged scale plane remains a separate engineering gap; SF does not stand in for SD.",
    }


def protocol(scales):
    allowance = 36 * 3 * 512
    return {
        "schema": 1,
        "arms": list(ARMS),
        "backend_sources": SOURCES,
        "geometry": GEOMETRY,
        "numerical_bounds": NUMERICAL_BOUNDS,
        "numerical_cases": CASES,
        "numerical_extra_long_cases": EXTRA_LONG_CASES,
        "numerical_output_tokens": 32,
        "query_dtype": "float16",
        "scale_policy": "fixed",
        "symmetric_support_policy": SUPPORT_POLICY,
        "scales": transferred_scales(scales),
        "serving_qualification_route": "aggregate_math",
        "torch": "2.11.0+cu129",
        "transformers": "5.5.3",
        "persistent_budget_bytes": 40 * 1024**3,
        "scale_allocator_allowance_bytes": allowance,
        "equal_slot_blocks": 17000,
        "torch_allocator_reserved_cap_bytes": 68 * 1024**3,
        "resource_policy": RESOURCE_POLICY,
        "serving_pilot_rules": PILOT_RULES,
        "engine": {
            "dtype": "float16",
            "tensor_parallel_size": 1,
            "max_model_len": 10240,
            "max_num_seqs": 96,
            "max_num_batched_tokens": 65536,
            "block_size": 16,
            "enforce_eager": False,
            "disable_log_stats": False,
            "enable_chunked_prefill": False,
            "enable_prefix_caching": False,
            "calculate_kv_scales": False,
            "distributed_executor_backend": "uni",
            "seed": 20261020,
            "stream_interval": 1,
            "generation_config": "vllm",
            "kv_cache_memory_bytes": 40 * 1024**3 - allowance,
            "compilation_config": {
                "mode": 0,
                "cudagraph_mode": "FULL_DECODE_ONLY",
                "cudagraph_capture_sizes": [1, 8, 32, 44, 45, 59, 60, 96],
            },
        },
        "quality_concurrency": 32,
        "eos_token_ids": [151645, 151643],
        "certificates": {
            "family_alpha": 0.025,
            "certificates_per_objective": 4,
            "native_accuracy_floor": 0.5,
            "gsm_gross_loss_margin": 0.02,
            "retrieval_gross_loss_margin": 0.05,
            "nll_relative_mean_margin": 0.01,
            "gsm_net_margin": -0.01,
            "bootstrap_seed": 20261020,
            "bootstrap_resamples": 100000,
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis", type=Path, required=True)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--reference-inputs", type=Path, required=True)
    parser.add_argument("--reference-scales", type=Path, required=True)
    parser.add_argument("--official-model-files", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    read = lambda p: json.loads(p.read_text())
    input_path = args.reference_inputs / "inputs.json.gz"
    analysis, evidence = read(args.analysis), read(args.evidence)
    selection = reference_gate(analysis, evidence, sha(input_path))
    source = Path(__file__).resolve().parents[3]
    if subprocess.check_output(
        ["git", "-C", str(source), "status", "--porcelain", "--", "tools/kv"], text=True
    ).strip():
        raise ValueError(
            "Commit the exact clean worker source before freezing its protocol"
        )
    revision = subprocess.check_output(
        ["git", "-C", str(source), "rev-parse", "HEAD"], text=True
    ).strip()
    names = subprocess.check_output(
        ["git", "-C", str(source), "ls-files", "tools/kv"], text=True
    ).splitlines()
    record = protocol(read(args.reference_scales))
    record.update(
        source_sha=revision,
        runner_files_sha256={
            name: sha(source / name) for name in names if name.endswith((".py", ".txt"))
        },
        reference_selection=selection,
        reference_reports_sha256={
            "analysis": sha(args.analysis),
            "evidence": sha(args.evidence),
        },
        zero_loss_feasibility=feasibility(0.025 / 4),
    )
    args.out.mkdir(parents=True, exist_ok=False)
    files = {
        "inputs.json.gz": input_path,
        "official-weights.json": args.reference_inputs / "official-weights.json",
        "reference-scales.json": args.reference_scales,
        "official-model-files.json": args.official_model_files,
    }
    for name, path in files.items():
        shutil.copyfile(path, args.out / name)
    shutil.copytree(args.reference_inputs / "tokenizer", args.out / "tokenizer")
    record["inputs_sha256"] = {name: sha(args.out / name) for name in files}
    record["tokenizer_files_sha256"] = {
        p.name: sha(p)
        for p in sorted((args.out / "tokenizer").iterdir())
        if p.is_file()
    }
    if (
        record["tokenizer_files_sha256"]
        != evidence["models"]["qwen3"]["tokenizer_files_sha256"]
    ):
        raise ValueError(
            "Worker tokenizer files differ from independently audited reference"
        )
    path = args.out / "protocol.json"
    path.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")
    read_inputs(args.out, sha(path))
    print(
        json.dumps(
            {
                "protocol_sha256": sha(path),
                "source_sha": revision,
                "scope": "Frozen quality protocol only; no production support or launch authorization implied",
            }
        )
    )


if __name__ == "__main__":
    main()
