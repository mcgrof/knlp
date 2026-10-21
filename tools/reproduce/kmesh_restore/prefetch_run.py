#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Qualify and compare read-once demand with early loading in paired blocks."""

from __future__ import annotations

import argparse
import importlib
import json
import os
from pathlib import Path
import sys
import time

from prefetch_campaign import export_csv, verify_capture
from prefetch_verify import verify_block
from restore_common import (
    RETAINED_DEMAND_COMMIT,
    VLLM_COMMIT,
    sha256,
    source_identity,
    write_json,
)
from run import HERE, execute, launch, serving, stop


def freeze_sources(args):
    """Verify sources, model bytes and every linked native binary before use."""
    sources = {
        "vllm": source_identity(args.vllm_source, VLLM_COMMIT),
        "lmcache": source_identity(args.lmcache_source, args.lmcache_commit),
    }
    model_files = json.loads((HERE / "model-files.json").read_text())
    for name, expected in model_files.items():
        if sha256(args.model / name) != expected:
            raise RuntimeError(f"pinned model artifact differs: {name}")
    native = {}
    for package in (args.vllm_source / "vllm", args.lmcache_source / "lmcache"):
        for binary in sorted(package.rglob("*.so")):
            native[str(binary)] = sha256(binary)
    modules = ["lmcache_rust_raw_block_io"]
    if args.gpu_provider == "vulkan_rm":
        modules.append("lmcache._vulkan_rm")
    for module in modules:
        imported = importlib.import_module(module)
        binary = Path(imported.__file__)
        native[str(binary)] = sha256(binary)
    if not native:
        raise RuntimeError("native library manifest is empty")
    return {
        "sources": sources,
        "model_files": model_files,
        "harness": {p.name: sha256(p) for p in sorted(HERE.glob("*.py"))},
        "native": native,
        "python": sys.version,
        "gpu_provider": args.gpu_provider,
    }


def compatible(old, new):
    """Reject an admission receipt from a different source or native build."""
    for key in (
        "sources",
        "model_files",
        "harness",
        "native",
        "python",
        "gpu_provider",
    ):
        if old[key] != new[key]:
            raise ValueError(f"admission provenance changed: {key}")


def main(args):
    """Run only the declared phase and retain all trials before analysis."""
    for name in ("output", "model", "vllm_source", "lmcache_source"):
        setattr(args, name, getattr(args, name).resolve())
    args.output.mkdir()
    started = time.time_ns()
    os.environ.update(
        OMP_NUM_THREADS="2",
        MKL_NUM_THREADS="2",
        HF_HUB_OFFLINE="1",
        LMCACHE_TRACK_USAGE="false",
        VLLM_WORKER_MULTIPROC_METHOD="spawn",
        VLLM_BATCH_INVARIANT="1",
        PYTHONHASHSEED="0",
    )
    os.environ["PYTHONPATH"] = os.pathsep.join(
        [
            str(args.lmcache_source),
            str(args.vllm_source),
            os.environ.get("PYTHONPATH", ""),
        ]
    )
    provenance = freeze_sources(args)
    write_json(args.output / "provenance.json", provenance)
    selected = None
    admitted = None
    if args.phase != "qualification":
        if args.qualified is None:
            raise ValueError("pilot and confirmation require a qualified capture")
        admitted = json.loads((args.qualified / "RESULT.json").read_text())
        if admitted["phase"] != "qualification" or not admitted["accepted"]:
            raise ValueError("native qualification was not accepted")
        compatible(
            json.loads((args.qualified / "provenance.json").read_text()), provenance
        )
    if args.phase == "confirmation":
        if args.pilot is None:
            raise ValueError("confirmation requires the frozen complete pilot")
        pilot = json.loads((args.pilot / "RESULT.json").read_text())
        if pilot["phase"] != "pilot" or not pilot["accepted"]:
            raise ValueError("pilot was not accepted")
        compatible(json.loads((args.pilot / "provenance.json").read_text()), provenance)
        selected = pilot["selection"]["selected_lead_ms"]
    for admission in (
        args.qualified,
        args.pilot if args.phase == "confirmation" else None,
    ):
        if admission is not None:
            recomputed, _ = verify_capture(admission)
            claimed = json.loads((admission / "RESULT.json").read_text())
            if recomputed != claimed:
                raise ValueError(
                    "admission result differs from raw offline verification"
                )
    import torch

    runtime = {
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(),
        "capability": list(torch.cuda.get_device_capability()),
        "kernel": os.uname().release,
        "driver_version": Path("/sys/module/nvidia/version").read_text().strip(),
    }
    write_json(args.output / "runtime.json", runtime)
    for admission in (
        args.qualified,
        args.pilot if args.phase == "confirmation" else None,
    ):
        if admission is not None:
            if json.loads((admission / "runtime.json").read_text()) != runtime:
                raise ValueError("admitted runtime, driver or device changed")
    blocks = {"qualification": 1, "pilot": 2, "confirmation": 10}[args.phase]
    samples = {"qualification": 1, "pilot": 5, "confirmation": 20}[args.phase]
    fanouts = (4, 8) if args.phase == "confirmation" else (4,)
    config = {
        "model": str(args.model),
        "gpu_provider": args.gpu_provider,
        "phase": args.phase,
        "fanouts": list(fanouts),
        "arena_bytes": 512 << 20,
        "decode_tokens": 8,
        "external_singleflight": True,
        "family_prefetch": True,
        "retained_fill": True,
        "max_num_seqs": 9,
        "selected_lead_ms": selected,
        "oracle": "../oracle/oracle.json",
    }
    write_json(
        args.output / "CONFIG.json",
        config
        | {
            "blocks_per_path_fanout": blocks,
            "measured_groups_per_arm_block": samples,
            "warmups_per_arm_block": 3,
            "arrival_offset_ms": 250,
            "late_hint_rule": "Skip an unsubmitted engine warm after child submission; drain accepted HTTP work.",
            "lead_selection": "Shortest common lead with >=9/10 scheduled-arrival readiness per path; fallback 250 ms.",
            "estimator": "Equal-weight block arm means; paired block bootstrap, 10000 draws, seed 20261020.",
            "exclusions": "None; any failed correctness or incomplete grid rejects the campaign.",
            "stopping_rule": "One fixed grid; no favorable-outcome extension.",
        },
    )
    config_path = args.output / "config.json"
    write_json(config_path, config)
    if args.phase == "qualification":
        for mode in ("staged", "direct"):
            log = args.output / f"corruption-{mode}.log"
            process = launch(
                "adapter.py",
                [
                    str(config_path),
                    mode,
                    str(args.output / f"corruption-{mode}"),
                    "--samples",
                    "1",
                    "--corrupt",
                ],
                log,
            )
            try:
                code = process.wait(timeout=180)
                if (
                    code == 0
                    or "restored GPU consumer bytes differ" not in log.read_text()
                ):
                    raise RuntimeError("corruption control missed exact-byte rejection")
            finally:
                stop(process)
            write_json(
                args.output / f"corruption-{mode}-rejected.json",
                {
                    "returncode": code,
                    "expected_payload_rejection": True,
                },
            )
    oracle = args.output / "oracle"
    oracle.mkdir()
    oracle_config = oracle / "config.json"
    write_json(oracle_config, config | {"output": str(oracle), "serving_samples": 1})
    execute("worker.py", [str(oracle_config), "oracle"], oracle / "oracle.log")
    receipts, identities = [], set()
    try:
        for block in range(blocks):
            for fanout in fanouts:
                modes = ("staged", "direct") if block % 2 == 0 else ("direct", "staged")
                for mode in modes:
                    block_id = f"{args.phase}-{mode}-n{fanout}-b{block:02d}"
                    cell = args.output / block_id
                    serving(
                        config
                        | {
                            "mode": mode,
                            "fanouts": [fanout],
                            "block_index": block,
                            "block_id": block_id,
                        },
                        cell,
                        samples,
                        "prefetch_worker.py",
                    )
                    process = json.loads((cell / "consumer-process.json").read_text())
                    identity = (
                        process["boot_id"],
                        process["pid"],
                        process["start_ticks"],
                    )
                    if identity in identities or process["block_id"] != block_id:
                        raise ValueError("engine process identity was reused")
                    identities.add(identity)
                    receipt = verify_block(cell)
                    receipts.append(
                        {
                            "block_id": block_id,
                            "verification": {
                                k: v for k, v in receipt.items() if k != "rows"
                            },
                        }
                    )
                    write_json(args.output / "blocks.json", receipts)
                    print("VERIFIED", block_id, flush=True)
        result, verified_rows = verify_capture(args.output)
        export_csv(args.output / "RESULTS.csv", verified_rows)
        write_json(args.output / "RESULT.json", result)
    except BaseException as error:
        write_json(
            args.output / "FAILURE.json",
            {
                "error_type": type(error).__name__,
                "error": str(error),
                "completed_blocks": len(receipts),
                "accepted": False,
            },
        )
        raise
    finally:
        write_json(
            args.output / "elapsed.json",
            {
                "start_unix_ns": started,
                "end_unix_ns": time.time_ns(),
                "phase": args.phase,
            },
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--phase", choices=("qualification", "pilot", "confirmation"), required=True
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--vllm-source", type=Path, required=True)
    parser.add_argument("--lmcache-source", type=Path, required=True)
    parser.add_argument("--lmcache-commit", default=RETAINED_DEMAND_COMMIT)
    parser.add_argument(
        "--gpu-provider", choices=("native", "vulkan_rm"), default="vulkan_rm"
    )
    parser.add_argument("--qualified", type=Path)
    parser.add_argument("--pilot", type=Path)
    main(parser.parse_args())
