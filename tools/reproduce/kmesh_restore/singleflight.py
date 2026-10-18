#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Compare exact-prefix load coalescing with its disabled control."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

from coldburst_verify import live_stats, verify_coldburst
from restore_common import ANCESTOR, write_json

HERE = Path(__file__).resolve().parent


def check_admission(row: dict, enabled: bool) -> None:
    """Reject undrained state and duplicate GPU ancestors in the enabled arm."""
    stats = row["singleflight"]
    if stats["inflight"] or stats["joined"]:
        raise ValueError("singleflight request state remains")
    live = live_stats(row["live_prefix"])
    if enabled and (
        live["resolved_external_load_requests"] != 1
        or live["max_live_ready_ancestor_blocks"] != ANCESTOR // 16
        or live["resolved_local_hit_requests"] != row["fanout"] - 1
    ):
        raise ValueError("coalescing did not produce one shared ancestor")


def verify(root: Path) -> dict:
    """Require numerical integrity, one ready ancestor, and exercised joins."""
    results = {}
    for gate in ("off", "on"):
        arm = root / gate
        plan = json.loads((arm / "plan.json").read_text())
        if plan["singleflight_control"] != gate:
            raise ValueError("missing explicit singleflight control")
        summary = verify_coldburst(arm)
        joins = 0
        for mode in ("staged", "direct"):
            for arrival in ("burst", "stagger"):
                rows = json.loads((arm / f"{mode}-{arrival}/consumer.json").read_text())
                for row in rows:
                    check_admission(row, gate == "on")
                joins += rows[-1]["singleflight"]["join_total"]
        if gate == "on" and not joins:
            raise ValueError("enabled arm never exercised a singleflight follower")
        results[gate] = summary | {"joined_requests_including_warmups": joins}
    off = json.loads((root / "off/provenance.json").read_text())
    on = json.loads((root / "on/provenance.json").read_text())
    for key in ("sources", "model_files", "harness", "native", "packages", "python"):
        if off[key] != on[key]:
            raise ValueError(f"control arms differ in {key}")
    return {
        "arms": results,
        "claim": "GPU ancestor admission only; native backend reads remain separate",
    }


def main() -> None:
    """Run both source-identical arms on the same GPU."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--model", type=Path)
    parser.add_argument("--vllm-source", type=Path)
    parser.add_argument("--lmcache-source", type=Path)
    parser.add_argument(
        "--gpu-provider", choices=("native", "vulkan_rm"), default="vulkan_rm"
    )
    parser.add_argument("--serving-samples", type=int, default=20)
    args = parser.parse_args()
    if not args.verify:
        if not all((args.model, args.vllm_source, args.lmcache_source)):
            parser.error("model and both component sources are required")
        args.output.mkdir()
        for gate in ("off", "on"):
            subprocess.run(
                [
                    sys.executable,
                    str(HERE / "run.py"),
                    "--output",
                    str(args.output / gate),
                    "--model",
                    str(args.model),
                    "--vllm-source",
                    str(args.vllm_source),
                    "--lmcache-source",
                    str(args.lmcache_source),
                    "--gpu-provider",
                    args.gpu_provider,
                    "--section",
                    "coldburst",
                    "--singleflight-control",
                    gate,
                    "--rounds",
                    "1",
                    "--adapter-samples",
                    "1",
                    "--serving-samples",
                    str(args.serving_samples),
                ],
                check=True,
            )
    write_json(args.output / "singleflight-verified.json", verify(args.output))


if __name__ == "__main__":
    main()
