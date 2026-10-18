# SPDX-License-Identifier: MIT
"""Check cascade engagement separately from output equivalence and timing."""

import argparse
import json
import math
from pathlib import Path
import statistics

HERE = Path(__file__).resolve().parent
MODES = ("disabled", "enabled", "forced")


def verify(root):
    profile = json.loads((HERE / "profile.json").read_text())
    workload = profile["workload"]
    cells = {name: json.loads((root / (name + ".json")).read_text()) for name in MODES}
    reference = cells["disabled"]
    summary = {}
    for name, cell in cells.items():
        if cell["mode"] != name or cell["workload"] != workload:
            raise ValueError("wrong mode or workload")
        if cell["runner"] != "vllm.v1.worker.gpu_model_runner.GPUModelRunner":
            raise ValueError("the cascade-capable V1 runner was not selected")
        if cell["backends"] != ["FLASH_ATTN"]:
            raise ValueError("FlashAttention was not selected")
        if cell["source_commit"] != profile["vllm_commit"]:
            raise ValueError("wrong vLLM source")
        if cell["prompts_sha256"] != reference["prompts_sha256"]:
            raise ValueError("different prompt inputs")
        if (
            not cell["native_sha256"]
            or cell["native_sha256"] != reference["native_sha256"]
        ):
            raise ValueError("missing or different native binaries")
        if cell["runtime"] != reference["runtime"]:
            raise ValueError("different runtime inputs")
        samples = cell["generation_wall_ms"]
        if len(samples) != workload["repeats"] or any(
            not math.isfinite(x) or x <= 0 for x in samples
        ):
            raise ValueError("missing or invalid generation timing")
        if len(cell["outputs"]) != workload["repeats"]:
            raise ValueError("missing generated outputs")
        for outputs in cell["outputs"]:
            if len(outputs) != workload["fanout"] or any(
                len(row) != workload["decode"] for row in outputs
            ):
                raise ValueError("incomplete generated continuations")
        if len(cell["cached_tokens"]) != workload["repeats"] or any(
            len(row) != workload["fanout"]
            or any(value < workload["ancestor"] for value in row)
            for row in cell["cached_tokens"]
        ):
            raise ValueError("timed family did not reuse its ancestor")
        stats = cell["engagement"]
        if name == "disabled" and stats["kernel_calls"]:
            raise ValueError("cascade ran in the disabled control")
        if name != "disabled" and stats["runner_calls"] == 0:
            raise ValueError("no cascade-capable runner observations")
        if name == "forced" and stats["kernel_calls"] == 0:
            raise ValueError("forced cascade did not reach its kernel")
        comparable = name == "disabled" or stats["kernel_calls"] > 0
        changed = sum(
            row != baseline
            for outputs, baseline_outputs in zip(cell["outputs"], reference["outputs"])
            for row, baseline in zip(outputs, baseline_outputs)
        )
        summary[name] = {
            "engagement": stats,
            "generation_wall_ms": samples,
            "median_generation_wall_ms": statistics.median(samples),
            "changed_continuations_vs_disabled": changed,
            "speed_comparison_has_engagement": comparable,
        }
    return {
        "workload": workload,
        "cells": summary,
        "limits": [
            "Warm prefix replay wall time includes cached-prefill scheduling and decode.",
            "Engagement is checked in a separate untimed generation, not each timed step.",
            "One process per mode and three repetitions do not establish a stable speedup.",
            "Floating-point reassociation can change greedy continuations; quality is untested.",
            "Cascade reads GPU-resident prefixes; no LMCache or storage transfer is measured.",
        ],
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    print(json.dumps(verify(parser.parse_args().output), indent=2))
