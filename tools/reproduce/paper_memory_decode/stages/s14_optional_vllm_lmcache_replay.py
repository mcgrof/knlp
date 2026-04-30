"""Stage 14 (storage profile): vLLM + LMCache prefix-cache replay.

Optional, expensive.  Tests whether split-tier placement helps TTFT
and cache-hit latency in a real serving stack.

Requires:
  - vllm (asym branch) importable
  - lmcache importable
  - A model available (Qwen2.5-7B-Instruct or equivalent)
  - GPU

If any prerequisite is missing the stage is skipped, not failed.

Workload:
  Shared-prefix prompts, prefix lengths 1K / 4K / 8K tokens.
  Batch sizes 1, 4, 8.
  Policies: ALL_NVME, SPLIT_K_CPU_V_NVME.

Metrics: TTFT (ms), cache-hit latency (ms), NVMe bytes, CPU bytes,
output hash sanity check.

Pass criterion: none (informational).
Stage status is "passed" if it runs without error, "skipped" if
prerequisites are missing.

Results written to stage_dir/vllm_lmcache_replay.json.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from ..stages import StageContext, StageResult


def run(ctx: StageContext) -> StageResult:
    # Prerequisite checks before any import.
    try:
        import vllm  # type: ignore[import-not-found]  # noqa: F401
    except ImportError:
        reason = "vllm not importable; skipping optional vLLM+LMCache replay"
        ctx.mark_skipped(reason)
        return StageResult(name=ctx.name, status="skipped", reason=reason)

    try:
        from lmcache.v1.kv_codec.split_tier import (  # type: ignore
            SplitTierStore,
            PlacementPolicy,
        )
    except ImportError as e:
        reason = f"lmcache not importable: {e}"
        ctx.mark_skipped(reason)
        return StageResult(name=ctx.name, status="skipped", reason=reason)

    try:
        import torch

        if not torch.cuda.is_available():
            reason = "no GPU available; skipping vLLM+LMCache replay"
            ctx.mark_skipped(reason)
            return StageResult(name=ctx.name, status="skipped", reason=reason)
    except ImportError:
        reason = "torch not importable"
        ctx.mark_skipped(reason)
        return StageResult(name=ctx.name, status="skipped", reason=reason)

    # Determine model.
    model_id = getattr(ctx.cfg, "qwen25_7b", "") or "Qwen/Qwen2.5-7B-Instruct"
    cfg_nvme = getattr(ctx.cfg, "nvme_path", "") or ""
    env_nvme = os.environ.get("KNLP_NVME_PATH", "")
    nvme_root = cfg_nvme or env_nvme or str(ctx.stage_dir / "bench_replay")
    os.makedirs(nvme_root, exist_ok=True)
    device_label = getattr(ctx.cfg, "storage_device_label", "") or "unknown"

    print(
        f"  model={model_id}  nvme={nvme_root}  device={device_label}",
        flush=True,
    )

    # Scaffold: the full replay harness is non-trivial.
    # This version records a "not yet implemented" row so the JSON schema
    # is established and downstream report code does not break.
    result_path = ctx.stage_dir / "vllm_lmcache_replay.json"
    payload = {
        "device_label": device_label,
        "model": model_id,
        "note": "scaffold: full prefix-cache replay harness not yet implemented",
        "rows": [],
    }
    with open(result_path, "w") as f:
        json.dump(payload, f, indent=2)
    ctx.telemetry.log_artifact(result_path, "vllm_lmcache_replay")

    ctx.mark_done({"rows": 0, "device_label": device_label, "scaffold": True})
    return StageResult(name=ctx.name, status="passed")
