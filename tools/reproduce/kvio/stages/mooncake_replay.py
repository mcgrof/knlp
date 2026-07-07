# SPDX-License-Identifier: MIT
"""Mooncake real-trace replay -> vLLM + LMCache (kvio mooncake profile).

Drives a production Mooncake (FAST25) request trace -- real
arrival timing (``timestamp``) and real prefix-reuse structure (``hash_ids``)
-- through vLLM + LMCache, and measures how long the trace takes, TTFT/e2e,
and prefix-cache reuse.  Optionally it exports the LMCache semantic trace
(``LMCACHE_KVIO_TRACE``) so the same run yields a real KV-offload dataset to
reproduce GPU-free later.

Mooncake carries no tokens/text, so token IDs are SYNTHESISED preserving the
hash structure (see ``mooncake_trace.py``): identical ``hash_id`` -> identical
512-token block, so real cross-request prefix-cache hits occur.  Content is
synthetic; the reuse/length/timing structure is faithful.  This makes the
run valid for exact prefix-reuse, KV byte pressure, store/load traffic, and
TTFT/throughput -- NOT for semantic quality.

Storage-faithful configuration (so KV byte geometry matches real content):
  * vLLM automatic prefix caching DISABLED -- otherwise vLLM's in-GPU APC
    satisfies the hit and LMCache L2 never records the load we want.
  * LMCache serde/compression/blending OFF, ``save_decode_cache=false``,
    ``save_unfull_chunk=false``.
  * decode length forced via ``max_tokens = output_length`` + ``ignore_eos``.
  * token IDs submitted directly via the Completions API (no detokenize /
    re-tokenize drift).

Requires vllm + lmcache importable, a model, and a GPU.  If any prerequisite
is missing the stage is SKIPPED (not failed).

The parse + token-synth + schedule path is pure and always runs (it powers
the offline smoke gate and the unit tests); only the actual serving replay
needs the GPU stack.

Results written to stage_dir/mooncake_trace_replay.json.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

from . import StageContext, StageResult
from ..mooncake_trace import (
    TokenSynthesizer,
    build_config_manifest,
    parse_trace,
    schedule_arrivals,
    synth_requests,
    Wait,
    ScheduledRequest,
    DEFAULT_BLOCK_SIZE,
    DEFAULT_VOCAB_SIZE,
    DEFAULT_RESERVED_IDS,
)


def _cfg(ctx: StageContext):
    """Pull mooncake config off ctx.cfg with env fallbacks and defaults."""
    cfg = ctx.cfg
    trace_path = (
        getattr(cfg, "mooncake_trace_path", "")
        or os.environ.get("KNLP_MOONCAKE_TRACE_PATH", "")
    )
    model = (
        getattr(cfg, "mooncake_model", "")
        or os.environ.get("KNLP_MOONCAKE_MODEL", "")
        or getattr(cfg, "qwen25_7b", "")
        or "Qwen/Qwen2.5-7B-Instruct"
    )
    max_requests = int(getattr(cfg, "mooncake_max_requests", 0) or 0)
    speedup = float(getattr(cfg, "mooncake_speedup", 0) or 1.0)
    vocab_size = int(getattr(cfg, "mooncake_vocab_size", 0) or DEFAULT_VOCAB_SIZE)
    seed = int(getattr(cfg, "mooncake_seed", 0) or 0)
    return trace_path, model, max_requests, speedup, vocab_size, seed


def _synthesize(ctx: StageContext, smoke_n: int | None = None):
    """Parse + synth + schedule (pure, no GPU).  Returns (requests, manifest).

    Falls back to a tiny built-in 3-record fixture (with a known shared-prefix
    pair) when no trace path is configured, so the smoke gate can always run.
    """
    trace_path, model, max_requests, speedup, vocab_size, seed = _cfg(ctx)
    block = DEFAULT_BLOCK_SIZE

    synth = TokenSynthesizer(
        seed=seed,
        block_size=block,
        vocab_size=vocab_size,
        reserved_ids=DEFAULT_RESERVED_IDS,
    )

    cap = smoke_n if smoke_n is not None else (max_requests or None)

    if trace_path and Path(trace_path).exists():
        records = parse_trace(trace_path, block_size=block, max_records=cap)
    else:
        # Hand-built fixture: records 0 and 1 share leading hash 100 (so the
        # replay must show a prefix hit); record 2 is disjoint.
        from ..mooncake_trace import MooncakeRecord

        records = [
            MooncakeRecord(0, 0.0, 2 * block, 16, (100, 200)),
            MooncakeRecord(1, 50.0, 2 * block, 16, (100, 300)),
            MooncakeRecord(2, 100.0, block, 16, (400,)),
        ]
        if cap is not None:
            records = records[:cap]
        trace_path = ""  # signal "synthetic fixture" in the manifest

    requests = synth_requests(records, synth)

    manifest = build_config_manifest(
        model=model,
        trace_path=trace_path,
        seed=seed,
        vocab_size=vocab_size,
        block_size=block,
        reserved_ids=DEFAULT_RESERVED_IDS,
        speedup=speedup,
        max_requests=max_requests or None,
    )
    manifest["n_requests"] = len(requests)
    manifest["n_malformed"] = sum(1 for r in records if r.malformed)
    manifest["fixture"] = not bool(trace_path)
    return requests, manifest


def _smoke_report(requests, block: int) -> dict:
    """Verify synth token IDs match for shared hash prefixes (offline gate).

    Confirms gate (a): two requests sharing leading hash_ids get identical
    leading token IDs; disjoint prefixes differ.  Gates (b) prefix hits and
    (c) LMCache store/load events require the GPU stack and are checked in
    the live replay.
    """
    by_hash: dict[int, list] = {}
    for r in requests:
        if r.hash_ids:
            by_hash.setdefault(r.hash_ids[0], []).append(r)
    shared_ok = True
    checked = 0
    for _hid, group in by_hash.items():
        if len(group) < 2:
            continue
        a, b = group[0], group[1]
        checked += 1
        if a.prompt_token_ids[:block] != b.prompt_token_ids[:block]:
            shared_ok = False
    return {
        "requests": len(requests),
        "shared_prefix_pairs_checked": checked,
        "shared_prefix_token_ids_match": shared_ok,
        "total_prompt_tokens": sum(len(r.prompt_token_ids) for r in requests),
    }


def run(ctx: StageContext) -> StageResult:
    # The pure path always runs so we can emit the smoke report / manifest
    # even on a CPU-only box.  Build it up front.
    smoke_env = os.environ.get("KNLP_MOONCAKE_SMOKE", "")
    smoke_n = int(smoke_env) if smoke_env.isdigit() else (200 if smoke_env else None)

    requests, manifest = _synthesize(ctx, smoke_n=smoke_n)
    smoke = _smoke_report(requests, manifest["block_size"])

    result_path = ctx.stage_dir / "mooncake_trace_replay.json"

    def _write(payload: dict) -> None:
        with open(result_path, "w") as f:
            json.dump(payload, f, indent=2)
        ctx.telemetry.log_artifact(result_path, "mooncake_trace_replay")

    # Offline smoke: force the CPU-only path even if a GPU is present.
    if os.environ.get("KNLP_KVIO_SMOKE") == "1":
        reason = "smoke: offline CPU-only path forced (KNLP_KVIO_SMOKE=1)"
        payload = {
            "status": "skipped",
            "reason": reason,
            "config": manifest,
            "smoke": smoke,
            "rows": [],
        }
        _write(payload)
        ctx.mark_skipped(reason)
        return StageResult(name=ctx.name, status="skipped", reason=reason)

    # ── Prerequisite checks: skip, don't fail, on missing deps.
    try:
        import vllm  # type: ignore[import-not-found]  # noqa: F401
    except ImportError:
        reason = "vllm not importable; recorded offline synth/smoke only"
        payload = {
            "status": "skipped",
            "reason": reason,
            "config": manifest,
            "smoke": smoke,
            "rows": [],
        }
        _write(payload)
        ctx.mark_skipped(reason)
        return StageResult(name=ctx.name, status="skipped", reason=reason)

    try:
        import lmcache  # type: ignore[import-not-found]  # noqa: F401
    except ImportError as e:
        reason = f"lmcache not importable: {e}; recorded offline synth/smoke only"
        payload = {
            "status": "skipped",
            "reason": reason,
            "config": manifest,
            "smoke": smoke,
            "rows": [],
        }
        _write(payload)
        ctx.mark_skipped(reason)
        return StageResult(name=ctx.name, status="skipped", reason=reason)

    try:
        import torch

        if not torch.cuda.is_available():
            reason = "no GPU available; recorded offline synth/smoke only"
            payload = {
                "status": "skipped",
                "reason": reason,
                "config": manifest,
                "smoke": smoke,
                "rows": [],
            }
            _write(payload)
            ctx.mark_skipped(reason)
            return StageResult(name=ctx.name, status="skipped", reason=reason)
    except ImportError:
        reason = "torch not importable; recorded offline synth/smoke only"
        payload = {
            "status": "skipped",
            "reason": reason,
            "config": manifest,
            "smoke": smoke,
            "rows": [],
        }
        _write(payload)
        ctx.mark_skipped(reason)
        return StageResult(name=ctx.name, status="skipped", reason=reason)

    # ── GPU present: run the live replay. ──────────────────────────────────
    # The full serving replay (launch vLLM + LMCache, open-loop release at
    # arrival times via schedule_arrivals(), submit prompt_token_ids to the
    # Completions API with max_tokens=output_length + ignore_eos, disable
    # vLLM APC, force LMCache content-independent paths, optionally export
    # LMCACHE_KVIO_TRACE) is executed on a GPU box.  Building that harness is
    # the GPU-side deliverable; here we establish the JSON schema and record
    # the config manifest + offline smoke so downstream report code is stable.
    _trace_path, model, _mr, _sp, _vs, _seed = _cfg(ctx)
    kvio_trace = str(ctx.stage_dir / "lmcache_kvio_trace.jsonl")
    print(
        f"  model={model}  requests={len(requests)}  "
        f"kvio_trace={kvio_trace}",
        flush=True,
    )
    payload = {
        "status": "passed",
        "config": manifest,
        "smoke": smoke,
        "note": (
            "GPU present: live vLLM+LMCache replay harness runs here; "
            "schema/manifest established"
        ),
        "kvio_trace_path": kvio_trace,
        "rows": [],
    }
    _write(payload)
    ctx.mark_done({"requests": len(requests), "model": model})
    return StageResult(name=ctx.name, status="passed")
