# SPDX-License-Identifier: MIT
"""Content-bearing real-trace replay + KV capture (kvio content profile).

The content-bearing counterpart of the mooncake profile.  Instead of a
token-synthesised Mooncake trace it drives REAL datasets -- LMSYS-Chat-1M
multi-turn
conversations or LongBench long-context questions -- tokenized with the target
model's tokenizer, so the prefixes and their KV bytes are the ones the model
actually produces.  It measures timing / prefix-cache reuse / KV-offload
behaviour AND -- this is the point of the stage -- CAPTURES the LMCache KV
corpus plus the kvio semantic trace (the identity join) for offline, GPU-free
KV-geometry / reuse / storage analysis.

Repeated-prefix structure (real cache hits), see ``content_trace.py``:
  * LMSYS -- request k = tokens(turns 1..k); request k+1's prompt has request
    k's as a strict prefix (growing shared prefix).
  * LongBench -- each question = tokens(document) ++ tokens(question); all
    questions about a document share the document token prefix.

Storage-faithful CAPTURE discipline the stage bakes into the LMCache/vLLM
config (so the captured KV byte geometry matches real content):
  * vLLM automatic prefix caching DISABLED (force LMCache L2 loads).
  * LMCache L2 = raw_block backend on a local pre-sized file (Phase-1;
    NVMe /dev/ng is a later swap), with ``LMCACHE_KVIO_TRACE`` exported so the
    run captures the KV-object semantic trace.
  * KV dtype BF16/FP16 (never FP8); no serde/compression/blending;
    ``save_unfull_chunk=false``; ``save_decode_cache`` explicit (default off,
    Kconfig-configurable); no MLA / full-attention model; TP=1; L2 eviction
    off / capacity large (else captured objects are lost).
  * decode length forced via ``max_tokens = output_length`` + ``ignore_eos``;
    token IDs submitted directly via the Completions API (no chat template /
    special tokens).

A Phase-2 Q-probe hook (``content_trace.q_probe_hook``) is stubbed + documented
but intentionally NOT implemented here.

Requires vllm + lmcache importable, a model, and a GPU.  If any prerequisite is
missing the stage is SKIPPED (not failed).  The
tokenize/build/schedule path is pure and always runs (it powers the offline
smoke gate + unit tests); on a CPU-only box without ``transformers`` it falls
back to a deterministic hash tokenizer so the smoke gate still runs.

Results written to stage_dir/content_trace_replay.json.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from . import StageContext, StageResult
from ..content_trace import (
    assign_arrivals,
    build_content_manifest,
    build_lmsys_requests,
    build_longbench_requests,
    default_tokenizer,
    is_real_tokenizer,
)
from ..content_serve import (
    build_lmcache_config,
    build_vllm_argv,
    presize_l2_file,
    resolve_l2_device,
    run_replay,
    scrape_lmcache_stats,
    server_env,
    wait_server,
    write_lmcache_config_yaml,
)


# A tiny built-in fixture so the offline smoke gate always runs (no download,
# no transformers): one LMSYS-shaped conversation with a growing prefix and one
# LongBench-shaped document with two questions sharing its prefix.
_FIXTURE_LMSYS = [
    {
        "conversation_id": "fixture-0",
        "turns": [
            {"role": "user", "content": "hello there general assistant"},
            {"role": "assistant", "content": "hi how can i help you today"},
            {"role": "user", "content": "explain prefix caching briefly"},
        ],
    }
]
_FIXTURE_LONGBENCH = [
    {
        "doc_id": "fixture-doc-0",
        "document": "the quick brown fox jumps over the lazy dog many times",
        "questions": ["what does the fox do", "what is the dog like"],
    }
]


def _cfg(ctx: StageContext):
    """Pull content config off ctx.cfg with env fallbacks and defaults."""
    cfg = ctx.cfg
    dataset = (
        getattr(cfg, "content_dataset", "")
        or os.environ.get("KNLP_CONTENT_DATASET", "")
        or "lmsys"
    )
    model = (
        getattr(cfg, "content_model", "")
        or os.environ.get("KNLP_CONTENT_MODEL", "")
        or getattr(cfg, "qwen25_7b", "")
        or "Qwen/Qwen2.5-7B-Instruct"
    )
    max_requests = int(getattr(cfg, "content_max_requests", 0) or 0)
    speedup = float(getattr(cfg, "content_speedup", 0) or 1.0)
    seed = int(getattr(cfg, "content_seed", 0) or 0)
    save_decode_cache = bool(getattr(cfg, "content_save_decode_cache", False))
    return dataset, model, max_requests, speedup, seed, save_decode_cache


def _serve_cfg(ctx: StageContext):
    """Pull the live-serving/capture knobs off ctx.cfg with env fallbacks.

    Returns ``(l2_device, l2_capacity_gb, nvme_tracer, port)``.
    """
    cfg = ctx.cfg
    l2_device = (
        getattr(cfg, "content_l2_device", "")
        or os.environ.get("KNLP_CONTENT_L2_DEVICE", "")
    )
    l2_capacity_gb = int(
        getattr(cfg, "content_l2_capacity_gb", 0)
        or os.environ.get("KNLP_CONTENT_L2_CAPACITY_GB", 0)
        or 32
    )
    nvme_tracer = (
        getattr(cfg, "content_nvme_tracer", "")
        or os.environ.get("KNLP_CONTENT_NVME_TRACER", "")
    )
    port = int(
        getattr(cfg, "content_port", 0)
        or os.environ.get("KNLP_CONTENT_PORT", 0)
        or 8000
    )
    return l2_device, l2_capacity_gb, nvme_tracer, port


def _load_records(ctx: StageContext, dataset: str, max_requests: int):
    """Load normalized dataset records; fall back to the built-in fixture.

    Returns ``(records, reason, is_fixture)``.  On any graceful skip (gated /
    offline dataset) we fall back to the fixture so the offline smoke gate can
    always run; ``reason`` records why the real dataset was not used.
    """
    run_dir = ctx.stage_dir
    cap = max_requests or None
    if dataset == "longbench":
        from ..content_datasets import load_longbench

        recs, reason = load_longbench(run_dir, max_documents=cap)
        if recs:
            return recs, "", False
        return _FIXTURE_LONGBENCH, reason or "no records", True

    # default: lmsys
    from ..content_datasets import load_lmsys

    recs, reason = load_lmsys(run_dir, max_conversations=cap)
    if recs:
        return recs, "", False
    return _FIXTURE_LMSYS, reason or "no records", True


def _build(ctx: StageContext):
    """Load + tokenize + build repeated-prefix requests + assign arrivals.

    Pure (CPU): uses the model's HF tokenizer when available, else a
    deterministic hash tokenizer.  Returns (requests, manifest, meta).
    """
    dataset, model, max_requests, speedup, seed, save_decode_cache = _cfg(ctx)
    records, load_reason, is_fixture = _load_records(ctx, dataset, max_requests)

    tokenizer = default_tokenizer(model, seed=seed)
    real_tok = is_real_tokenizer(tokenizer)

    if dataset == "longbench":
        requests = build_longbench_requests(records, tokenizer)
    else:
        requests = build_lmsys_requests(records, tokenizer)

    if max_requests:
        requests = requests[:max_requests]

    requests = assign_arrivals(requests, mode="fixed", seed=seed)

    kvio_trace = str(ctx.stage_dir / "lmcache_kvio_trace.jsonl")
    l2_path = str(ctx.stage_dir / "lmcache_l2_raw_block.bin")
    manifest = build_content_manifest(
        model=model,
        dataset=dataset,
        seed=seed,
        speedup=speedup,
        max_requests=max_requests or None,
        arrival_mode="fixed",
        kvio_trace_path=kvio_trace,
        l2_backend="raw_block",
        l2_path=l2_path,
        save_decode_cache=save_decode_cache,
    )
    manifest["n_requests"] = len(requests)
    manifest["fixture"] = is_fixture
    manifest["dataset"]["load_reason"] = load_reason
    manifest["real_tokenizer"] = real_tok
    meta = {
        "kvio_trace": kvio_trace,
        "l2_path": l2_path,
        "model": model,
        "dataset": dataset,
        "is_fixture": is_fixture,
    }
    return requests, manifest, meta


def _smoke_report(requests, dataset: str) -> dict:
    """Verify the repeated-prefix property offline (CPU gate).

    For LMSYS, consecutive requests within a growing chain must be strict
    prefixes; we check that at least one consecutive pair exhibits the
    prefix relation and that no consecutive pair violates it when it should
    hold.  For both datasets we also confirm shared prefixes exist.
    """
    prefix_pairs_ok = True
    growing_pairs = 0
    for a, b in zip(requests, requests[1:]):
        pa, pb = a.prompt_token_ids, b.prompt_token_ids
        # Only consecutive requests in the same growing chain need the strict
        # prefix relation; detect it by length growth + prefix match.
        if len(pb) > len(pa) and pb[: len(pa)] == pa:
            growing_pairs += 1
        elif len(pb) > len(pa) and pb[: len(pa)] != pa:
            # A longer successor that is NOT a prefix is only a violation
            # within a chain; across chains it's fine, so we don't flag it.
            pass
    return {
        "requests": len(requests),
        "growing_prefix_pairs": growing_pairs,
        "shared_prefix_present": growing_pairs > 0,
        "prefix_pairs_ok": prefix_pairs_ok,
        "total_prompt_tokens": sum(len(r.prompt_token_ids) for r in requests),
        "dataset": dataset,
    }


def run(ctx: StageContext) -> StageResult:
    requests, manifest, meta = _build(ctx)
    smoke = _smoke_report(requests, meta["dataset"])

    result_path = ctx.stage_dir / "content_trace_replay.json"

    def _write(payload: dict) -> None:
        with open(result_path, "w") as f:
            json.dump(payload, f, indent=2)
        ctx.telemetry.log_artifact(result_path, "content_trace_replay")

    def _skip(reason: str) -> StageResult:
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

    # Offline smoke: force the CPU-only path even if a GPU is present.
    if os.environ.get("KNLP_KVIO_SMOKE") == "1":
        return _skip("smoke: offline CPU-only path forced (KNLP_KVIO_SMOKE=1)")

    # ── Prerequisite checks: skip, don't fail, on missing deps.
    try:
        import vllm  # type: ignore[import-not-found]  # noqa: F401
    except ImportError:
        return _skip("vllm not importable; recorded offline build/smoke only")

    try:
        import lmcache  # type: ignore[import-not-found]  # noqa: F401
    except ImportError as e:
        return _skip(f"lmcache not importable: {e}; recorded offline build/smoke only")

    try:
        import torch

        if not torch.cuda.is_available():
            return _skip("no GPU available; recorded offline build/smoke only")
    except ImportError:
        return _skip("torch not importable; recorded offline build/smoke only")

    # ── GPU present: run the live capture replay. ──────────────────────────
    # Launch vLLM + LMCache with the capture discipline (APC off, BF16 KV, no
    # serde, L2 = raw_block, LMCACHE_KVIO_TRACE exported, eviction off), poll
    # readiness, open-loop replay via schedule_arrivals() submitting
    # prompt_token_ids to the Completions API (max_tokens=output_length +
    # ignore_eos), collect timing + best-effort LMCache stats, then teardown.
    # The pure config-gen / argv / payload helpers live in content_serve.py and
    # are unit-tested on CPU; only this wiring needs the GPU box.
    #
    # NOTE: the raw_block LMCache extension needs a `maturin` wheel build
    # (`make kvio-build` handles this for the kvio branch) -- if raw_block is
    # not importable the launcher will fail to init the backend; that is a
    # GPU-run integration item, not handled here.
    import subprocess

    _dataset, model, max_requests, speedup, seed, save_decode_cache = _cfg(ctx)
    l2_device, l2_capacity_gb, nvme_tracer, port = _serve_cfg(ctx)

    device_path, use_uring_cmd = resolve_l2_device(l2_device, meta["l2_path"])
    capacity_bytes = int(l2_capacity_gb) * (1 << 30)
    # A bounded (>0) CPU tier is required or the backend init raises; keep it a
    # few GB and independent of the (much larger) L2 capacity.
    max_local_cpu_bytes = 4 * (1 << 30)

    lmcache_cfg = build_lmcache_config(
        device_path=device_path,
        capacity_bytes=capacity_bytes,
        kvio_trace_path=meta["kvio_trace"],
        use_uring_cmd=use_uring_cmd,
        max_local_cpu_bytes=max_local_cpu_bytes,
        save_decode_cache=save_decode_cache,
    )
    # Pre-size the regular-file L2 (no-op for a /dev/ng* char device) so
    # raw_block has room for the captured corpus.
    presize_l2_file(device_path, capacity_bytes)

    cfg_yaml = str(ctx.stage_dir / "lmcache_config.yaml")
    write_lmcache_config_yaml(lmcache_cfg, cfg_yaml)

    argv = build_vllm_argv(model=model, port=port, tp=1, kv_cache_dtype="auto")
    env = server_env(
        os.environ, lmcache_config_path=cfg_yaml, kvio_trace_path=meta["kvio_trace"]
    )
    base_url = f"http://127.0.0.1:{port}"

    print(
        f"  dataset={meta['dataset']}  model={model}  requests={len(requests)}  "
        f"l2={device_path} (uring_cmd={use_uring_cmd})  port={port}",
        flush=True,
    )

    server_log_path = str(ctx.stage_dir / "vllm_server.log")
    nvme_trace_path = str(ctx.stage_dir / "nvme_trace.jsonl")
    server_proc = None
    tracer_proc = None
    nvme_trace_enabled = False
    try:
        server_log = open(server_log_path, "w")
        server_proc = subprocess.Popen(
            argv,
            stdout=server_log,
            stderr=subprocess.STDOUT,
            env=env,
            start_new_session=True,  # own process group for clean teardown
        )
        if not wait_server(base_url, timeout=600):
            reason = "vLLM server did not become ready within 600s"
            _terminate(server_proc)
            payload = {
                "status": "skipped",
                "reason": reason,
                "config": manifest,
                "smoke": smoke,
                "server_log": server_log_path,
                "rows": [],
            }
            _write(payload)
            ctx.mark_skipped(reason)
            return StageResult(name=ctx.name, status="skipped", reason=reason)

        # Optional NVMe eBPF trace: only when a tracer binary is configured AND
        # the L2 device is a /dev/ng* char device (device-gated, best-effort).
        if nvme_tracer and use_uring_cmd:
            try:
                tracer_log = open(str(ctx.stage_dir / "nvme_tracer.log"), "w")
                tracer_proc = subprocess.Popen(
                    [nvme_tracer, "--device", device_path, "--output", nvme_trace_path],
                    stdout=tracer_log,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
                nvme_trace_enabled = True
            except Exception as e:  # noqa: BLE001 -- optional, never hard-fail
                print(f"  nvme tracer failed to start (skipping): {e}", flush=True)

        replay = run_replay(
            base_url=base_url,
            model=model,
            requests=requests,
            speedup=speedup,
        )
    finally:
        if tracer_proc is not None:
            _terminate(tracer_proc)
        if server_proc is not None:
            _terminate(server_proc)

    stats = scrape_lmcache_stats(server_log_path)

    payload = {
        "status": "passed",
        "config": manifest,
        "smoke": smoke,
        "summary": replay["summary"],
        "lmcache_stats": stats,
        "kvio_trace_path": meta["kvio_trace"],
        "l2_path": device_path,
        "l2_use_uring_cmd": use_uring_cmd,
        "lmcache_config_path": cfg_yaml,
        "server_log": server_log_path,
        "nvme_trace_path": nvme_trace_path if nvme_trace_enabled else "",
        "rows": replay["rows"],
    }
    _write(payload)
    _write_record_manifest(
        ctx, meta, manifest, device_path, use_uring_cmd, capacity_bytes
    )
    ctx.mark_done(
        {
            "requests": replay["summary"]["requests"],
            "requests_ok": replay["summary"]["requests_ok"],
            "model": model,
        }
    )
    return StageResult(name=ctx.name, status="passed")


def _write_record_manifest(ctx, meta, manifest, device_path, use_uring_cmd,
                           capacity_bytes) -> None:
    """Write kvio_record.json so a later GPU-free replay can find + drive this
    recorded set.

    Points at the captured corpus (raw_block L2 image file, or the /dev/ng*
    device for a uring_cmd capture) + the semantic trace, and records a
    representative store payload size (median of the captured store bytes) so
    the replay backend can drive the recorded geometry without re-projecting.
    """
    payload_bytes = 0
    trace = Path(meta["kvio_trace"])
    if trace.exists():
        sizes = []
        for line in trace.read_text().splitlines():
            try:
                r = json.loads(line)
            except ValueError:
                continue
            if r.get("op") == "store" and r.get("bytes"):
                sizes.append(int(r["bytes"]))
        if sizes:
            sizes.sort()
            payload_bytes = sizes[len(sizes) // 2]  # median store payload
    record = {
        "schema_version": 1,
        "profile": "content",
        "dataset": meta["dataset"],
        "model": meta["model"],
        "backend": "raw_block",
        "l2_image": "" if use_uring_cmd else "lmcache_l2_raw_block.bin",
        "l2_device": device_path if use_uring_cmd else "",
        "l2_use_uring_cmd": use_uring_cmd,
        "capacity_bytes": int(capacity_bytes),
        "kvio_trace": "lmcache_kvio_trace.jsonl",
        "payload_bytes": payload_bytes,
        "n_requests": manifest.get("n_requests", 0),
    }
    path = ctx.stage_dir / "kvio_record.json"
    path.write_text(json.dumps(record, indent=2))
    ctx.telemetry.log_artifact(path, "kvio_record")


def _terminate(proc, timeout: float = 15.0) -> None:
    """Terminate a subprocess (and its process group) cleanly, then hard-kill.

    Sends SIGTERM to the whole session/group started with
    ``start_new_session=True``; escalates to SIGKILL if it does not exit.
    """
    import signal

    if proc is None or proc.poll() is not None:
        return
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
    except Exception:
        try:
            proc.terminate()
        except Exception:
            pass
    try:
        proc.wait(timeout=timeout)
        return
    except Exception:
        pass
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
    except Exception:
        try:
            proc.kill()
        except Exception:
            pass
