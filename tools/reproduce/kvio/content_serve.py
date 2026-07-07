# SPDX-License-Identifier: MIT
"""Serving + capture helpers for the content-trace replay stage.

This module holds the pieces that turn built requests into a live vLLM +
LMCache serving run that CAPTURES the KV corpus + the kvio semantic trace:

  * ``build_lmcache_config``  -- the validated LMCache raw_block plugin recipe
    (this is the bug-trap; unit-tested hard).
  * ``build_vllm_argv``       -- the vLLM launch argv (APC off, non-MP
    ``LMCacheConnectorV1``, TP=1, never fp8).
  * ``completion_payload``    -- the ``/v1/completions`` request body
    (token-ID prompt, ``max_tokens=output_length``, ``ignore_eos``).
  * ``resolve_l2_device``     -- the ``CONFIG_KNLP_KVIO_CONTENT_L2_DEVICE`` knob:
    empty -> local pre-sized file (``use_uring_cmd`` false); ``/dev/ng*`` ->
    raw NVMe char device (``use_uring_cmd`` true).
  * orchestration helpers (``write_lmcache_config_yaml``, ``server_env``,
    ``wait_server``, ``run_replay``, ``scrape_lmcache_stats``) that the stage
    wires together.

Design rule: this module is PURE at import time -- it imports only the Python
standard library (and, lazily, ``yaml`` inside the one function that needs it).
It imports NO vllm / lmcache / torch / GPU packages, so it imports and is
unit-testable on a CPU-only box.  The GPU gating lives in the stage.

The plugin recipe encoded by ``build_lmcache_config`` was confirmed by reading
the LMCache raw_block backend source
(``lmcache/v1/storage_backend/plugins/rust_raw_block_backend.py`` +
``lmcache/v1/storage_backend/__init__.py`` +
``lmcache/v1/storage_backend/raw_block/core.py``):

  * ``storage_plugins=["rust_raw_block"]`` selects the plugin; its class is
    resolved from ``extra_config["storage_plugin.rust_raw_block.module_path"]``
    + ``...class_name`` by the dynamic backend loader.
  * the backend reads ``extra_config["rust_raw_block.device_path" /
    ".io_engine" / ".use_uring_cmd" / ".capacity_bytes"]``.
  * ``use_uring_cmd`` requires an EXISTING NVMe generic namespace character
    device whose basename matches ``ng<ctrl>n<ns>`` (e.g. ``/dev/ng0n1``); a
    regular file must therefore use ``use_uring_cmd=false``.
  * a regular-file L2 must be pre-sized: the core clamps effective capacity to
    ``min(capacity_bytes, device_size)``, so a 0-byte file yields no slots.
"""

from __future__ import annotations

import json
import os
import re
import sys
import threading
import time
import urllib.request
from typing import Optional, Sequence

from .serving_replay import SynthRequest, schedule_arrivals

# LMCache dynamic-plugin coordinates for the raw_block backend (confirmed
# against the LMCache source path + class name).
RAW_BLOCK_MODULE_PATH = "lmcache.v1.storage_backend.plugins.rust_raw_block_backend"
RAW_BLOCK_CLASS_NAME = "RustRawBlockBackend"

# NVMe generic namespace character-device basename pattern (ng<ctrl>n<ns>),
# matching the validation in RawBlockCore; only such a path may use uring_cmd.
_NVME_CHAR_RE = re.compile(r"^ng\d+n\d+$")


# ── Device-knob resolution ─────────────────────────────────────────────────


def is_nvme_char_device(device_path: str) -> bool:
    """True when ``device_path`` is an NVMe generic namespace char device.

    Matches ``/dev/ng<ctrl>n<ns>`` (e.g. ``/dev/ng0n1``) -- the ONLY shape for
    which ``use_uring_cmd`` is valid (io_uring_cmd passthrough).  A regular
    file (the Phase-1 / offline L2) is not a char device and must not enable
    uring_cmd.
    """
    if not device_path:
        return False
    return bool(_NVME_CHAR_RE.match(os.path.basename(device_path.rstrip("/"))))


def resolve_l2_device(l2_device: str, l2_path: str) -> tuple[str, bool]:
    """Resolve ``(device_path, use_uring_cmd)`` for the raw_block L2 backend.

    ``CONFIG_KNLP_KVIO_CONTENT_L2_DEVICE`` semantics:
      * empty      -> use the local pre-sized file at ``l2_path`` (Phase-1 /
        offline), a regular file, so ``use_uring_cmd=False``.
      * ``/dev/ng*`` -> use it as the raw NVMe char device with
        ``use_uring_cmd=True`` (real io_uring_cmd capture).
      * any other set path -> use it as a regular-file device with
        ``use_uring_cmd=False`` (not a char device).
    """
    dev = (l2_device or "").strip()
    if dev:
        return dev, is_nvme_char_device(dev)
    return l2_path, False


# ── The validated LMCache config recipe (bug-trap; unit-tested hard) ────────


def build_lmcache_config(
    *,
    device_path: str,
    capacity_bytes: int,
    kvio_trace_path: str,
    use_uring_cmd: bool,
    max_local_cpu_bytes: int,
    save_decode_cache: bool = False,
    io_engine: str = "io_uring",
    module_path: str = RAW_BLOCK_MODULE_PATH,
    class_name: str = RAW_BLOCK_CLASS_NAME,
) -> dict:
    """Return the LMCache config (yaml-able dict) for storage-faithful capture.

    Encodes the confirmed recipe so a vLLM serving run routes KV to the
    raw_block L2 and fires the kvio trace:

      * ``local_cpu`` on with a bounded (> 0) CPU tier -- required, else the
        backend init raises / the launcher skips.
      * the ``rust_raw_block`` storage PLUGIN (non-MP) on ``device_path`` via
        io_uring; ``use_uring_cmd`` only when ``device_path`` is a ``/dev/ng*``
        char device.
      * NO serde / compression / blending keys, ``save_unfull_chunk=false`` --
        any of those would change the captured KV bytes so they no longer match
        real content.  ``save_decode_cache`` is explicit (default off).

    ``max_local_cpu_bytes`` is converted to LMCache's ``max_local_cpu_size``,
    which is expressed in GiB (float) in the LMCache schema; it is kept > 0.
    """
    if capacity_bytes <= 0:
        raise ValueError(f"capacity_bytes must be > 0, got {capacity_bytes}")
    if use_uring_cmd and not is_nvme_char_device(device_path):
        raise ValueError(
            "use_uring_cmd=True requires an NVMe char device path "
            f"(e.g. /dev/ng0n1), got {device_path!r}"
        )
    # LMCache's max_local_cpu_size is GB (float); keep strictly > 0.
    max_local_cpu_gb = max(int(max_local_cpu_bytes), 1) / 1e9

    cfg: dict = {
        "local_cpu": True,
        "max_local_cpu_size": max_local_cpu_gb,
        # Cache-shortcut / content-dependent paths OFF so the captured KV byte
        # geometry matches real content (see the module + stage docstrings).
        "save_decode_cache": bool(save_decode_cache),
        "save_unfull_chunk": False,
        "storage_plugins": ["rust_raw_block"],
        "extra_config": {
            "storage_plugin.rust_raw_block.module_path": module_path,
            "storage_plugin.rust_raw_block.class_name": class_name,
            "rust_raw_block.device_path": device_path,
            "rust_raw_block.io_engine": io_engine,
            "rust_raw_block.use_uring_cmd": bool(use_uring_cmd),
            "rust_raw_block.capacity_bytes": int(capacity_bytes),
            # kvio semantic-trace path mirror.  The CONFIRMED mechanism is the
            # LMCACHE_KVIO_TRACE env var (set by the stage on the server env);
            # this config mirror is best-effort and its exact key must be
            # confirmed against the kvio LMCache branch on the GPU box.
            # TODO(gpu-box): confirm the config key name (or drop this mirror
            # if the kvio branch only reads the env var).
            "rust_raw_block.kvio_trace_path": kvio_trace_path,
        },
    }
    return cfg


def write_lmcache_config_yaml(config: dict, path: str | os.PathLike) -> str:
    """Serialize an LMCache config dict to a YAML file; return the path.

    ``yaml`` is imported lazily so this module stays importable on a CPU box
    without PyYAML (only the live serving path calls this).
    """
    import yaml  # lazy: keeps the module import stdlib-only

    with open(path, "w") as f:
        yaml.safe_dump(config, f, sort_keys=False)
    return str(path)


def presize_l2_file(path: str | os.PathLike, capacity_bytes: int) -> None:
    """Create/pre-size a regular-file L2 to ``capacity_bytes`` (sparse).

    RawBlockCore clamps effective capacity to ``min(capacity_bytes,
    device_size)``, so a regular-file L2 must already be at least that large or
    it yields zero slots.  Char devices (``/dev/ng*``) are skipped -- their
    size comes from the device.
    """
    p = str(path)
    if is_nvme_char_device(p):
        return
    parent = os.path.dirname(p)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(p, "a+b") as f:
        f.truncate(int(capacity_bytes))


# ── vLLM launch argv ────────────────────────────────────────────────────────


def build_vllm_argv(
    *,
    model: str,
    port: int,
    tp: int = 1,
    kv_cache_dtype: str = "auto",
    max_model_len: int = 0,
    gpu_memory_utilization: float = 0.9,
    python_exe: str = "",
) -> list[str]:
    """Build the vLLM OpenAI ``api_server`` argv for storage-faithful capture.

    Bakes in the capture discipline:
      * ``--no-enable-prefix-caching`` -- vLLM automatic prefix caching OFF, so
        an in-GPU APC hit cannot satisfy a request before LMCache L2 records
        the store/load we want to capture.
      * non-MP ``LMCacheConnectorV1`` (``kv_role=kv_both``) via
        ``--kv-transfer-config``.  We DELIBERATELY do NOT use the MP /
        ``lmcache_mp_connector`` variant: it routes through an un-instrumented
        adapter and would capture NOTHING (no raw_block route, no kvio trace).
      * ``--tensor-parallel-size 1`` -- single-rank raw_block (no per-TP device
        partitioning).
      * KV dtype BF16/FP16, never fp8 (``auto`` keeps the model dtype); passing
        an fp8 dtype is rejected because it would change the captured KV bytes.
    """
    if "fp8" in kv_cache_dtype.lower():
        raise ValueError(
            f"kv_cache_dtype must not be fp8 for KV capture, got {kv_cache_dtype!r}"
        )
    exe = python_exe or sys.executable
    # Non-MP connector: LMCacheConnectorV1 (instrumented path), kv_both role.
    kv_transfer = json.dumps(
        {"kv_connector": "LMCacheConnectorV1", "kv_role": "kv_both"}
    )
    argv = [
        exe,
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--model",
        model,
        "--tensor-parallel-size",
        str(int(tp)),
        "--kv-cache-dtype",
        kv_cache_dtype,
        "--no-enable-prefix-caching",
        "--kv-transfer-config",
        kv_transfer,
        "--port",
        str(int(port)),
        "--gpu-memory-utilization",
        str(gpu_memory_utilization),
    ]
    if max_model_len:
        argv += ["--max-model-len", str(int(max_model_len))]
    return argv


# ── Completions request body ────────────────────────────────────────────────


def completion_payload(
    request: SynthRequest,
    *,
    model: str = "",
    temperature: float = 0.0,
    stream: bool = False,
) -> dict:
    """Build the ``/v1/completions`` request body for a ``SynthRequest``.

    Submits the prompt as raw TOKEN IDs (vLLM accepts token-ID prompts) -- no
    chat template, no detokenize/re-tokenize drift.  ``max_tokens =
    output_length`` + ``ignore_eos`` force the exact decode length;
    ``temperature=0`` for determinism.
    """
    return {
        "model": model,
        "prompt": list(request.prompt_token_ids),
        "max_tokens": int(request.output_length),
        "temperature": temperature,
        "ignore_eos": True,
        "stream": bool(stream),
    }


# ── Server environment ──────────────────────────────────────────────────────


def server_env(
    base_env: Optional[dict],
    *,
    lmcache_config_path: str,
    kvio_trace_path: str,
) -> dict:
    """Return a copy of ``base_env`` with the LMCache capture env vars set.

    ``LMCACHE_CONFIG_FILE`` selects the written YAML; ``LMCACHE_KVIO_TRACE``
    tells LMCache where to write the KV-object semantic (identity-join) trace.
    """
    env = dict(base_env if base_env is not None else os.environ)
    env["LMCACHE_CONFIG_FILE"] = str(lmcache_config_path)
    env["LMCACHE_KVIO_TRACE"] = str(kvio_trace_path)
    return env


# ── Readiness poll (follows scripts/fused-quant .../stage1_nsys_profile.py) ──


def wait_server(base_url: str, timeout: int = 600, interval: float = 1.0) -> bool:
    """Poll ``<base_url>/health`` until 200 or ``timeout`` seconds elapse."""
    deadline = time.time() + timeout
    url = base_url.rstrip("/") + "/health"
    while time.time() < deadline:
        try:
            resp = urllib.request.urlopen(url, timeout=2)
            if resp.getcode() == 200:
                return True
        except Exception:
            pass
        time.sleep(interval)
    return False


# ── Open-loop replay (reuses schedule_arrivals for arrival timing) ──────────


def _post_completion(
    base_url: str, payload: dict, timeout: int
) -> tuple[Optional[float], Optional[float], Optional[dict], Optional[str]]:
    """POST one streaming completion; return (ttft_s, e2e_s, usage, error).

    Streaming lets us time first-token (TTFT) vs full completion (e2e).  On any
    error we return the error string so the replay records it without aborting.
    """
    url = base_url.rstrip("/") + "/v1/completions"
    body = json.dumps(payload).encode()
    req = urllib.request.Request(
        url, data=body, headers={"Content-Type": "application/json"}
    )
    t0 = time.time()
    ttft: Optional[float] = None
    usage: Optional[dict] = None
    try:
        resp = urllib.request.urlopen(req, timeout=timeout)
        for raw in resp:
            line = raw.decode("utf-8", "replace").strip()
            if not line or not line.startswith("data:"):
                continue
            data = line[len("data:"):].strip()
            if data == "[DONE]":
                break
            if ttft is None:
                ttft = time.time() - t0
            try:
                chunk = json.loads(data)
                if chunk.get("usage"):
                    usage = chunk["usage"]
            except json.JSONDecodeError:
                pass
        e2e = time.time() - t0
        return ttft, e2e, usage, None
    except Exception as e:  # noqa: BLE001 -- record, do not abort the replay
        return ttft, time.time() - t0, usage, str(e)


def run_replay(
    *,
    base_url: str,
    model: str,
    requests: Sequence[SynthRequest],
    speedup: float = 1.0,
    max_in_flight: Optional[int] = None,
    request_timeout: int = 600,
) -> dict:
    """Open-loop replay of ``requests`` against a live completions endpoint.

    Arrival timing comes from the shared ``schedule_arrivals`` scheduler
    (release at each request's ``timestamp_ms``, compressed by ``speedup``).
    Concurrency is bounded by a semaphore when ``max_in_flight`` is set (open
    loop with admission control).  Per-request TTFT + e2e latency and a run
    summary are returned; nothing here is GPU/vLLM-specific (plain HTTP).
    """
    rows: list[dict] = []
    rows_lock = threading.Lock()
    sem = threading.Semaphore(max_in_flight) if max_in_flight else None
    threads: list[threading.Thread] = []

    def _worker(req: SynthRequest) -> None:
        try:
            payload = completion_payload(req, model=model, stream=True)
            ttft, e2e, usage, err = _post_completion(
                base_url, payload, request_timeout
            )
            row = {
                "index": req.index,
                "timestamp_ms": req.timestamp_ms,
                "prompt_tokens": len(req.prompt_token_ids),
                "output_length": req.output_length,
                "ttft_s": ttft,
                "e2e_s": e2e,
                "usage": usage,
                "error": err,
            }
            with rows_lock:
                rows.append(row)
        finally:
            if sem is not None:
                sem.release()

    wall_t0 = time.time()
    for ev in schedule_arrivals(requests, speedup=speedup):
        # ev is a ScheduledRequest (max_in_flight not passed here, so the
        # scheduler is a plain iterator; admission is enforced by the
        # semaphore below -- the generator's Wait protocol is the alternative).
        now = time.time() - wall_t0
        if ev.release_s > now:
            time.sleep(ev.release_s - now)
        if sem is not None:
            sem.acquire()
        th = threading.Thread(target=_worker, args=(ev.request,), daemon=True)
        th.start()
        threads.append(th)
    for th in threads:
        th.join()
    wall_s = time.time() - wall_t0

    rows.sort(key=lambda r: r["index"])
    ok = [r for r in rows if r["error"] is None]
    ttfts = [r["ttft_s"] for r in ok if r["ttft_s"] is not None]
    e2es = [r["e2e_s"] for r in ok if r["e2e_s"] is not None]
    summary = {
        "wall_clock_s": wall_s,
        "requests": len(rows),
        "requests_ok": len(ok),
        "requests_error": len(rows) - len(ok),
        "total_prompt_tokens": sum(r["prompt_tokens"] for r in rows),
        "ttft_s_mean": (sum(ttfts) / len(ttfts)) if ttfts else None,
        "e2e_s_mean": (sum(e2es) / len(e2es)) if e2es else None,
    }
    return {"rows": rows, "summary": summary}


# ── Best-effort LMCache stats scrape ────────────────────────────────────────


def scrape_lmcache_stats(server_log_path: str | os.PathLike) -> dict:
    """Best-effort prefix-cache hit/store counts scraped from the server log.

    LMCache logs retrieve/store activity; we grep the server log for lines that
    mention hit/store/retrieve rather than depend on a stats endpoint that may
    not be exposed.  Never raises -- returns ``{}`` on any problem.
    """
    stats: dict = {"available": False}
    try:
        with open(server_log_path, "r", errors="replace") as f:
            text = f.read()
    except Exception:
        return stats
    hits = len(re.findall(r"\b(hit|retrieved|reused)\b", text, re.IGNORECASE))
    stores = len(re.findall(r"\b(store[sd]?|saved|offload)\b", text, re.IGNORECASE))
    stats.update(
        {
            "available": True,
            "log_hit_mentions": hits,
            "log_store_mentions": stores,
            "note": "coarse log-scrape; replace with a stats endpoint when exposed",
        }
    )
    return stats
