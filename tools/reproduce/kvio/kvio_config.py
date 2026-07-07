# SPDX-License-Identifier: MIT
"""kvio reproduce config.

Maps a kernel-style ``.config`` (CONFIG_KNLP_KVIO_* keys) into the attributes
the mooncake / content replay stages read off ``ctx.cfg``.  All keys are
namespaced under ``CONFIG_KNLP_KVIO_`` so kvio never collides with any other
reproduce orchestrator's symbols.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


def _as_bool(v: Any) -> bool:
    return str(v).strip().lower() in ("y", "1", "true", "yes")


def _as_int(v: Any, default: int) -> int:
    try:
        return int(v)
    except (TypeError, ValueError):
        return default


@dataclass
class KvioConfig:
    # ── Orchestrator: profile + companion repos ───────────────────
    profile: str = "content"  # "mooncake" | "content"
    worktree_root: str = ".."
    results_root: str = "./results/kvio"
    vllm_repo: str = "https://github.com/mcgrof/vllm.git"
    vllm_ref: str = "asymmetric-kv-plumbing"
    vllm_dir: str = "vllm"
    lmcache_repo: str = "https://github.com/mcgrof/LMCache.git"
    lmcache_ref: str = "kvio"
    lmcache_dir: str = "lmcache"

    # ── Mooncake real-trace replay ────────────────────────────────
    mooncake_trace_path: str = ""
    mooncake_model: str = "Qwen/Qwen2.5-7B-Instruct"
    mooncake_max_requests: int = 0
    mooncake_speedup: int = 1
    mooncake_vocab_size: int = 32000
    mooncake_seed: int = 0

    # ── Content-bearing real-trace replay + KV capture ────────────
    content_dataset: str = "lmsys"
    content_model: str = "Qwen/Qwen2.5-7B-Instruct"
    content_max_requests: int = 0
    content_speedup: int = 1
    content_seed: int = 0
    content_save_decode_cache: bool = False
    content_l2_device: str = ""
    content_l2_capacity_gb: int = 32
    content_nvme_tracer: str = ""
    content_port: int = 8000

    # ── Record (Phase 1 capture) / replay (Phase 2 GPU-free) ──────
    record_dir: str = ""
    backend: str = "lmcache"
    smoke: bool = False
    replay_driver: str = ""
    replay_device: str = ""
    replay_engine: str = "io_uring"

    raw: dict = field(default_factory=dict)

    @classmethod
    def from_raw(cls, raw: dict) -> "KvioConfig":
        g = raw.get
        return cls(
            profile=g("CONFIG_KNLP_KVIO_PROFILE", cls.profile),
            worktree_root=g("CONFIG_KNLP_KVIO_WORKTREE_ROOT", cls.worktree_root),
            results_root=g("CONFIG_KNLP_KVIO_RESULTS_ROOT", cls.results_root),
            vllm_repo=g("CONFIG_KNLP_KVIO_VLLM_REPO", cls.vllm_repo),
            vllm_ref=g("CONFIG_KNLP_KVIO_VLLM_REF", cls.vllm_ref),
            vllm_dir=g("CONFIG_KNLP_KVIO_VLLM_DIR", cls.vllm_dir),
            lmcache_repo=g("CONFIG_KNLP_KVIO_LMCACHE_REPO", cls.lmcache_repo),
            lmcache_ref=g("CONFIG_KNLP_KVIO_LMCACHE_REF", cls.lmcache_ref),
            lmcache_dir=g("CONFIG_KNLP_KVIO_LMCACHE_DIR", cls.lmcache_dir),
            mooncake_trace_path=g("CONFIG_KNLP_KVIO_MOONCAKE_TRACE_PATH", ""),
            mooncake_model=g("CONFIG_KNLP_KVIO_MOONCAKE_MODEL", cls.mooncake_model),
            mooncake_max_requests=_as_int(
                g("CONFIG_KNLP_KVIO_MOONCAKE_MAX_REQUESTS", 0), 0
            ),
            mooncake_speedup=_as_int(g("CONFIG_KNLP_KVIO_MOONCAKE_SPEEDUP", 1), 1),
            mooncake_vocab_size=_as_int(
                g("CONFIG_KNLP_KVIO_MOONCAKE_VOCAB_SIZE", 32000), 32000
            ),
            mooncake_seed=_as_int(g("CONFIG_KNLP_KVIO_MOONCAKE_SEED", 0), 0),
            content_dataset=g("CONFIG_KNLP_KVIO_CONTENT_DATASET", cls.content_dataset),
            content_model=g("CONFIG_KNLP_KVIO_CONTENT_MODEL", cls.content_model),
            content_max_requests=_as_int(
                g("CONFIG_KNLP_KVIO_CONTENT_MAX_REQUESTS", 0), 0
            ),
            content_speedup=_as_int(g("CONFIG_KNLP_KVIO_CONTENT_SPEEDUP", 1), 1),
            content_seed=_as_int(g("CONFIG_KNLP_KVIO_CONTENT_SEED", 0), 0),
            content_save_decode_cache=_as_bool(
                g("CONFIG_KNLP_KVIO_CONTENT_SAVE_DECODE_CACHE", "n")
            ),
            content_l2_device=g("CONFIG_KNLP_KVIO_CONTENT_L2_DEVICE", ""),
            content_l2_capacity_gb=_as_int(
                g("CONFIG_KNLP_KVIO_CONTENT_L2_CAPACITY_GB", 32), 32
            ),
            content_nvme_tracer=g("CONFIG_KNLP_KVIO_CONTENT_NVME_TRACER", ""),
            content_port=_as_int(g("CONFIG_KNLP_KVIO_CONTENT_PORT", 8000), 8000),
            record_dir=g("CONFIG_KNLP_KVIO_RECORD_DIR", ""),
            backend=g("CONFIG_KNLP_KVIO_BACKEND", cls.backend),
            smoke=_as_bool(g("CONFIG_KNLP_KVIO_SMOKE", "n")),
            replay_driver=g("CONFIG_KNLP_KVIO_REPLAY_DRIVER", ""),
            replay_device=g("CONFIG_KNLP_KVIO_REPLAY_DEVICE", ""),
            replay_engine=g("CONFIG_KNLP_KVIO_REPLAY_ENGINE", cls.replay_engine),
            raw=raw,
        )
