"""Parse .config and expose typed access for the rest of the package."""

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Any


def _parse_value(raw: str) -> Any:
    raw = raw.strip()
    if raw == "y":
        return True
    if raw == "n":
        return False
    if raw.startswith('"') and raw.endswith('"'):
        return raw[1:-1]
    try:
        return int(raw)
    except ValueError:
        return raw


def parse_config(path: str | Path) -> dict[str, Any]:
    """Read a Kconfig-format .config file into a flat dict."""
    out: dict[str, Any] = {}
    p = Path(path)
    if not p.exists():
        return out
    for line in p.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        key, _, raw = line.partition("=")
        out[key.strip()] = _parse_value(raw)
    return out


@dataclass
class DecodeConfig:
    """Typed view of CONFIG_KNLP_* values from .config."""

    profile: str = "decode"
    worktree_root: str = ".."
    results_root: str = "./results/decode"
    build_root: str = "./build/decode"

    vllm_repo: str = "https://github.com/mcgrof/vllm.git"
    vllm_ref: str = "asymmetric-kv-plumbing"
    vllm_dir: str = "vllm"
    flashinfer_repo: str = "https://github.com/mcgrof/flashinfer.git"
    flashinfer_ref: str = "asym-prefill-refactor-stage"
    flashinfer_dir: str = "flashinfer"
    lmcache_repo: str = "https://github.com/mcgrof/LMCache.git"
    lmcache_ref: str = "asymmetric-kv-codec"
    lmcache_dir: str = "lmcache"

    qwen25_7b: str = "Qwen/Qwen2.5-7B-Instruct"
    wikitext: str = "wikitext"
    gsm8k: str = "gsm8k"

    enable_wandb: bool = False
    wandb_project: str = "knlp-memory-decode"
    wandb_entity: str = ""
    wandb_mode: str = "online"
    enable_trackerio: bool = False
    trackerio_project: str = "knlp-memory-decode"
    trackerio_url: str = ""

    upload_artifacts: bool = False
    allow_hardware_skips: bool = True
    fail_on_dirty_git: bool = True
    require_pinned_refs: bool = False
    parallel_gpus: str = "auto"
    max_parallel_gpu_jobs: int = 8

    raw: dict[str, Any] = None  # type: ignore[assignment]

    @classmethod
    def from_file(cls, path: str | Path) -> "DecodeConfig":
        raw = parse_config(path)
        get = lambda k, d: raw.get(k, d)
        return cls(
            profile=get("CONFIG_KNLP_REPRODUCE_PROFILE", "decode"),
            worktree_root=get("CONFIG_KNLP_WORKTREE_ROOT", ".."),
            results_root=get("CONFIG_KNLP_RESULTS_ROOT", "./results/decode"),
            build_root=get("CONFIG_KNLP_BUILD_ROOT", "./build/decode"),
            vllm_repo=get("CONFIG_KNLP_VLLM_REPO", cls.vllm_repo),
            vllm_ref=get("CONFIG_KNLP_VLLM_REF", cls.vllm_ref),
            vllm_dir=get("CONFIG_KNLP_VLLM_DIR", cls.vllm_dir),
            flashinfer_repo=get("CONFIG_KNLP_FLASHINFER_REPO", cls.flashinfer_repo),
            flashinfer_ref=get("CONFIG_KNLP_FLASHINFER_REF", cls.flashinfer_ref),
            flashinfer_dir=get("CONFIG_KNLP_FLASHINFER_DIR", cls.flashinfer_dir),
            lmcache_repo=get("CONFIG_KNLP_LMCACHE_REPO", cls.lmcache_repo),
            lmcache_ref=get("CONFIG_KNLP_LMCACHE_REF", cls.lmcache_ref),
            lmcache_dir=get("CONFIG_KNLP_LMCACHE_DIR", cls.lmcache_dir),

            qwen25_7b=get("CONFIG_KNLP_MODEL_QWEN25_7B", cls.qwen25_7b),
            wikitext=get("CONFIG_KNLP_DATASET_WIKITEXT", cls.wikitext),
            gsm8k=get("CONFIG_KNLP_DATASET_GSM8K", cls.gsm8k),
            enable_wandb=bool(get("CONFIG_KNLP_ENABLE_WANDB", False)),
            wandb_project=get("CONFIG_KNLP_WANDB_PROJECT", cls.wandb_project),
            wandb_entity=get("CONFIG_KNLP_WANDB_ENTITY", ""),
            wandb_mode=get("CONFIG_KNLP_WANDB_MODE", cls.wandb_mode),
            enable_trackerio=bool(get("CONFIG_KNLP_ENABLE_TRACKERIO", False)),
            trackerio_project=get(
                "CONFIG_KNLP_TRACKERIO_PROJECT", cls.trackerio_project
            ),
            trackerio_url=get("CONFIG_KNLP_TRACKERIO_URL", ""),
            upload_artifacts=bool(get("CONFIG_KNLP_UPLOAD_ARTIFACTS", False)),
            allow_hardware_skips=bool(get("CONFIG_KNLP_ALLOW_HARDWARE_SKIPS", True)),
            fail_on_dirty_git=bool(get("CONFIG_KNLP_FAIL_ON_DIRTY_GIT", True)),
            require_pinned_refs=bool(get("CONFIG_KNLP_REQUIRE_PINNED_REFS", False)),
            parallel_gpus=get("CONFIG_KNLP_PARALLEL_GPUS", "auto"),
            max_parallel_gpu_jobs=int(get("CONFIG_KNLP_MAX_PARALLEL_GPU_JOBS", 8)),
            raw=raw,
        )

    @property
    def model_qwen25_7b(self) -> str:
        return self.qwen25_7b

    def is_enabled(self) -> bool:
        return bool(self.raw and self.raw.get("CONFIG_KNLP_REPRODUCE_DECODE"))

    def repos(self) -> list[tuple[str, str, str, str]]:
        """Return [(name, repo_url, ref, abs_clone_path), ...]."""
        wt = Path(self.worktree_root).resolve()
        return [
            ("vllm", self.vllm_repo, self.vllm_ref, str(wt / self.vllm_dir)),
            (
                "flashinfer",
                self.flashinfer_repo,
                self.flashinfer_ref,
                str(wt / self.flashinfer_dir),
            ),
            (
                "lmcache",
                self.lmcache_repo,
                self.lmcache_ref,
                str(wt / self.lmcache_dir),
            ),
        ]
