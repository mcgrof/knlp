"""Pre-flight checks that surface problems before we burn build/GPU time."""
from __future__ import annotations
import shutil
from pathlib import Path

from .decode_config import DecodeConfig
from .hardware import HostInfo


# Minimum disk requirements per profile (GB).
MIN_DISK_GB = {"decode": 200, "decode-sat": 100, "decode-full": 500}


def run_checks(cfg: DecodeConfig, host: HostInfo) -> list[str]:
    """Return a list of human-readable issues.  Empty list means OK."""
    issues: list[str] = []

    # --- Tools ----------------------------------------------------------
    if not host.has_git:
        issues.append("git not found in PATH")
    if not host.has_pip:
        issues.append("pip/pip3 not found in PATH")
    if not host.has_cmake:
        issues.append("cmake not found in PATH")
    elif host.cmake_version:
        try:
            major = int(host.cmake_version.split(".")[0])
            if major < 4:
                issues.append(
                    f"cmake {host.cmake_version} is too old; vLLM build wants cmake>=4. "
                    "Run: pip install --upgrade cmake"
                )
        except ValueError:
            pass

    # --- Hardware -------------------------------------------------------
    if host.gpu_count == 0:
        if cfg.profile == "decode" or cfg.profile == "decode-full":
            issues.append("no GPU detected; decode/decode-full profiles require an NVIDIA H100")
    else:
        # Memory sanity for 7B
        if host.gpu_memory_mb and max(host.gpu_memory_mb) < 70_000:
            issues.append(
                f"GPU memory {max(host.gpu_memory_mb)} MB < 70 GB; "
                "Qwen2.5-7B-Instruct full-stack stages need ≥80 GB"
            )

    # --- Disk -----------------------------------------------------------
    need = MIN_DISK_GB.get(cfg.profile, 200)
    if host.free_disk_gb and host.free_disk_gb < need:
        issues.append(
            f"only {host.free_disk_gb:.1f} GB free; profile {cfg.profile} wants ≥{need} GB"
        )

    # --- Secrets / network ----------------------------------------------
    if not host.has_hf_token:
        issues.append(
            "HF token not present (HF_TOKEN env or ~/.cache/huggingface/token); "
            "Qwen2.5 download may fail"
        )
    if cfg.enable_wandb and not host.has_wandb_key:
        issues.append("CONFIG_KNLP_ENABLE_WANDB=y but WANDB_API_KEY not set")
    if cfg.enable_trackerio and not host.has_trackerio_key:
        issues.append("CONFIG_KNLP_ENABLE_TRACKERIO=y but TRACKERIO_API_KEY/TRACKERIO_TOKEN not set")

    # --- Companion-repo paths -------------------------------------------
    wt = Path(cfg.worktree_root).resolve()
    if not wt.exists():
        issues.append(f"worktree root {wt} does not exist; create it or set CONFIG_KNLP_WORKTREE_ROOT")

    # --- Pinned-ref policy ----------------------------------------------
    if cfg.require_pinned_refs:
        for name, _, ref, _ in cfg.repos():
            if len(ref) != 40 or not all(c in "0123456789abcdef" for c in ref):
                issues.append(
                    f"CONFIG_KNLP_REQUIRE_PINNED_REFS=y but {name} ref '{ref}' is not a 40-char SHA"
                )

    return issues


def render(cfg, host, issues: list[str]) -> str:
    if not issues:
        return "decode-doctor: all checks passed."
    lines = ["decode-doctor: issues found:"]
    for i in issues:
        lines.append(f"  - {i}")
    return "\n".join(lines)
