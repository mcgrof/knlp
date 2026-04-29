"""Manifest for a reproduction run.

Captured once at run start and saved to results/<profile>/<run_id>/manifest.json.
Stable schema_version 1; schema bumps are major events.
"""

from __future__ import annotations
import datetime as _dt
import json
import os
import subprocess
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .decode_config import DecodeConfig
from .hardware import HostInfo


SCHEMA_VERSION = 1


@dataclass
class GitInfo:
    remote: str = ""
    ref: str = ""
    commit: str = ""
    dirty: bool = False
    path: str = ""


def _run(cmd: list[str], cwd: str | None = None) -> str:
    try:
        return (
            subprocess.check_output(cmd, cwd=cwd, stderr=subprocess.DEVNULL, timeout=30)
            .decode()
            .strip()
        )
    except Exception:
        return ""


def git_info(path: str) -> GitInfo:
    p = Path(path)
    if not (p / ".git").exists():
        return GitInfo(path=str(p))
    return GitInfo(
        path=str(p),
        remote=_run(["git", "config", "--get", "remote.origin.url"], cwd=str(p)),
        ref=_run(["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=str(p)),
        commit=_run(["git", "rev-parse", "HEAD"], cwd=str(p)),
        dirty=bool(_run(["git", "status", "--porcelain"], cwd=str(p))),
    )


@dataclass
class Manifest:
    schema_version: int = SCHEMA_VERSION
    run_id: str = ""
    profile: str = ""
    started_at: str = ""
    host: dict[str, Any] = field(default_factory=dict)
    hardware: dict[str, Any] = field(default_factory=dict)
    git: dict[str, dict[str, Any]] = field(default_factory=dict)
    models: dict[str, dict[str, Any]] = field(default_factory=dict)
    datasets: dict[str, dict[str, Any]] = field(default_factory=dict)
    telemetry: dict[str, Any] = field(default_factory=dict)
    config: dict[str, Any] = field(default_factory=dict)
    stages: list[dict[str, Any]] = field(default_factory=list)


def make_run_id(profile: str) -> str:
    return f"{profile}-{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}"


def build_manifest(cfg: DecodeConfig, host: HostInfo, run_id: str) -> Manifest:
    m = Manifest(
        run_id=run_id,
        profile=cfg.profile,
        started_at=_dt.datetime.utcnow().isoformat() + "Z",
    )
    m.host = {"hostname": host.hostname, "provider": host.provider}
    m.hardware = {
        "gpu_count": host.gpu_count,
        "gpu_names": host.gpu_names,
        "gpu_memory_mb": host.gpu_memory_mb,
        "cuda_version": host.cuda_version,
        "rocm_version": host.rocm_version,
        "driver_version": host.driver_version,
        "free_disk_gb": round(host.free_disk_gb, 1),
        "total_ram_gb": round(host.total_ram_gb, 1),
        "python_version": host.python_version,
        "cmake_version": host.cmake_version,
    }
    knlp_root = Path(__file__).resolve().parents[3]
    m.git = {"knlp": asdict(git_info(str(knlp_root)))}
    for name, _, _, path in cfg.repos():
        if Path(path).exists():
            m.git[name] = asdict(git_info(path))
    m.models = {"qwen25_7b": {"hf_id": cfg.qwen25_7b}}
    m.datasets = {
        "wikitext": {"name": cfg.wikitext},
        "gsm8k": {"name": cfg.gsm8k},
    }
    m.telemetry = {
        "wandb": {
            "enabled": cfg.enable_wandb,
            "project": cfg.wandb_project,
            "entity": cfg.wandb_entity,
            "mode": cfg.wandb_mode,
        },
        "trackerio": {
            "enabled": cfg.enable_trackerio,
            "project": cfg.trackerio_project,
            "url": cfg.trackerio_url,
        },
    }
    m.config = {
        k: v for k, v in (cfg.raw or {}).items() if k.startswith("CONFIG_KNLP_")
    }
    return m


def save_manifest(m: Manifest, results_dir: Path) -> Path:
    results_dir.mkdir(parents=True, exist_ok=True)
    path = results_dir / "manifest.json"
    path.write_text(json.dumps(asdict(m), indent=2, default=str))
    return path


def save_environment(results_dir: Path) -> None:
    """Snapshot pip freeze, nvidia-smi -q, torch env into the run dir."""
    results_dir.mkdir(parents=True, exist_ok=True)
    pip = _run(["pip", "freeze"]) or _run(["pip3", "freeze"])
    if pip:
        (results_dir / "pip-freeze.txt").write_text(pip)
    nv = _run(["nvidia-smi", "-q"])
    if nv:
        (results_dir / "nvidia-smi-q.txt").write_text(nv)
    torch_env = _run(["python3", "-m", "torch.utils.collect_env"])
    if torch_env:
        (results_dir / "torch-env.txt").write_text(torch_env)
