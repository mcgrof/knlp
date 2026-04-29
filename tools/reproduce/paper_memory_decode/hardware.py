"""Detect host hardware and reproduction-relevant capabilities."""

from __future__ import annotations
import os
import shutil
import subprocess
from dataclasses import dataclass, field


@dataclass
class HostInfo:
    hostname: str = ""
    gpu_count: int = 0
    gpu_names: list[str] = field(default_factory=list)
    gpu_memory_mb: list[int] = field(default_factory=list)
    cuda_version: str = ""
    rocm_version: str = ""
    driver_version: str = ""
    free_disk_gb: float = 0.0
    total_ram_gb: float = 0.0
    python_version: str = ""
    has_hf_token: bool = False
    has_wandb_key: bool = False
    has_trackerio_key: bool = False
    has_git: bool = False
    has_pip: bool = False
    has_cmake: bool = False
    cmake_version: str = ""
    provider: dict[str, str] = field(default_factory=dict)


def _run(cmd: list[str], timeout: int = 10) -> str:
    try:
        out = subprocess.check_output(cmd, stderr=subprocess.DEVNULL, timeout=timeout)
        return out.decode().strip()
    except Exception:
        return ""


def detect() -> HostInfo:
    h = HostInfo()
    h.hostname = _run(["hostname"]) or os.uname().nodename
    h.python_version = _run(["python3", "--version"]).replace("Python ", "")
    h.has_git = bool(shutil.which("git"))
    h.has_pip = bool(shutil.which("pip") or shutil.which("pip3"))
    h.has_cmake = bool(shutil.which("cmake"))
    if h.has_cmake:
        ver = _run(["cmake", "--version"]).splitlines()
        if ver:
            h.cmake_version = ver[0].replace("cmake version ", "")

    # NVIDIA GPUs
    if shutil.which("nvidia-smi"):
        names = _run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"]
        ).splitlines()
        mems = _run(
            ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"]
        ).splitlines()
        drv = _run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"]
        ).splitlines()
        h.gpu_count = len([n for n in names if n.strip()])
        h.gpu_names = [n.strip() for n in names if n.strip()]
        h.gpu_memory_mb = [int(m.strip()) for m in mems if m.strip().isdigit()]
        if drv:
            h.driver_version = drv[0].strip()
        cuda_h = _run(["nvcc", "--version"])
        if "release" in cuda_h:
            for tok in cuda_h.split():
                if tok.startswith("V") and "." in tok:
                    h.cuda_version = tok.lstrip("V")
                    break

    # AMD ROCm
    if shutil.which("rocm-smi"):
        rocm = _run(["rocm-smi", "--version"]).splitlines()
        if rocm:
            h.rocm_version = rocm[0]

    # Disk + RAM — check multiple candidate paths and keep the max.
    # On RunPod the large volume is at /runpod-volume, not the root fs.
    _disk_candidates = [
        os.environ.get("KNLP_WORKTREE_ROOT", ".."),
        "/runpod-volume",
        "/workspace",
        ".",
    ]
    _best_free = 0.0
    for _dp in _disk_candidates:
        try:
            _st = os.statvfs(_dp)
            _free = (_st.f_bavail * _st.f_frsize) / (1024**3)
            if _free > _best_free:
                _best_free = _free
        except Exception:
            pass
    h.free_disk_gb = _best_free
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    kb = int(line.split()[1])
                    h.total_ram_gb = kb / (1024**2)
                    break
    except Exception:
        pass

    # Secrets (existence only — never read the value)
    h.has_hf_token = bool(os.environ.get("HF_TOKEN")) or os.path.exists(
        os.path.expanduser("~/.cache/huggingface/token")
    )
    h.has_wandb_key = bool(os.environ.get("WANDB_API_KEY"))
    h.has_trackerio_key = bool(
        os.environ.get("TRACKERIO_API_KEY") or os.environ.get("TRACKERIO_TOKEN")
    )

    # Provider tags (best-effort, no secrets)
    for env_key, label in [
        ("RUNPOD_POD_ID", "runpod"),
        ("PRIMEINTELLECT_JOB_ID", "primeintellect"),
        ("AWS_INSTANCE_ID", "aws"),
        ("GCP_INSTANCE_ID", "gcp"),
        ("LAMBDA_INSTANCE_ID", "lambda"),
    ]:
        if os.environ.get(env_key):
            h.provider[label] = os.environ[env_key]

    return h


def headline_gpu(h: HostInfo) -> str:
    return h.gpu_names[0] if h.gpu_names else "no GPU detected"
