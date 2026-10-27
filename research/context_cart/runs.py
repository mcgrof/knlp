# SPDX-License-Identifier: MIT
"""Run directory bookkeeping: resolved configs, environment, source lock."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


def dump_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(
        value, indent=2, sort_keys=True, ensure_ascii=False, default=_default
    )
    path.write_text(text + "\n", encoding="utf-8")


def _default(value: Any) -> Any:
    if dataclasses.is_dataclass(value):
        return dataclasses.asdict(value)
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "tolist"):
        return value.tolist()
    raise TypeError(f"cannot serialize {type(value).__name__}")


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def file_digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def git_revision(root: Path) -> dict[str, Any]:
    def run(*args: str) -> str:
        return subprocess.run(
            ["git", *args], cwd=root, check=True, capture_output=True, text=True
        ).stdout.strip()

    try:
        head = run("rev-parse", "HEAD")
        dirty = bool(run("status", "--porcelain", "--untracked-files=no"))
        return {"revision": head, "dirty": dirty}
    except Exception as error:  # pragma: no cover - environment dependent
        return {"revision": None, "dirty": None, "error": str(error)}


def environment() -> dict[str, Any]:
    info: dict[str, Any] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "processors_visible": os.cpu_count(),
        "time_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    try:
        quota = Path("/sys/fs/cgroup/cpu.max").read_text().split()
        info["cgroup_cpu_max"] = quota
    except Exception:
        pass
    try:
        import torch

        info["torch"] = torch.__version__
        info["cuda"] = torch.version.cuda
        info["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            info["gpu_name"] = torch.cuda.get_device_name(0)
            props = torch.cuda.get_device_properties(0)
            info["gpu_memory_bytes"] = int(props.total_memory)
            info["gpu_capability"] = f"{props.major}.{props.minor}"
            info["gpu_count"] = torch.cuda.device_count()
    except Exception as error:  # pragma: no cover
        info["torch_error"] = str(error)
    for name in ("transformers", "peft", "safetensors", "numpy"):
        try:
            module = __import__(name)
            info[name] = getattr(module, "__version__", "unknown")
        except Exception:
            info[name] = None
    return info


def peak_memory() -> dict[str, int]:
    try:
        import torch

        if torch.cuda.is_available():
            return {
                "max_allocated_bytes": int(torch.cuda.max_memory_allocated()),
                "max_reserved_bytes": int(torch.cuda.max_memory_reserved()),
            }
    except Exception:
        pass
    return {}


def reset_peak_memory() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
    except Exception:
        pass


def usable_threads() -> int:
    """Honour a container CPU quota instead of the host processor count."""
    try:
        quota, period = Path("/sys/fs/cgroup/cpu.max").read_text().split()
        if quota != "max":
            return max(1, int(int(quota) / int(period)))
    except Exception:
        pass
    return max(1, os.cpu_count() or 1)


def configure_threads() -> int:
    threads = usable_threads()
    os.environ.setdefault("OMP_NUM_THREADS", str(threads))
    os.environ.setdefault("MKL_NUM_THREADS", str(threads))
    try:
        import torch

        torch.set_num_threads(threads)
    except Exception:
        pass
    return threads


class Stopwatch:
    def __init__(self, sync: bool = False):
        self.sync = sync
        self.marks: dict[str, float] = {}

    def _now(self) -> float:
        if self.sync:
            import torch

            if torch.cuda.is_available():
                torch.cuda.synchronize()
        return time.perf_counter()

    def start(self, name: str) -> None:
        self.marks[f"{name}:start"] = self._now()

    def stop(self, name: str) -> float:
        seconds = self._now() - self.marks[f"{name}:start"]
        self.marks[name] = self.marks.get(name, 0.0) + seconds
        return seconds

    def report(self) -> dict[str, float]:
        return {k: v for k, v in self.marks.items() if not k.endswith(":start")}
