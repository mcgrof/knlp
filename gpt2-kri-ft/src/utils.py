"""Small shared helpers: seeding, device selection, precision pick."""

from __future__ import annotations

import os
import random
import sys
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def pick_dtype(precision: str = "auto") -> torch.dtype:
    """Choose dtype for training/eval.

    "auto" prefers bf16 on devices that support it, fp16 otherwise on
    GPU, fp32 on CPU. The W7900 supports bf16.
    """
    p = precision.lower()
    if p == "fp32":
        return torch.float32
    if p == "fp16":
        return torch.float16
    if p == "bf16":
        return torch.bfloat16
    if not torch.cuda.is_available():
        return torch.float32
    if torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


@dataclass
class StepLog:
    step: int
    loss: float
    lr: float
    tokens_per_sec: float
    seconds: float
    extra: Optional[dict] = None


def log_step(log: StepLog, fh=sys.stdout) -> None:
    extra = ""
    if log.extra:
        extra = " " + " ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" for k, v in log.extra.items())
    print(
        f"[step {log.step:6d}] loss={log.loss:.4f} lr={log.lr:.2e} "
        f"tok/s={log.tokens_per_sec:8.0f} t={log.seconds:5.1f}s{extra}",
        file=fh,
        flush=True,
    )


def banner(msg: str) -> None:
    bar = "=" * max(8, len(msg) + 4)
    print(bar, flush=True)
    print(f"  {msg}", flush=True)
    print(bar, flush=True)


def report_device() -> None:
    if torch.cuda.is_available():
        n = torch.cuda.device_count()
        for i in range(n):
            name = torch.cuda.get_device_name(i)
            total = torch.cuda.get_device_properties(i).total_memory / 1e9
            print(f"gpu{i}: {name}  total={total:.1f} GB")
        print(f"torch={torch.__version__}  cuda={torch.version.cuda}  hip={getattr(torch.version, 'hip', None)}")
    else:
        print("no CUDA/ROCm device visible; running on CPU")


class Timer:
    def __init__(self) -> None:
        self.t0 = time.time()
        self.last = self.t0

    def lap(self) -> float:
        t = time.time()
        dt = t - self.last
        self.last = t
        return dt

    def total(self) -> float:
        return time.time() - self.t0
