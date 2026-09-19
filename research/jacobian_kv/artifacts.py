# SPDX-License-Identifier: GPL-2.0
"""Run provenance: what was read, by what code, at what cost, and what stopped.

The contract is adapted from ``tools/kv/fp8_failure/common.py`` -- immutable
manifests, config hashing, disjoint calibration and evaluation indices, no
silent overwrite -- with four changes this lane needs.

First, the execution-order addition: hash the *tensors* a run actually read,
not the directory they came from.  A sibling workflow lost time to an evaluator
that recorded the directory, so a mis-staged arm read as a clean result.

Second, cost is recorded even for free local runs.  Every later hardware
decision is a dollars-per-evaluated-token comparison, and the local runs are
the only free calibration point for what an evaluated token costs here.

Third, ``git_state`` is called with an explicit repository path.  The inherited
default resolves one directory too deep and silently degrades to ``"unknown"``,
which would put a run with no provenance in the archive looking like any other.

Fourth, ``versions()`` reports through ``importlib.metadata`` rather than
importing serving stacks as a side effect of writing a manifest.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import subprocess
import sys
import time
from typing import Any, Optional

SCHEMA_VERSION = 2

MEASUREMENT_LEVELS = (
    "vjp_probe",
    "fake_quant_teacher_forced",
    "real_quant_teacher_forced",
    "dynamic_cache_decode",
)


# ---------------------------------------------------------------------------
# provenance primitives
# ---------------------------------------------------------------------------


def repo_root() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def git_state(repo: Optional[str] = None):
    """``(commit, dirty)``, raising rather than degrading to ``"unknown"``."""
    repo = repo or repo_root()
    commit = subprocess.check_output(
        ["git", "-C", repo, "rev-parse", "HEAD"], text=True
    ).strip()
    dirty = bool(
        subprocess.check_output(
            ["git", "-C", repo, "status", "--porcelain"], text=True
        ).strip()
    )
    if not commit:
        raise RuntimeError(f"no git commit for {repo}")
    return commit, dirty


def versions() -> dict:
    """Package versions without importing the packages."""
    from importlib.metadata import PackageNotFoundError, version

    out = {"python": sys.version.split()[0]}
    for pkg in ("torch", "transformers", "numpy", "safetensors", "accelerate"):
        try:
            out[pkg] = version(pkg)
        except PackageNotFoundError:
            out[pkg] = None
    try:
        import torch

        out["torch_hip"] = torch.version.hip
        out["torch_cuda"] = torch.version.cuda
    except Exception:  # noqa: BLE001
        pass
    return out


def config_hash(obj: Any) -> str:
    payload = json.dumps(obj, sort_keys=True, default=str).encode()
    return hashlib.sha256(payload).hexdigest()[:16]


def sha256_file(path: str, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def sha256_tensor(t) -> str:
    """Hash a tensor's actual bytes, so a mis-staged arm cannot read as clean."""
    import torch

    x = t.detach().to("cpu").contiguous()
    if x.dtype in (torch.bfloat16, torch.float16):
        x = x.to(torch.float32)
    return hashlib.sha256(x.numpy().tobytes()).hexdigest()[:32]


def sha256_ids(ids) -> str:
    import torch

    x = ids.detach().to("cpu").to(torch.int64).contiguous()
    return hashlib.sha256(x.numpy().tobytes()).hexdigest()[:32]


# ---------------------------------------------------------------------------
# the manifest
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class JKVManifest:
    """One immutable record per run.  ``finalize`` refuses an unusable one."""

    schema_version: int = SCHEMA_VERSION
    run_id: str = ""
    stage: str = ""

    source_model_id: str = ""
    source_model_revision: str = ""
    target_model_id: str = ""
    target_model_revision: str = ""
    tokenizer_sha256: str = ""
    input_artifact_sha256: dict = dataclasses.field(default_factory=dict)

    dtype: str = "float32"
    attn_backend: str = "eager"
    device: str = ""
    cache_layout: str = "B,n_kv_heads,T,head_dim"
    rope_status: str = ""
    rope_theta_source: float = 0.0
    rope_theta_target: float = 0.0
    rope_scaling_source: Any = None
    rope_scaling_target: Any = None

    kv_split: str = ""
    context_length: int = 0
    continuation_length: int = 0
    selected_layers: tuple = ()
    mapper_rank: int = 0
    ridge_lambda: float = 0.0
    probe_count: int = 0
    probe_distribution: str = "categorical_fisher"
    measurement_level: str = "fake_quant_teacher_forced"

    calib_dataset: str = ""
    calib_indices: tuple = ()
    eval_dataset: str = ""
    eval_indices: tuple = ()
    seed: int = 0

    code_commit: str = ""
    code_dirty: bool = False
    results_commit: str = ""
    config_hash: str = ""
    cmdline: str = ""
    versions: dict = dataclasses.field(default_factory=dict)

    gpu_model: str = ""
    gpu_count: int = 1
    gpu_hours: float = 0.0
    wall_hours: float = 0.0
    observed_tokens_per_s: float = 0.0
    evaluated_tokens: int = 0
    hourly_price_usd: float = 0.0
    billed_cost_usd: float = 0.0
    pod_id: str = ""
    peak_mem_bytes: int = 0

    stopped_arms: tuple = ()
    notes: str = ""

    def finalize(self) -> dict:
        if self.measurement_level not in MEASUREMENT_LEVELS:
            raise ValueError(f"unknown measurement level {self.measurement_level!r}")
        if not self.target_model_revision or self.target_model_revision in ("main", ""):
            raise ValueError("target_model_revision must be a pinned revision")
        overlap = set(self.calib_indices) & set(self.eval_indices)
        if overlap:
            raise ValueError(
                f"calibration and evaluation overlap on {sorted(overlap)[:8]}"
            )
        if not self.code_commit or self.code_commit == "unknown":
            raise ValueError("refusing to record a run with no code provenance")
        d = dataclasses.asdict(self)
        d["finalized_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        d["argv"] = " ".join(sys.argv)
        return d


# ---------------------------------------------------------------------------
# writers
# ---------------------------------------------------------------------------


def ensure_run_dir(base: str, run_id: str) -> str:
    """A fresh directory per run.  Refuses to reuse one that has results."""
    path = os.path.join(base, run_id)
    if os.path.isdir(path) and any(
        f.endswith((".jsonl", ".json")) for f in os.listdir(path)
    ):
        raise FileExistsError(
            f"{path} already holds results; pick a new run id rather than overwriting"
        )
    os.makedirs(path, exist_ok=True)
    return path


def write_json(path: str, obj: Any) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(obj, fh, indent=2, sort_keys=True, default=str)
    os.replace(tmp, path)


class JsonlWriter:
    """Append-per-row so a crashed run still leaves everything it measured."""

    def __init__(self, path: str):
        self.path = path
        self._fh = open(path, "a", buffering=1)

    def write(self, row: dict) -> None:
        self._fh.write(json.dumps(row, sort_keys=True, default=str) + "\n")

    def close(self) -> None:
        self._fh.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def write_verdict(path: str, title: str, body: str) -> None:
    with open(path, "w") as fh:
        fh.write(f"# {title}\n\n{body.rstrip()}\n")
