"""Shared helpers for muvera-small benchmark scripts.

Reads the kconfig-style .config produced by `make defconfig-muvera-small`
and exposes per-key access. Also stamps a single run directory under
CONFIG_MUVERA_OUTPUT_BASE that all scripts in one run share, so the
report writer at the end can find every artifact.
"""
from __future__ import annotations

import argparse
import json
import os
import shlex
import socket
import subprocess
import sys
import time
from pathlib import Path

DEFAULT_OUTPUT_BASE = "/data/knlp-key-results"


def _strip_quotes(s):
    if (s.startswith('"') and s.endswith('"')) or (s.startswith("'") and s.endswith("'")):
        return s[1:-1]
    return s


def parse_kconfig(path: Path) -> dict:
    """Parse kconfig-style key=value lines. Strips quotes. Supports y/n/m
    plus integer and string values. Lines starting with # are comments.
    Empty values become "" (not None) so callers can `or` defaults."""
    out = {}
    if not path.exists():
        return out
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        k, v = line.split("=", 1)
        k = k.strip()
        v = _strip_quotes(v.strip())
        out[k] = v
    return out


def get_run_dir(cfg: dict) -> Path:
    """Returns or creates a single shared run dir for this benchmark
    invocation. The first script to touch it stamps a timestamp; later
    scripts reuse the same dir via a sidecar pointer file in /tmp.
    """
    base = cfg.get("CONFIG_MUVERA_OUTPUT_BASE") or DEFAULT_OUTPUT_BASE
    base = Path(base)
    base.mkdir(parents=True, exist_ok=True)

    # Use a tmp pointer keyed by current shell session so multiple `make`
    # targets in one invocation share the same dir. Falls back to the
    # newest matching dir if the pointer is missing.
    pointer = Path("/tmp/.muvera_small_run_dir")
    if pointer.exists():
        try:
            d = Path(pointer.read_text().strip())
            if d.exists() and d.is_dir() and d.parent == base:
                return d
        except Exception:
            pass

    # Stamp a fresh dir
    ts = time.strftime("%Y%m%d-%H%M")
    d = base / f"muvera-small-{ts}"
    d.mkdir(parents=True, exist_ok=True)
    (d / "plots").mkdir(exist_ok=True)
    pointer.write_text(str(d))
    return d


def reset_run_dir():
    """Wipe the shared pointer so the next call to get_run_dir() stamps
    a new directory. Useful in tests."""
    p = Path("/tmp/.muvera_small_run_dir")
    if p.exists():
        p.unlink()


def standard_argparse(prog_help: str) -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=prog_help)
    ap.add_argument("--config", default=".config",
                     help="kconfig-style .config file (default: .config)")
    return ap


def stamp_environment(cfg: dict, run_dir: Path):
    """Write environment.txt + git_commit.txt + config.json on first call.
    Idempotent: only writes if the files don't exist."""
    env_path = run_dir / "environment.txt"
    if not env_path.exists():
        lines = []
        lines.append(f"hostname: {socket.gethostname()}")
        lines.append(f"timestamp: {time.strftime('%Y-%m-%d %H:%M:%S %Z')}")
        try:
            uname = subprocess.check_output(["uname", "-a"], text=True).strip()
            lines.append(f"uname: {uname}")
        except Exception: pass
        try:
            with open("/proc/cpuinfo") as f:
                cpuinfo = f.read()
            model = next((l for l in cpuinfo.splitlines()
                          if l.startswith("model name")), None)
            if model: lines.append(f"cpu: {model.split(':',1)[1].strip()}")
        except Exception: pass
        try:
            mi = subprocess.check_output(["free", "-h"], text=True).strip()
            lines.append("memory:\n  " + mi.replace("\n", "\n  "))
        except Exception: pass
        try:
            import torch
            lines.append(f"torch: {torch.__version__}")
            lines.append(f"cuda available: {torch.cuda.is_available()}")
            if torch.cuda.is_available():
                lines.append(f"  device 0 name: {torch.cuda.get_device_name(0)}")
        except Exception: pass
        # Python pkgs
        try:
            for mod in ["sentence_transformers", "pylate", "faiss", "datasets",
                         "transformers", "numpy", "matplotlib", "psutil"]:
                try:
                    m = __import__(mod)
                    v = getattr(m, "__version__", "installed")
                    lines.append(f"  {mod}: {v}")
                except ImportError:
                    lines.append(f"  {mod}: NOT INSTALLED")
        except Exception: pass
        env_path.write_text("\n".join(lines) + "\n")

    git_path = run_dir / "git_commit.txt"
    if not git_path.exists():
        try:
            sha = subprocess.check_output(
                ["git", "rev-parse", "HEAD"], text=True,
                cwd=Path(__file__).resolve().parents[3]).strip()
            branch = subprocess.check_output(
                ["git", "branch", "--show-current"], text=True,
                cwd=Path(__file__).resolve().parents[3]).strip()
            git_path.write_text(f"commit: {sha}\nbranch: {branch}\n")
        except Exception:
            git_path.write_text("(not a git working tree)\n")

    cfg_path = run_dir / "config.json"
    if not cfg_path.exists():
        cfg_path.write_text(json.dumps(cfg, indent=2, sort_keys=True))


def parse_int_list(s: str) -> list[int]:
    if not s: return []
    return [int(x) for x in s.replace(",", " ").split() if x]


def parse_str_list(s: str) -> list[str]:
    if not s: return []
    return [x for x in s.replace(",", " ").split() if x]


def kbool(cfg: dict, key: str, default: bool = False) -> bool:
    v = cfg.get(key)
    if v is None:
        return default
    return str(v).lower() in ("y", "yes", "true", "1", "on")
