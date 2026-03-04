"""Shared utilities for QK Router experiments."""

import json
import os
import time
from pathlib import Path

import yaml


def load_config(config_path: str) -> dict:
    """Load YAML config, merging with base.yaml if not base itself."""
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    config_dir = os.path.dirname(config_path)
    base_path = os.path.join(config_dir, "base.yaml")
    if os.path.basename(config_path) != "base.yaml" and os.path.exists(base_path):
        with open(base_path) as f:
            base = yaml.safe_load(f)
        merged = _deep_merge(base, cfg)
        return merged
    return cfg


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge override into base."""
    result = base.copy()
    for k, v in override.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = _deep_merge(result[k], v)
        else:
            result[k] = v
    return result


def save_json(data, path: str):
    """Save dict/list to JSON with nice formatting."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)


def load_json(path: str):
    """Load JSON file."""
    with open(path) as f:
        return json.load(f)


def ensure_dirs(*paths):
    """Create directories if they don't exist."""
    for p in paths:
        os.makedirs(p, exist_ok=True)


class Timer:
    """Simple context-manager timer."""

    def __init__(self, label=""):
        self.label = label
        self.elapsed_s = 0.0

    def __enter__(self):
        self.start = time.perf_counter()
        return self

    def __exit__(self, *args):
        self.elapsed_s = time.perf_counter() - self.start
        if self.label:
            print(f"[{self.label}] {self.elapsed_s:.3f}s")
