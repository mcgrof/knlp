"""Loader for the four frozen atlas configs (models / thresholds / datasets / serving). Keeps the
'freeze before the paper tier' discipline honest: one place resolves the YAML, validates the keys
later phases depend on, and fails loudly if a config drifts out of shape. GPU-free."""

import os

import yaml

_DIR = os.path.join(
    os.path.dirname(  # repo root: tools/kv/fp8_failure/configs.py -> up 4
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    ),
    "configs",
    "kv",
)

_FILES = {
    "models": "fp8_failure_models.yaml",
    "thresholds": "fp8_failure_thresholds.yaml",
    "datasets": "fp8_failure_datasets.yaml",
    "serving": "fp8_failure_serving.yaml",
}

# Minimal required top-level keys per config (validated on load so a typo fails fast, not mid-run).
_REQUIRED = {
    "models": ("tiers",),
    "thresholds": ("classification", "recovery", "statistics", "metrics"),
    "datasets": ("calibration", "holdout", "seeds", "smoke"),
    "serving": ("backend", "cells", "models"),
}


def load(name, config_dir=None):
    """Load one config by short name ('models'|'thresholds'|'datasets'|'serving')."""
    if name not in _FILES:
        raise KeyError(f"unknown config {name!r}; known: {sorted(_FILES)}")
    path = os.path.join(config_dir or _DIR, _FILES[name])
    with open(path) as f:
        obj = yaml.safe_load(f)
    missing = [k for k in _REQUIRED[name] if k not in obj]
    if missing:
        raise ValueError(f"config {name!r} missing required keys: {missing}")
    return obj


def load_all(config_dir=None):
    return {n: load(n, config_dir) for n in _FILES}


def iter_models(models_cfg, tiers=("A", "B", "C")):
    """Flatten the tiered registry into a list of model dicts (tier annotated on each)."""
    out = []
    for t in tiers:
        for m in models_cfg["tiers"].get(t, []):
            out.append({**m, "tier": t})
    return out
